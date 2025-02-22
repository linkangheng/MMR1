import re
import os
from datetime import datetime
from math_verify import parse, verify
import nltk
import jieba
from nltk.translate import meteor_score
from nltk.metrics import precision, recall, f_measure
import warnings

def log(content, sol, other_info, reward, tag=None):
    log_dir = os.getenv("LOG_DIR", None)
    if log_dir is None:
        warnings.warn("LOG_DIR is not set, log will not be saved")
        return
    log_path = os.path.join(log_dir, f"{tag}.log")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    with open(log_path, "a") as f:
        try:
            f.write(f"------------- {current_time} {tag} reward: {reward} -------------\n")
            f.write(f"Content: {content}\n")
            f.write(f"Solution: {sol}\n")
            if other_info is not None:
                f.write(f"Other info: {other_info}\n")
        except:
            f.write("writeing error")

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        log(content, sol, None, reward, "rec_acc")
    return rewards

def format_reward(completions, pattern, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def think_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return format_reward(completions, pattern)

def perpo_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion follow the perpo format."""
    def compute_iou(gt_list, model_answer_list):
        # Calculate intersection over union (IoU) between two bounding boxes
        # Get intersection coordinates
        x1 = max(gt_list[0], model_answer_list[0])  # 左上角x坐标的最大值
        y1 = max(gt_list[1], model_answer_list[1])  # 左上角y坐标的最大值  
        x2 = min(gt_list[2], model_answer_list[2])  # 右下角x坐标的最小值
        y2 = min(gt_list[3], model_answer_list[3])  # 右下角y坐标的最小值

        # 计算交集面积
        if model_answer_list[0] > model_answer_list[2] or model_answer_list[1] > model_answer_list[3]:
            return -0.6 # 如果预测框的坐标不合法，则返回-0.6
        if x2 <= x1 or y2 <= y1:  # 如果没有重叠区域
            return -0.2

        intersection = (x2 - x1) * (y2 - y1)

        # 计算两个框的面积
        gt_area = (gt_list[2] - gt_list[0]) * (gt_list[3] - gt_list[1])  # 真实框面积
        pred_area = (model_answer_list[2] - model_answer_list[0]) * (model_answer_list[3] - model_answer_list[1])  # 预测框面积

        # 计算并集面积
        union = gt_area + pred_area - intersection

        # 计算IoU
        if union <= 0:  # 避免除0错误
            return 0.0

        iou = intersection / union
        return iou

    rewards = []
    if 'content' in completions[0][0]:
        contents = [completion[0]["content"] for completion in completions]
    else:
        contents = completions
    for completion, sol in zip(contents, solution):
        try:
            gt_list = eval(sol.strip())
            gt_list = [float(i) for i in gt_list]

            model_answer_text = completion.strip()
            if model_answer_text.strip().startswith("[") and model_answer_text.strip().endswith("]"): 
                model_answer_list = eval(model_answer_text.strip())
                model_answer_list = [float(i) for i in model_answer_list]
            else:
                rewards.append(-1.0)
                continue
            iou = compute_iou(gt_list, model_answer_list)
            if iou <= 0:
                rewards.append(iou)
            else:
                rewards.append(iou**2)
        except:
            rewards.append(-1.0)
        log(completion, sol, None, rewards[-1], "rec_iou")
    return rewards

def perpo_ocr_edit_distance_reward(prompts, completions, solution, **kwargs):
    def contain_chinese_string(text):
        chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
        return bool(chinese_pattern.search(text))
    def cal_per_metrics(pred, gt):
        metrics = {}
        
        if contain_chinese_string(gt) or contain_chinese_string(pred):
            reference = jieba.lcut(gt)
            hypothesis = jieba.lcut(pred)
        else:
            reference = gt.split()
            hypothesis = pred.split()
        
        metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
        metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

        reference = set(reference)
        hypothesis = set(hypothesis)
        metrics["f_measure"] = f_measure(reference, hypothesis)

        metrics["precision"] = precision(reference, hypothesis)
        metrics["recall"] = recall(reference, hypothesis)
        metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
        return metrics

    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    
    for completion, sol in zip(contents, solution):
        try:
            # Normalize strings by removing extra whitespace
            completion_text = " ".join(completion.strip().split())
            solution_text = " ".join(sol.strip().split())
            
            # Calculate edit distance
            edit_dist = cal_per_metrics(completion_text, solution_text)['edit_dist']
            
            # Convert edit distance to reward (higher for smaller distances)
            max_len = max(len(completion_text), len(solution_text))
            if max_len == 0:
                reward = 0.0
            else:
                # Normalize by max length and convert to reward between 0 and 1
                normalized_dist = 1 - edit_dist
                reward = max(0.0, normalized_dist)

        except Exception as e:
            print(f"Error in perpo_ocr_edit_distance_reward: {e}")
            with open('./perpo_ocr_edit_distance_reward_error.txt', "a") as f:
                try:
                    f.write(f"Prompt: {prompts}\n")
                    f.write(f"Content: {completion}\n")
                    f.write(f"Solution: {sol}\n")
                except:
                    f.write("writeing error")
            print(f"Error in perpo_ocr_edit_distance_reward: {completion} {sol}")
            reward = 0.0
        rewards.append(reward)
        log(completion, sol, None, reward, "rec_edit_dist")
    return rewards