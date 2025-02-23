import re
import os
import torch
import jieba
import numpy as np
import math
import warnings
from datetime import datetime
from math_verify import parse, verify
from scipy.optimize import linear_sum_assignment
import nltk
from nltk.translate import meteor_score
from nltk.metrics import precision, recall, f_measure
from open_r1.utils import extract_bbox_answer, compute_iou

def log(content, sol, other_info, reward, tag=None):
    log_dir = os.getenv("LOG_DIR", None)
    os.makedirs(log_dir, exist_ok=True)
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
                for k, v in other_info.items():
                    f.write(f"{k}: {v}\n")
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
    """<think>...</think><answer>...</answer>"""
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
    return rewards

def gat_num_click_reward(completions, **kwargs):
    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    for content in contents:
        try:
            content_match = re.search(r'<think>(.*?)</think>', content)
            answer_match = re.search(r'<answer>(.*?)</answer>', content)
            clicks_pred = content_match.group(1).strip() if content_match else content.strip()
            answer_pred = answer_match.group(1).strip() if answer_match else content.strip()
            pred_num = len(eval(clicks_pred.strip()))
            gt_num = eval(answer_pred.strip())
            if pred_num == gt_num:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards

def gat_click_reward(completions, solution, **kwargs):
    # Function to match the points
    def match_and_compute_distances(pred, gt, return_indices=False):
        """
        Matches points from pred to gt based on the minimum L2 distance using the Hungarian algorithm.
        Returns the distances between the matched points. Optionally returns the indices of matched points.

        Parameters:
        - pred: List of prediction points (list of tuples)
        - gt: List of ground truth points (list of tuples)
        - return_indices: Whether to return the indices of matched points (default: False)

        Returns:
        - distances: Tensor of distances between matched points
        - (row_ind, col_ind): Indices of matched points (if return_indices is True)
        """
        pred_tensor = torch.tensor(pred, dtype=torch.float32)
        gt_tensor = torch.tensor(gt, dtype=torch.float32)
        
        cost_matrix = torch.cdist(pred_tensor, gt_tensor, p=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
        
        # Extract matched points
        matched_pred = pred_tensor[torch.from_numpy(row_ind)]
        matched_gt = gt_tensor[torch.from_numpy(col_ind)]
        
        # Compute distances
        distances = torch.norm(matched_pred - matched_gt, p=2, dim=1)
        
        if return_indices:
            return distances, (torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64))
        else:
            return distances
    
    rewards = []
    clicks_gt = kwargs['clicks']
    contents = [completion[0]["content"] for completion in completions]
    for content, click_gt in zip(contents, clicks_gt):
        try:
            content_match = re.search(r'<think>(.*?)</think>', content)
            clicks_pred = eval(f"[{content_match.group(1).strip() if content_match else content.strip()}]")
            distances = match_and_compute_distances(clicks_pred, click_gt)
            scores = [ (1 - distance / 1000 * math.sqrt(2)).item() for distance in distances]
            rewards.append(np.mean(scores))
        except:
            rewards.append(0.0)
    return rewards

def gat_sequence_reward(completions, solution, **kwargs):
    format_scores = format_reward(completions)
    num_click_scores = gat_num_click_reward(completions)
    click_scores = gat_click_reward(completions, solution, **kwargs)
    accuracy_scores = accuracy_reward(completions, solution)
    rewards = []
    for fmt, num, clk, acc in zip(format_scores, num_click_scores, 
                                 click_scores, accuracy_scores):
        score = 0.0
        # stage1: format is correct?
        if fmt < 1.0:
            rewards.append(score)
            continue
        score += 0.25
        # stage2: len(clicks) == answer
        if num == 0.0:
            rewards.append(score)
            continue
        score += 0.25
        # stage3: accuracy of clicks and answer
        score += 0.7 * clk + 0.3 * acc
        rewards.append(score)
    contents = [completion[0]["content"] for completion in completions]
    for completion, sol, reward in zip(contents, kwargs['clicks'], rewards):
        log(completion, sol, None, reward, "got_sequence_reward")
    return rewards

def qwenvl_rec_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<|object_ref_start|>.*?<|object_ref_end|><|box_start|>.*?<|box_end|><|im_end|>"
    completion_contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def qwenvl_rec_iou_reward(completions, solution, **kwargs):
    rewards = []
    contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    for completion, sol in zip(contents, solution):
        bbox, is_qwen2vl = extract_bbox_answer(completion)
        iou = compute_iou(bbox, eval(sol))
        rewards.append(iou**2)
        log(completion + f"\nBounding box: {bbox}", sol, None, iou**2, "qwenvl_rec_iou_reward")
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