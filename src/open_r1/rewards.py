import re
import os
import time
from datetime import datetime
from math_verify import parse, verify
from rewards_utils import calculate_f_score, merge_lines
import Levenshtein


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
        # if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH", f"/data/ICCV2025/PaR/MMR1/logs/train_{time.strftime('%Y-%m-%d')}.log")
        with open(log_path, "a") as f:
            try:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            except:
                f.write("writing error")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def perpo_format_reward(completions, **kwargs):
    """Reward function that checks if the completion follow the perpo format."""
    matches = []
    for completion in completions:
        try:
            rst = eval(completion.strip())
            matches.append(isinstance(rst, list) and len(rst) == 4)
        except:
            matches.append(False)
    return [1.0 if match else 0.0 for match in matches]

def perpo_iou_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion follow the perpo format."""
    def compute_iou(gt_list, model_answer_list):
        x1 = max(gt_list[0], model_answer_list[0])
        y1 = max(gt_list[1], model_answer_list[1])
        x2 = min(gt_list[2], model_answer_list[2])
        y2 = min(gt_list[3], model_answer_list[3])
        # make sure the model_answer_list is valid
        if model_answer_list[0] > model_answer_list[2] or model_answer_list[1] > model_answer_list[3]:
            return -0.6
        # make sure the model_answer_list is not empty
        if x2 <= x1 or y2 <= y1:
            return -0.2
        # calculate the intersection
        intersection = (x2 - x1) * (y2 - y1)

        gt_area = (gt_list[2] - gt_list[0]) * (gt_list[3] - gt_list[1])
        pred_area = (model_answer_list[2] - model_answer_list[0]) * (model_answer_list[3] - model_answer_list[1])

        union = gt_area + pred_area - intersection

        if union <= 0:
            return 0.0

        iou = intersection / union
        return iou
    
    rewards = [0.0 for _ in completions]
    for i, (completion, sol) in enumerate(zip(completions, solution)):
        if perpo_format_reward(completion) == 0.0:
            continue
        rst = eval(completion.strip())
        gt = eval(sol.strip())
        rewards[i] = compute_iou(rst, gt)
    return rewards


def slowper_format_reward(completions, **kwargs):
    """Reward function that checks if the completion follow the perpo format."""
    matches = []
    for completion in completions:
        try:
            # rst = eval(completion.strip())
            # matches.append(isinstance(rst, list) and len(rst) == 4)
            matches.append("Line" in completion and "--" in completion)
        except:
            matches.append(False)
    return [1.0 if match else 0.0 for match in matches]

def yjs_perpo_reward(completions, solution, **kwargs):
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
    contents = [completion[0]["content"] for completion in completions]
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
            rewards.append(iou**2)
        except:
            rewards.append(-1.0)
    return rewards



def log(content, sol, other_info, reward, tag=None):
    log_path = os.getenv("LOG_PATH", f"./checkpoints/logs/train_{time.strftime('%Y-%m-%d')}.log")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    # with open(log_path, "a") as f:
    #     try:
    #         print('write')
    #         f.write(f"------------- {current_time} {tag} reward: {reward} -------------\n")
    #         f.write(f"Content: {content}\n")
    #         f.write(f"Solution: {sol}\n")
    #         if other_info is not None:
    #             f.write(f"Other info: {other_info}\n")
            
    #     except:
    #         f.write("writeing error")
    #     f.flush()
    with open(log_path, "a") as f:
        try:
            f.write(f"------------- {current_time} {tag} reward: {reward} -------------\n")
            f.flush()  # 立即刷新
            f.write(f"Content: {content}\n")
            f.flush()
            f.write(f"Solution: {sol}\n")
            f.flush()
            if other_info is not None:
                f.write(f"Other info: {other_info}\n")
                f.flush()
        except:
            f.write("writing error")
            f.flush()

def slowper_f1_reward(completions, solution, **kwargs):
    # parse the completion and solution
    # computer the iou
    
    
    def dot_parser(outputs_think):
        if 'Circle' not in outputs_think:
            outputs_think_p = outputs_think.split('Line:\n')[1]
            if outputs_think_p[-1] == '\n':
                outputs_think_p = outputs_think_p[:-1]
            # outputs_think_c_list = []
            outputs_think_c = ''
        else:
            outputs_think_p = outputs_think.split('Circle:\n')[0].split('Line:\n')[1]
            if outputs_think_p[-1] == '\n':
                outputs_think_p = outputs_think_p[:-1]
            # todo: if there is no circle
            if len(outputs_think.split('Line:\n')[1].split('Circle:\n'))<2:
                outputs_think_c = ''
            else:
                outputs_think_c = outputs_think.split('Line:\n')[1].split('Circle:\n')[1]
                if outputs_think_c[-1] == '\n':
                    outputs_think_c = outputs_think_c[:-1]

        outputs_think_p_list = outputs_think_p.split('\n')
        outputs_think_c_list = outputs_think_c.split('\n')
        # print(outputs_think_p_list)
        # fig, ax = plt.subplots(figsize=(10,10))
        # ax.set_xlim(-10, 10)
        # ax.set_ylim(-10, 10)

        line_list = []
        for p in outputs_think_p_list:
            # try:
            if '--' in p:
                p0 = eval(p.split(' -- ')[0])
                p1 = eval(p.split(' -- ')[-1])
                # ax.plot([p0[0], p1[0]], [p0[1], p1[1]])
                # # print([p0, p1])
                # ax.scatter(p0[0], p0[1], s=10)
                # ax.scatter(p1[0], p1[1], s=10)

                line_list.append(((p0[0], p0[1]), (p1[0], p1[1])))


        # out_dict[name] = line_list

        # return out_dict
        return line_list


    def compute_iou(pred, gt):
        new_lines = merge_lines(pred)

        # f_score, precision, recall = calculate_f_score(new_lines, gt[key], iou_threshold = 0.75)
        f_score, precision, recall = calculate_f_score(new_lines, gt, iou_threshold = 0.75)

        # todo: short and long line rewards

        return f_score
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = [0.0 for _ in contents]
    for i, (completion, sol) in enumerate(zip(contents, solution)):
        print(completion)
        try:
            rst = dot_parser(completion.strip())
        except:
            print('invalid output!')
            f.write('invalid output!')
            continue
        # if slowper_format_reward(completion)[i] == 0.0:
        #     continue
        # import ipdb;ipdb.set_trace()
        # print(completion)

        # rst = dot_parser(eval(completion.strip()))
        # gt = dot_parser(eval(sol.strip()))


        # rst = dot_parser(completion.strip())
        gt = dot_parser(sol.strip())
        rewards[i] = compute_iou(rst, gt)
        log(completion, sol, None, rewards[i], tag=None)
    return rewards

def slowper_ed_reward(completions, solution, **kwargs):
    def compute_edit_distance(pred_text, gt_text):
        """
        计算基于编辑距离的 Reward。
        """
        # 计算 Levenshtein 编辑距离
        edit_distance = Levenshtein.distance(pred_text, gt_text)

        # 计算归一化相似度
        max_len = max(len(pred_text), len(gt_text))
        similarity_score = 1 - (edit_distance / max_len) if max_len > 0 else 0

        return similarity_score
        # # 负编辑距离 Reward
        # neg_edit_distance_reward = -edit_distance

        # return similarity_score, neg_edit_distance_reward


    contents = [completion[0]["content"] for completion in completions]
    rewards = [0.0 for _ in contents]
    for i, (completion, sol) in enumerate(zip(contents, solution)):
        print(completion)
        rewards[i] = compute_edit_distance(completion.strip(), sol.strip())
        log(completion, sol, None, rewards[i], tag=None)
        return rewards


