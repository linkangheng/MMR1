import re
import os
import time
from datetime import datetime
from math_verify import parse, verify
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
import math
from open_r1.utils import extract_bbox_answer, compute_iou

def log(content, sol, other_info, reward, tag=None):
    log_path = os.getenv("LOG_PATH", f"/data/ICCV2025/PaR/MMR1/logs/train_{time.strftime('%Y-%m-%d')}.log")
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
        # log(content, sol, None, reward, "accuracy_reward")

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def counting_wo_format_reward(completions, **kwargs):
    """We only need to check if the completion is a number without any other text."""
    pattern = r"^\d+$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward_wo_format(completions, solution, **kwargs):
    """Reward function that checks if the completion is a number without any other text."""
    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    for content, sol in zip(contents, solution):
        try:
            rst = eval(content.strip())
            rewards.append(1.0 if rst == sol else 0.0)
        except:
            rewards.append(0.0)
    return rewards

def answer_format_reward(completions, **kwargs):
    """Reward function that checks if the completion is a valid answer."""
    pattern = r"^<answer>.*?</answer>$"
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

def got_num_click_reward(completions, **kwargs):
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
            # rewards.append( abs(pred_num - gt_num) / max(pred_num, gt_num))
            if pred_num == gt_num:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards

def got_click_reward(completions, solution, **kwargs):
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

def got_sequence_reward(completions, solution, **kwargs):
    # 分阶段验证：格式正确 → 点击次数正确 → 点击位置+答案正确
    format_scores = format_reward(completions)
    num_click_scores = got_num_click_reward(completions)
    click_scores = got_click_reward(completions, solution, **kwargs)
    accuracy_scores = accuracy_reward(completions, solution)
    rewards = []
    for fmt, num, clk, acc in zip(format_scores, num_click_scores, 
                                 click_scores, accuracy_scores):
        score = 0.0
        # 阶段1：格式不正确直接0分
        if fmt < 1.0:
            rewards.append(score)
            continue
        score += 0.25
        # 阶段2：点击次数误差超过阈值则0分
        if num == 0.0:
            rewards.append(score)
            continue
        score += 0.25
        # 阶段3：综合点击位置精度和答案正确性
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

def qwenvl_rec_iou_1_reward(completions, solution, **kwargs):
    rewards = []
    contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    for completion, sol in zip(contents, solution):
        bbox, is_qwen2vl = extract_bbox_answer(completion)
        iou = compute_iou(bbox, eval(sol))
        rewards.append(iou)
        log(completion + f"\nBounding box: {bbox}", sol, None, iou, "qwenvl_rec_iou_reward")
    return rewards