from open_r1.rewards import *
# REWARD MAPING
reward_funcs_registry = {
    # "count_acc": accuracy_reward,
    "count_format": format_reward,
    "perpo_format": perpo_format_reward,
    "perpo_iou": perpo_iou_reward,
    "yjs": yjs_perpo_reward,
    "slowper_format": slowper_format_reward,
    "slowper_f1": slowper_f1_reward,
    "slowper_ed": slowper_ed_reward,
    "perpo_iou": perpo_iou_reward,
    "r1v_acc": accuracy_reward,
    "r1v_format": think_format_reward,
    "perpo_reward": perpo_reward,
    "perpo_ocr_edit_distance_reward": perpo_ocr_edit_distance_reward,
}

# SYSTEM PROMPTS
LLAVA_SYS = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

QWEN2_SYS = (
    "You are a helpful assistant. "
)

R1V_SYS = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

SLOWPER_PROMPT = (
"""You are an expert in geometric shape recognition. Given an image, your task is to accurately describe all the line segments and circles present in the image. 

For each **line segment**, provide its two endpoints. For each **circle**, provide its center coordinates and radius. The output format should be like:  
Line: 
(x1, y1) -- (x2, y2)
(x3, y3) -- (x4, y4)

Circle: (cx, cy, r)
"""
)

OCR_SYS = (
    "You are given an image. You should ocr the image and output the text in the image without other output."
)

system_prompt_registry = {
    "default": QWEN2_SYS,
    "llava": LLAVA_SYS,
    "qwen": QWEN2_SYS,
    "r1v": R1V_SYS,
    "ocr": OCR_SYS,
    # "slow_perception": SLOWPER_PROMPT
    "slow_perception": ''
}




question_template_registry = {
    "default": "{question}",
    "slow_perception": "",
    "counting_reasoning": "{question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.",
    "r1v": "{question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.",
}
 
answer_template_registry = {
    "default": "{answer}",
    "r1v": "<answer> {answer} </answer>",
}