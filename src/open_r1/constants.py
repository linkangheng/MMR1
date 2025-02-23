from open_r1.rewards import *

# REWARD MAPING
reward_funcs_registry = {
    # R1-V task
    "r1v_acc": accuracy_reward,
    "r1v_format": think_format_reward,
    # Grouding as thought task
    "gat_format": format_reward,
    "gat_accuracy": accuracy_reward,
    "gat_click": gat_click_reward,
    "gat_num_click": gat_num_click_reward,
    "gat_sequence": gat_sequence_reward,
    # Grounding task
    "qwenvl_rec_format": qwenvl_rec_format_reward,
    "qwenvl_rec_iou": qwenvl_rec_iou_reward,
    # PerPO task
    "perpo_reward": perpo_reward,
    # OCR task
    "perpo_ocr_edit_distance_reward": perpo_ocr_edit_distance_reward,
    # Slow percetion task
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

GAT_SYS = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first Output center coordinates of specified objects as (x,y) pairs in <think> tags, the coordinate range is [0,1000], "
    "then provide the total count in <answer> tags, respectively, i.e., "
    "<think> (x1,y1),(x2,y2)...(xn,yn) </think><answer> n </answer>"
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
    "gat": GAT_SYS,
}

question_template_registry = {
    "default": "{question}",
    "r1v": "{question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.",
    "gat": "{question} follow the format <think> (x1, y1),(x2, y2)...(xn, yn) </think><answer> n </answer>",
    "qwenvl_rec_format": "Output the bounding box of the {question} in the image.",
}
 
answer_template_registry = {
    "default": "{answer}",
    "r1v": "<answer> {answer} </answer>",
    "gat": "<answer> {answer} </answer>",
    "qwenvl_rec_format": "{answer}",
}