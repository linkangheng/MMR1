from open_r1.rewards import *

# REWARD MAPING
reward_funcs_registry = {
    "count_acc": accuracy_reward,
    "count_format": format_reward,
    "perpo_format": perpo_format_reward,
    "perpo_iou": perpo_iou_reward,
    "yjs" : yjs_perpo_reward,
    "answer_format": answer_format_reward,
    "counting_wo_format": counting_wo_format_reward,
    "count_acc_wo_format": accuracy_reward_wo_format,
    "got_format": format_reward,
    "got_accuracy": accuracy_reward,
    "got_click": got_click_reward,
    "got_num_click": got_num_click_reward,
    "got_sequence": got_sequence_reward,
    "qwenvl_rec_format": qwenvl_rec_format_reward,
    "qwenvl_rec_iou": qwenvl_rec_iou_reward,
    "qwenvl_rec_iou_1": qwenvl_rec_iou_1_reward,
}

# SYSTEM PROMPTS
GROUNDING_PROMPT = (
    "When describing images, always specify object locations using bounding box coordinates [x1,y1,x2,y2]. "
    "First analyze the image, then enclose your reasoning in <think> </think> tags and provide coordinates in <answer> </answer>."
)
 
LLAVA_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

QWEN2_PROMPT = (
    "You are a helpful assistant. "
)

COUNTING_REASONING_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

GOT_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first Output center coordinates of specified objects as (x,y) pairs in <think> tags, the coordinate range is [0,1000], "
    "then provide the total count in <answer> tags, respectively, i.e., "
    "<think> (x1,y1),(x2,y2)...(xn,yn) </think><answer> n </answer>"
)

COUNTING_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The direct answer is enclosed within <answer> </answer> tags, i.e., "
    "<answer> answer here </answer>"
)

system_prompt_registry = {
    "default": QWEN2_PROMPT,
    "llava": LLAVA_PROMPT,
    "qwen": QWEN2_PROMPT,
    "counting_reasoning": COUNTING_REASONING_PROMPT,
    "counting": COUNTING_PROMPT,
    "counting_got": GOT_SYSTEM_PROMPT,
}

question_template_registry = {
    "default": "{question}",
    "counting_reasoning": "{question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.",
    "counting": "{question} Output the final answer (number) in <answer> </answer> tags.",
    "counting_wo_format": "{question} Directly output the final answer (number) without any other text.",
    "counting_got": "{question} follow the format <think> (x1, y1),(x2, y2)...(xn, yn) </think><answer> n </answer>",
    "qwenvl_rec_format": "Output the bounding box of the {question} in the image.",
}
 
answer_template_registry = {
    "default": "{answer}",
    "counting_reasoning": "<answer> {answer} </answer>",
    "counting": "<answer> {answer} </answer>",
    "counting_wo_format": "{answer}",
    "counting_got": "<answer> {answer} </answer>",
    "qwenvl_rec_format": "{answer}",
}