from open_r1.rewards import *

# REWARD MAPING
reward_funcs_registry = {
    "count_acc": accuracy_reward,
    "count_format": format_reward,
    "perpo_format": perpo_format_reward,
    "perpo_iou": perpo_iou_reward,
    "answer_format": answer_format_reward,
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
}

question_template_registry = {
    "default": "{question}",
    "counting_reasoning": "{question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.",
    "counting": "{question} Output the final answer (number) in <answer> </answer> tags.",
}
 
answer_template_registry = {
    "default": "{answer}",
    "counting_reasoning": "<answer> {answer} </answer>",
    "counting": "<answer> {answer} </answer>",
}