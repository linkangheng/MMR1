from open_r1.rewards import *
# REWARD MAPING
reward_funcs_registry = {
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

OCR_SYS = (
    "You are given an image. You should ocr the image and output the text in the image without other output."
)

system_prompt_registry = {
    "default": QWEN2_SYS,
    "llava": LLAVA_SYS,
    "qwen": QWEN2_SYS,
    "r1v": R1V_SYS,
    "ocr": OCR_SYS,
}

question_template_registry = {
    "default": "{question}",
    "r1v": "{question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.",
}
 
answer_template_registry = {
    "default": "{answer}",
    "r1v": "<answer> {answer} </answer>",
}