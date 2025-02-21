from open_r1.rewards import *
from open_r1.temperature import *
# REWARD MAPING
reward_funcs_registry = {
    "count_acc": accuracy_reward,
    "count_format": format_reward,
    "perpo_format": perpo_format_reward,
    "perpo_iou": perpo_iou_reward,
    "yjs_grounding": yjs_perpo_reward,
    "perpo_ocr": perpo_ocr_edit_distance_reward,
}

# SYSTEM PROMPTS
GROUNDING_PROMPT = (
    "You are given an image and an object to box in text.You should output the bounding box of the object, which should only be a list of floats."
    "Here is an example of what you should output: [x_min, y_min, x_max, y_max]. Please mind directly to output the list, do not add any other text."
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

OCR_PROMPT = (
    "You are given an image. You should ocr the image and output the text in the image without other output."
)


system_prompt_registry = {
    "default": QWEN2_PROMPT,
    "llava": LLAVA_PROMPT,
    "qwen": QWEN2_PROMPT,
    "counting_reasoning": COUNTING_REASONING_PROMPT,
    "grounding": GROUNDING_PROMPT,
    "ocr": OCR_PROMPT,
}

question_template_registry = {
    "default": "{question}",
    "counting_reasoning": "{question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.",
}
 
answer_template_registry = {
    "default": "{answer}",
    "counting_reasoning": "<answer> {answer} </answer>",
}

temperature_func_registry = {
    "linear": temperature_linear,
    "constant": temperature_constant,
}
