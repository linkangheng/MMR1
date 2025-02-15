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

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

GROUNDING_PROMPT = (
    "You are given an image and an order. You should output a list of floats, which means the bounding box of the object in the image without other output."
    "Here is an example: [x_min, y_min, x_max, y_max]. Please mind directly to output the list, do not add any other text."
)

OCR_PROMPT = (
    "You are given an image. You should ocr the image and output the text in the image without other output."
)

