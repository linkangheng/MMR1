import os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load the model on the available device(s)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_id = "/mnt/jfs-test/models/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# prepare inputs
image_path = "<replace to your image path>" 
question = "<replace to your question>"
messages = [
    {
        "role": "user",
        "content": [
            
            {
                "type": "image",
                "image": f"file://{image_path}",
            },
            {"type": "text", "text": f"{question}"},
        ],
    }
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(
    **inputs,
    do_sample=True,
    temperature=1,
    top_k = 10,
    top_p = 0.7,
    max_new_tokens=64)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)