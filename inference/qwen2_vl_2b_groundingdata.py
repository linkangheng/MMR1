import os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from open_r1.rewards import *
from open_r1.constants import *
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--world_size", type=str, default=1)
parser.add_argument("--rank", type=str, default=0)
args = parser.parse_args()

model_id = "/mnt/jfs-test/models/Qwen2-VL-2B-Instruct"
# model_id = "/mnt/jfs-test/checkpoints/mmr1/debug/qwen2-vl-2b_vllm_lucas_counting_molmo10k_roll7"
dataset_path = "/data/LLaVA/llava/ppo_vl/ppo_datasets/ppo_perpo_train_dataset.json"
output_data_path = "/mnt/jfs-test/data/grounding_crouse_learn_3k.json"
with open(dataset_path, "r") as f:
    data = json.load(f)
#     data = data[args.rank*args.world_size:(args.rank+1)*args.world_size]
# def split_list(lst, n):
#     """Split a list into n (roughly) equal-sized chunks"""
#     chunk_size = len(lst) // n + (1 if len(lst) % n else 0)
#     return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

# data = split_list(data, int(args.world_size))[int(args.rank)]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

for i in tqdm(range(len(data))):
    image_path =  data[i]["image"]
    prompt = data[i]['question']
    solution = data[i]['answer']

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path, 
                },
                {"type": "text", "text": "You are given an image and an object to box in text.You should output the bounding box of the object, which should only be a list of floats. \nHere is an example of what you should output: [x_min, y_min, x_max, y_max]. Please mind directly to output the list, do not add any other text."},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
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

    data[i]['reward'] = yjs_perpo_reward(completions=output_text, solution=[solution])
    data[i]['problem'] = prompt
    data[i]['solution'] = solution
    data[i]['completion'] = output_text[0]
    del data[i]['answer']
    del data[i]['question']
    del data[i]['answers_gen']
    del data[i]['image_id']
    del data[i]['image_size']
    del data[i]['id']
    del data[i]['iou_scores']

    # Sort data list by reward value in descending order
data.sort(key=lambda x: x['reward'][0], reverse=True)

with open(output_data_path, "w") as f:
    f.write(json.dumps(data, indent=4))
