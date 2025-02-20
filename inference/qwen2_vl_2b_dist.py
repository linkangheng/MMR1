from transformers import Qwen2VLForConditionalGeneration, LlavaForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
import datetime
import math
import argparse
import numpy as np
from collections import defaultdict
from src.open_r1.rewards import yjs_perpo_reward
EVAL_ROOT="/data/ICCV2025/PaR/MMR1/eval"

choises = [
    # "pixmo_count_test540_counting_problems",
    # "pixmo_count_validation540_counting_problems",
    # "superclevr_test200_counting_problems",
    "countbenchqa_test491_counting_problems"
]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def extract_number_answer(output_str):
    # Try to find the number within <answer> tags, if can not find, return None
    answer_pattern = r'<answer>\s*(\d+)\s*</answer>'
    match = re.search(answer_pattern, output_str)

    if match:
        return int(match.group(1))
    return None

def str_map_number(output_str):
    str_map_number = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17}
    for k, v in str_map_number.items():
        if k in output_str:
            return v
    return None

def extract_number_answer_baseline(output_str):
    answer_pattern = r'\d+'
    match = re.search(answer_pattern, output_str)
    if match:
        return int(match.group(0))
    else:
        return str_map_number(output_str)

def merge_pass_n(source):
    merged = defaultdict(
        lambda: {
            "question": None,
            "ground_truth": None,
            "extracted_answer": [],
            "original_output": [],
            "model_output": []
        })

    for item in source:
        question = item["question"]
        image_path = question["image_path"]
        ground_truth = item["ground_truth"]
        extracted_answer = item["extracted_answer"]
        if not question["question"]:
            continue
        merged[image_path]["question"] = question
        merged[image_path]["ground_truth"] = ground_truth
        merged[image_path]["extracted_answer"].append(extracted_answer)
        merged[image_path]["original_output"].append(item["original_output"])
        merged[image_path]["model_output"].append(item["model_output"])

    return [{"question": value["question"], "ground_truth": value["ground_truth"], "extracted_answer": value["extracted_answer"], "original_output": value["original_output"], "model_output": value["model_output"]} for value in merged.values()]

def eval(args):
    for task in choises:
        fmt=f"/tmp/{task}_{os.path.basename(args.model_path)}_{datetime.datetime.now().strftime('%m%d')}_pass@{args.pass_n}"
        # fmt=f"/tmp/{task}_{os.path.basename(args.model_path)}_0218_pass@{args.pass_n}"
        files = [f"{fmt}_rank{str(i)}.json" for i in range(0, int(args.world_size))]
        # 将 files 全加载到 json里
        data = []
        for file in files:
            with open(file, "r") as f:
                data.extend(json.load(f))
        data = merge_pass_n(data)
        correct_number = 0
        for item in data:
            extracted_answer = item['extracted_answer']
            ground_truth = item['ground_truth']
            if  ground_truth in extracted_answer:
                correct_number += 1
        accuracy = correct_number / len(data) * 100
        with open(fmt.replace("/tmp/", "/data/ICCV2025/PaR/MMR1/eval/logs/") + f"{datetime.datetime.now().strftime('%H%M')}.json", "w") as f:
            json.dump({
                "task": task,
                "accuracy": accuracy,
                "data": data
            }, f, indent=2)
        print(f"{task} accuracy:\n {accuracy:.2f}%")
    return accuracy

def main(args):
    # load_models 
    if 'qwen' in args.model_path.lower():
        generator = Qwen2VLForConditionalGeneration
    elif 'llava' in args.model_path.lower():
        generator = LlavaForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = generator.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # main evaluation
    results = {k: None for k in choises}
    for TASK in choises:
        PROMPT_PATH=f"{EVAL_ROOT}/prompts/{TASK}.jsonl"
        OUTPUT_PATH=f"/tmp/{TASK}_{os.path.basename(args.model_path)}_{datetime.datetime.now().strftime('%m%d')}_pass@{args.pass_n}_rank{args.rank}.json"

        data = []
        with open(PROMPT_PATH, "r") as f:
            for line in f:
                data.append(json.loads(line))
        data = get_chunk(data, int(args.world_size), int(args.rank))
        # prepare for pass@n
        if args.pass_n > 1:
            data = np.repeat(data, args.pass_n)

        # QUESTION_TEMPLATE = "{question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
        QUESTION_TEMPLATE = "{question} Output the final answer (number) in <answer> </answer> tags."
        messages = []

        for i in data:
            message = [{
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": f"file://{os.path.join('/mnt/jfs-test/data/clevr_cogen_a_eval', i['image_path'])}"
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(question=i['question'])
                    }
                ]
            }]
            messages.append(message)


        all_outputs = []  # List to store all answers

        # Process data in batches
        for i in tqdm(range(0, len(messages), args.batchsize), desc="Processing batches"):
            batch_messages = messages[i:i + args.batchsize]

            # Preparation for inference
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]

            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(
                **inputs, 
                use_cache=True, 
                max_new_tokens=256, 
                do_sample=True if float(args.temperature) > 0 else False, 
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=10
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            all_outputs.extend(batch_output_text)

        final_output = []

        for input_example, model_output in zip(data,all_outputs):
            original_output = model_output
            ground_truth = input_example['ground_truth']
            # extrac_func = extract_number_answer_baseline if args.baseline else extract_number_answer
            extrac_func = yjs_perpo_reward
            reward = extrac_func(original_output)

            # Create a result dictionary for this example
            result = {
                'problem': input_example,
                'solution': ground_truth,
                'completion': original_output,
                'reward': reward,
            }
            final_output.append(result)

        # save final_output
        with open(OUTPUT_PATH, "w") as f:
            json.dump(final_output, f, indent=4)

if __name__ == "__main__":
    # load args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--rank", type=str, default=0)
    parser.add_argument("--world_size", type=str, default=1)
    parser.add_argument("--pass_n", type=int, default=1)
    parser.add_argument("--stage", type=str, default=0)
    parser.add_argument("--temperature", type=str, default=0)
    parser.add_argument("--top_p", type=str, default=0)
    parser.add_argument("--baseline", type=bool, default=False)
    parser.add_argument("--batchsize", type=int, default=64)
    args = parser.parse_args()

    if args.stage == "infer":
        main(args)
    elif args.stage == "eval":
        eval(args)

