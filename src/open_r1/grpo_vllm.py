# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import load_dataset, load_from_disk
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from rewards import *
from prompts import *
from temperature import *

from trainer.grpo_trainer_vllm import Qwen2VLGRPOTrainer
import time

import json
import wandb

# wandb.init(project="R1-multimodal", name="Qwen2_5_7B_R1-multimodal")



from arguments import GRPOScriptArguments




reward_funcs_registry = {
    "count_acc": accuracy_reward,
    "count_format": format_reward,
    "perpo_format": perpo_format_reward,
    "perpo_iou": perpo_iou_reward,
    "yjs_grounding": yjs_perpo_reward,
    "yjs_ocr": yjs_perpo_ocr_reward,
}

prompt_registry = {
    "llava": LLAVA_PROMPT,
    "qwen": QWEN2_PROMPT,
    "reasoning": SYSTEM_PROMPT,
    "grounding": GROUNDING_PROMPT, 
    "ocr": OCR_PROMPT,
}

temperature_func_registry = {
    "linear": temperature_linear, 
    "const": temperature_const, 
}

order_dataset_registry = {
    "llava1.5-7b_easy2diff": '_easy2diff_llava1.57b', 
    "qwen22b_easy2diff": '_easy2diff_qwen22b', 
    "random": '', 
}

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # Get temperature function
    temperature_func = temperature_func_registry[script_args.temperature_func]
    # modify some hyperparameters
    if script_args.kl_approximator == 'fullkimi':
        training_args.sync_ref_model = True
        training_args.ref_model_mixup_alpha = 1.0
        training_args.ref_model_sync_steps = 1

    
    # save args to output_dir
    save_args_to_txt(script_args, os.path.join(training_args.output_dir, 'script_args.txt'))
    save_args_to_txt(training_args, os.path.join(training_args.output_dir, 'training_args.txt'))
    save_args_to_txt(model_args, os.path.join(training_args.output_dir, 'model_args.txt'))



    # Load the dataset
    try:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        if "image" not in dataset[script_args.dataset_train_split].features:
            raise ValueError("Some bugs happens when loading the dataset.. Plase check the hf-dataset is correct created.")
    except:
        dataset = load_from_disk(script_args.dataset_name+order_dataset_registry[script_args.order_dataset])

    # Format into conversation
    system_prompt = prompt_registry[script_args.prompt_template]
    def make_conversation(example):
        return {
            "prompt": json.dumps([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["problem"]},
            ])
        }

    def make_conversation_image(example):
        # only for reasoning task
        # QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

        # general template
        QUESTION_TEMPLATE = "{Question}"
        question = example["problem"]
        question = question + "." if not question.endswith(".") else question
        return {
            "prompt": json.dumps([
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question)},
                    ],
                },
            ]),
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image, num_proc=64)
    else:
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        script_args=script_args, 
        temperature_func=temperature_func
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

def save_args_to_txt(args, filename):
    """
    将 argparse 解析的参数保存到 txt 文件中
    :param args: argparse.Namespace 对象，包含解析后的参数
    :param filename: 要保存的文件名
    """
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print('training_args:\n', training_args)
    print('script_args:\n', script_args)
    print('model_args:\n', model_args)
    main(script_args, training_args, model_args)
