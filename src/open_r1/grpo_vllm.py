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
from datasets import load_dataset, load_from_disk, Image
from trl import GRPOConfig, ModelConfig, TrlParser, get_peft_config
from open_r1.constants import reward_funcs_registry, system_prompt_registry, question_template_registry, answer_template_registry
from open_r1.utils import *
from open_r1.trainer.grpo_trainer_vllm import Qwen2VLGRPOTrainer
from open_r1.arguments import GRPOScriptArguments

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
    # prepare parameters for Kimi-KL
    if script_args.kl_approximator == 'fullkimi':
        script_args.use_kl = True
        training_args.sync_ref_model = True
        training_args.ref_model_mixup_alpha = 1.0
        training_args.ref_model_sync_steps = 1
    
    # save args to output_dir
    save_args_to_txt(script_args, os.path.join(training_args.output_dir, 'script_args.txt'))
    save_args_to_txt(training_args, os.path.join(training_args.output_dir, 'training_args.txt'))
    save_args_to_txt(model_args, os.path.join(training_args.output_dir, 'model_args.txt'))

    # Load the dataset
    if "json" in script_args.dataset_name:
        # json dataset
        dataset = load_dataset('json',data_files=script_args.dataset_name,)
    else:
        try:
             
            dataset = load_dataset(script_args.dataset_name)
            if "image" not in dataset[script_args.dataset_train_split].features:
                raise ValueError("The dataset is created locally.")
        except:
            # hf-local dataset
            dataset = load_from_disk(script_args.dataset_name)
    
    # sample from trainset
    if script_args.train_sample_size is not None:
        dataset[script_args.dataset_train_split] = dataset['train'].select(range(script_args.train_sample_size))

    # Format into conversation
    system_prompt = system_prompt_registry[script_args.system_prompt_template]
    question_template = question_template_registry[script_args.question_template]
    answer_template = answer_template_registry[script_args.answer_template]
    map_func = json_map if "json" in script_args.dataset_name else make_conversation_image
    dataset = dataset.map(
        map_func,
        fn_kwargs={
            "system_prompt": system_prompt,
            "question_template": question_template,
            "answer_template": answer_template
        },
        num_proc=64
    ).cast_column("image", Image())
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
        script_args=script_args
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print('training_args:\n', training_args)
    print('script_args:\n', script_args)
    print('model_args:\n', model_args)
    main(script_args, training_args, model_args)