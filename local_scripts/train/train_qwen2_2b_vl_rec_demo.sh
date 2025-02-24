#!/bin/bash

NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12345

# MODIFY HERE: please prepare the env related variables
MMR1_PATH="<path to mmr1>" # path to mmr1
CHECKPOINT_PATH="<path to checkpoint>" # directory to save the checkpoint
RUN_NAME="<run_name>" # describe what your experiment is about

# Default Setting
OUTPUT_DIR="${CHECKPOINT_PATH}/${RUN_NAME}" # path to save the output
SRC_PATH="${OUTPUT_DIR}/src" # path to backup the source code
export LOG_DIR="${OUTPUT_DIR}" # path to save the log
export WANDB_PROJECT="MMR1" # project name in wandb
export WANDB_TAGS="qwen2-vl-perpo-demo" # tags for the experiment in wandb

if [ ! -d "${OUTPUT_DIR}"/src ]; then
    mkdir -p ${OUTPUT_DIR}/src
fi

# backup the source code
cp -r ${MMR1_PATH}/src ${SRC_PATH}

# run the training
torchrun \
    --nproc_per_node="7" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    ${MMR1_PATH}/src/open_r1/grpo_vllm.py \
    --deepspeed ${MMR1_PATH}/local_scripts/zero3.json \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path /mnt/jfs-test/models/Qwen2-VL-2B-Instruct \
    --dataset_name /mnt/jfs-test/data/perpo \
    --max_prompt_length 2048 \
    --max_completion_length 768 \
    --per_device_train_batch_size 7 \
    --gradient_accumulation_steps 2 \
    --num_generations 7 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --report_to wandb \
    --max_pixels 1000000 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 200 \
    --reward_funcs "perpo_reward" \
    --save_only_model true \
    --system_prompt_template "default" \
    --question_template "default" \
    --answer_template "default"