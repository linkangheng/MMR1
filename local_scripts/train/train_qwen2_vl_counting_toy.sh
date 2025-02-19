#!/bin/bash

NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12345
OUTPUT_DIR="<checkpoint path>"
export LOG_PATH="${OUTPUT_DIR}/train.log"
torchrun \
    --nproc_per_node="7" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/open_r1/grpo_vllm.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path /mnt/jfs-test/models/Qwen2-VL-2B-Instruct \
    --dataset_name /mnt/jfs-test/data/clevr_cogen_a_train \
    --max_prompt_length 2048 \
    --num_generations 8 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 1000000 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-2B-GRPO-8k \
    --save_steps 100 \
    --max_steps 100 \
    --save_only_model true
    # >> train.log 2>&1
