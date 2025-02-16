#!/bin/bash

NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12345
cd /data/ICCV2025/PaR/MMR1
OUTPUT_DIR="/mnt/jfs-test/checkpoints/mmr1/debug/qwen2-vl-2b_vllm_lucas_counting"
RUN_NAME="qwen2-vl-2b_vllm_lucas_counting_defualt_settings"
export LOG_PATH="${OUTPUT_DIR}/train.log"
export WANDB_PROJECT="MMR1"

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
    --dataset_name /mnt/jfs-test/data/lucas_counting/molmo/jsons/lucas_counting_molmo_1892054.json \
    --max_prompt_length 2048 \
    --num_generations 7 \
    --max_completion_length 768 \
    --per_device_train_batch_size 7 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 1000000 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --report_to none \
    --reward_funcs "count_acc" "count_format" \
    --save_only_model true \
    --train_sample_size 1000 \
    --question_template "counting_reasoning" \
    --answer_template "counting_reasoning" \
    --system_prompt "counting_reasoning"
    # >> train.log 2>&1