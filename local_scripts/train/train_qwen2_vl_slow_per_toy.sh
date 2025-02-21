#!/bin/bash

NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12345
cd /data/ICCV2025/PaR/R1-Multimodal-Journey
DATASET=/data/dataspace/slow_percept-sub1-rl

TRAINBS=$1
ROLLOUT_SIZE=$((TRAINBS * 7))
MAX_STEP=$((15000 / (TRAINBS * 7) + 1))
echo "MAX_STEP: $MAX_STEP"
OUTPUT_DIR="checkpoints/slowper/r1v_qwen2-slow_percept-sub1-2b_vllm_rollout$ROLLOUT_SIZE"
MODELNAME=checkpoints/Qwen2-VL-2B-JIHE/v0-20241214-092625/checkpoint-12374
# MODELNAME=/mnt/shared-storage/groups/hypertext/athenawei/checkpoints/Qwen2-VL-2B-Instruct

export LOG_PATH="${OUTPUT_DIR}/train.log"
python -m torch.distributed.run  \
    --nproc_per_node="7" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/open_r1/grpo_vllm.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path $MODELNAME \
    --dataset_name $DATASET \
    --max_prompt_length 2048 \
    --num_generations ${ROLLOUT_SIZE} \
    --max_completion_length 768 \
    --per_device_train_batch_size $TRAINBS \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 1000000 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-2B-JIHE-ROLLOUT$ROLLOUT_SIZE \
    --save_steps 100 \
    --max_steps $MAX_STEP \
    --reward_funcs "slowper_f1" \
    --save_only_model true \
    --system_prompt_template "qwen" \
    --question_template "slow_perception" \
    # >> train.log 2>&1