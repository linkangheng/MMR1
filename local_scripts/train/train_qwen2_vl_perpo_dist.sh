#!/bin/bash
export NCCL_ALGO=^Ring
export NCCL_NET_OVERHEAD=1000000
export TORCHRUN=/data/ICCV2025/PaR/MMR1/local_scripts/torchrun.sh

NNODES=2
ROLLOUT_SIZE=2
OUTPUT_DIR="/mnt/jfs-test/checkpoints/mmr1/debug/qwen2-vl-2b_vllm_perpo_roll${ROLLOUT_SIZE}"
RUN_NAME="qwen2-vl-2b_vllm_perpo_roll${ROLLOUT_SIZE}"
LOG_PATH="${OUTPUT_DIR}/train.log"
WANDB_PROJECT="MMR1"
WANDB_MODE="disabled"

rlaunch --gpu 8 --cpu 64 --memory=$((1024*800)) --charged-group pretrain2 --private-machine=yes --positive-tags feature/gpfs=yes \
    --set-env DISTRIBUTED_JOB=true \
    --set-env LOG_PATH="${LOG_PATH}" \
    --set-env WANDB_PROJECT="${WANDB_PROJECT}" \
    --set-env WANDB_MODE="${WANDB_MODE}" \
    --set-env HF_DATASETS_CACHE="/mnt/jfs-test/.cache/huggingface" \
    --mount=juicefs+s3://oss.i.shaipower.com/kanelin-jfs:/mnt/jfs-test -P $NNODES -- $TORCHRUN \
    src/open_r1/grpo_vllm.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path /mnt/jfs-test/models/Qwen2-VL-2B-Instruct \
    --dataset_name /mnt/jfs-test/data/perpo \
    --max_prompt_length 2048 \
    --max_completion_length 768 \
    --per_device_train_batch_size 4 \
    --num_generations ${ROLLOUT_SIZE} \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 1000000 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --report_to none \
    --reward_funcs "yjs" \
    --system_prompt_template "qwen" \
    --save_only_model true \
    --seed 42