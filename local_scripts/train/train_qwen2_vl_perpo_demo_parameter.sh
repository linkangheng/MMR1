#!/bin/bash

NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12345
cd /data/MMR1
use_kl=(True)
kl_approximator=(k3)
entropy_reg=(True)
entropy_weight=(0.01)
temperature_func=("linear")
sync_ref_model=(False)
ref_model_mixup_alpha=(0.9)
ref_model_sync_steps=(64)
order_dataset=("llava1.5-7b_easy2diff")
reward_rule=(1)
reward_scale=(1)
reward_baseline=(0)
num_generations=(8)
learning_rate=(1e-6)

machine_id=${MACHINE_ID}

OUTPUT_DIR="/mnt/jfs-test/yjs/results/perpo/Qwen2-VL-2B-Instruct/grpo_rule-${reward_rule[$machine_id]}-${reward_scale[$machine_id]}scale-baseline${reward_baseline[$machine_id]}_${num_generations[$machine_id]}roll_${learning_rate[$machine_id]}lr_${kl_approximator[$machine_id]}kl_${order_dataset[$machine_id]}order_${entropy_reg[$machine_id]}enpy_${entropy_weight[$machine_id]}w_${temperature_func[$machine_id]}t_${sync_ref_model[$machine_id]}sync_${ref_model_mixup_alpha[$machine_id]}alpha_${ref_model_sync_steps[$machine_id]}steps_groundingprompt"


if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p "${OUTPUT_DIR}"
fi
cp /data/MMR1/src/open_r1/arguments.py ${OUTPUT_DIR}/arguments.py
cp /data/MMR1/src/open_r1/grpo_vllm.py ${OUTPUT_DIR}/grpo_vllm.py
cp /data/MMR1/src/open_r1/trainer/grpo_trainer_vllm.py ${OUTPUT_DIR}/grpo_trainer_vllm.py
cp /data/MMR1/src/open_r1/rewards.py ${OUTPUT_DIR}/rewards.py
cp /data/MMR1/src/open_r1/constants.py ${OUTPUT_DIR}/constants.py
cp /data/MMR1/src/open_r1/temperature.py ${OUTPUT_DIR}/temperature.py



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
    --dataset_name /mnt/jfs-test/data/perpo \
    --max_prompt_length 2048 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --report_to wandb \
    --max_pixels 1000000 \
    --num_train_epochs 1 \
    --run_name Qwen2-2B_${OUTPUT_DIR} \
    --save_steps 200 \
    --reward_funcs "yjs_grounding" \
    --save_only_model true \
    --order_dataset ${order_dataset[$machine_id]} \
    --use_kl ${use_kl[$machine_id]} \
    --kl_approximator ${kl_approximator[$machine_id]} \
    --entropy_reg ${entropy_reg[$machine_id]} \
    --entropy_weight ${entropy_weight[$machine_id]} \
    --temperature_func ${temperature_func[$machine_id]} \
    --sync_ref_model ${sync_ref_model[$machine_id]} \
    --ref_model_mixup_alpha ${ref_model_mixup_alpha[$machine_id]} \
    --ref_model_sync_steps ${ref_model_sync_steps[$machine_id]} \
    --learning_rate ${learning_rate[$machine_id]} \
    --num_generations ${num_generations[$machine_id]} \
    --reward_rule ${reward_rule[$machine_id]} \
    --reward_scale ${reward_scale[$machine_id]} \
    --reward_baseline ${reward_baseline[$machine_id]} \
    --prompt_template "grounding"
    
    # >> train.log 2>&1