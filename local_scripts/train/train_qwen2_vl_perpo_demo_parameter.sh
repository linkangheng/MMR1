#!/bin/bash

NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12345
cd /data/MMR1
use_kl=(True False True True True True True True True False True)
kl_approximator=(k3 k3 kimikl fullkimi k3 k3 k3 k3 k3 k3 k3)
entropy_reg=(False False False False True True False True True True)
entropy_weight=(-0.01 -0.01 -0.01 -0.01 -0.01 -0.01 -0.01 -0.01 -0.01 -0.01 -0.01)
temperature_func=("linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear" "linear")
sync_ref_model=(False False False False False False False True False False False)
ref_model_mixup_alpha=(0.9 0.9 0.9 0.9 0.9 0.9 0.9 1.0 0.9 0.9 0.9)
ref_model_sync_steps=(64 64 64 64 64 64 64 16 64 64 64)
reward_scale=(1 1 1 1 1 1 1 1 1 1 1)
num_generations=(7 7 7 7 7 7 7 7 7 7 7)
learning_rate=(5e-6 5e-6 5e-6 5e-6 5e-6 5e-6 5e-6 1e-6 1e-5 1.5e-5 3e-5)
temperature_begin=(0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8)
temperature_end=(1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5)

machine_id=${MACHINE_ID}
MMR1_path=/data/MMR1

OUTPUT_DIR=/mnt/jfs-test/yjs/results/perpo/Qwen2-VL-2B-Instruct/grpo_${use_kl[$machine_id]}kl_${reward_scale[$machine_id]}scale_${num_generations[$machine_id]}roll_${learning_rate[$machine_id]}lr_${kl_approximator[$machine_id]}kl_${entropy_reg[$machine_id]}enpy_${entropy_weight[$machine_id]}w_${temperature_func[$machine_id]}t_${temperature_begin[$machine_id]}begin_${temperature_end[$machine_id]}end_${sync_ref_model[$machine_id]}sync_${ref_model_mixup_alpha[$machine_id]}alpha_${ref_model_sync_steps[$machine_id]}steps_groundingprompt


if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p ${OUTPUT_DIR}
fi
cp ${MMR1_path}/src/open_r1/arguments.py ${OUTPUT_DIR}/arguments.py
cp ${MMR1_path}/src/open_r1/grpo_vllm.py ${OUTPUT_DIR}/grpo_vllm.py
cp ${MMR1_path}/src/open_r1/trainer/grpo_trainer_vllm.py ${OUTPUT_DIR}/grpo_trainer_vllm.py
cp ${MMR1_path}/src/open_r1/rewards.py ${OUTPUT_DIR}/rewards.py
cp ${MMR1_path}/src/open_r1/constants.py ${OUTPUT_DIR}/constants.py
cp ${MMR1_path}/src/open_r1/temperature.py ${OUTPUT_DIR}/temperature.py



export LOG_PATH="${OUTPUT_DIR}/train.log"
export WANDB_PROJECT="MMR1"
torchrun \
    --nproc_per_node="7" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    ${MMR1_path}/src/open_r1/grpo_vllm.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path /mnt/jfs-test/models/Qwen2-VL-2B-Instruct \
    --dataset_name /mnt/jfs-test/data/perpo_easy2diff_llava1.57b \
    --max_prompt_length 2048 \
    --max_completion_length 768 \
    --per_device_train_batch_size 7 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --report_to wandb \
    --max_pixels 1000000 \
    --num_train_epochs 1 \
    --run_name debug \
    --save_steps 200 \
    --reward_funcs "yjs_grounding" \
    --save_only_model true \
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
    --reward_scale ${reward_scale[$machine_id]} \
    --temperature_begin ${temperature_begin[$machine_id]} \
    --temperature_end ${temperature_end[$machine_id]} \
    --origin_pg False \
    --no_mean_for_same_reward False \
    --system_prompt_template "grounding"
    
    # >> train.log 2>&1