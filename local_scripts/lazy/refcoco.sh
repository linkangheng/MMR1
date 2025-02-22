# Baseline: /mnt/jfs-test/models/Qwen2-VL-2B-Instruct
MODEL_DIR="/mnt/jfs-test/checkpoints/mmr1/debug"
MODELS="$MODEL_DIR/qwen2-vl-2b_vllm_refcoco_5k_qwenvl_format_roll*"

cd /data/ICCV2025/PaR/MMR1/eval # switch to the correct directory
for model in $MODELS
do
    echo "Evaluating $model"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    for task_id in {0..7}
    do
        export CUDA_VISIBLE_DEVICES=$task_id
        python evaluate_refcoco.py \
            --model_path $model \
            --task_id $task_id \
            --timestamp $TIMESTAMP &
    done
    wait
    python evaluate_refcoco.py \
        --model_path $model \
        --timestamp $TIMESTAMP \
        --mode orgnize
done
