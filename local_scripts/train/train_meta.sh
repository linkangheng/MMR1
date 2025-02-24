# /data/workspace/ms-swift/checkpoints/Qwen2-VL-2B-SLOWPER/v45-20250223-000640/checkpoint-1500
# /data/workspace/ms-swift/checkpoints/Qwen2-VL-2B-SLOWPER/v47-20250223-061805/checkpoint-31
# /data/workspace/ms-swift/checkpoints/Qwen2-VL-2B-SLOWPER-2k/v0-20250223-064355/checkpoint-62
# /data/workspace/ms-swift/checkpoints/Qwen2-VL-2B-SLOWPER-5k/v0-20250223-065648/checkpoint-154
# /data/workspace/MMR1/checkpoints/Qwen2-VL-2B-JIHE-SP/v0-20241214-093034/checkpoint-12374/global_step12374

# N_GPUS=7
# TRAINBS=1
# TAG=sft_1500steps
# MODELNAME=/data/workspace/ms-swift/checkpoints/Qwen2-VL-2B-SLOWPER/v45-20250223-000640/checkpoint-1500
# REWARD_FUNC=slowper_ed
# bash local_scripts/train/train_qwen2_vl_slow_per_toy.sh $N_GPUS $TRAINBS $TAG $MODELNAME $REWARD_FUNC


N_GPUS=8
TRAINBS=1
TAG=sft-all-qwen_prompt
MODELNAME=/data/workspace/MMR1/checkpoints/Qwen2-VL-2B-JIHE/v0-20241214-092625/checkpoint-12374
REWARD_FUNC=slowper_f1
bash local_scripts/train/train_qwen2_vl_slow_per_toy.sh $N_GPUS $TRAINBS $TAG $MODELNAME $REWARD_FUNC


# N_GPUS=7
# TRAINBS=1
# TAG=sft_all_sp
# MODELNAME=/data/workspace/MMR1/checkpoints/Qwen2-VL-2B-JIHE-SP/v0-20241214-093034/checkpoint-12374/global_step12374

# bash local_scripts/train/train_qwen2_vl_slow_per_toy.sh $N_GPUS $TRAINBS $TAG $MODELNAME


