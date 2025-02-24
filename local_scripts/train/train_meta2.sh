




TRAINBS=1
TAG=sft_5k_data
MODELNAME=/data/workspace/ms-swift/checkpoints/Qwen2-VL-2B-SLOWPER-5k/v0-20250223-065648/checkpoint-154

bash local_scripts/train/train_qwen2_vl_slow_per_toy.sh $TRAINBS $TAG $MODELNAME


TRAINBS=1
TAG=sft_2k_data
MODELNAME=/data/workspace/ms-swift/checkpoints/Qwen2-VL-2B-SLOWPER-2k/v0-20250223-064355/checkpoint-62

bash local_scripts/train/train_qwen2_vl_slow_per_toy.sh $TRAINBS $TAG $MODELNAME


TRAINBS=2
TAG=sft_5k_data
MODELNAME=/data/workspace/ms-swift/checkpoints/Qwen2-VL-2B-SLOWPER-5k/v0-20250223-065648/checkpoint-154

bash local_scripts/train/train_qwen2_vl_slow_per_toy.sh $TRAINBS $TAG $MODELNAME