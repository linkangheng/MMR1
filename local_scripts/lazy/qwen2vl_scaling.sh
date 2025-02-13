for i in 2 4 8 16 32 64 128; do
    bash local_scripts/train/train_qwen2_vl_perpo.sh $i
done