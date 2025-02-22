scales=(1 5 10 20)
base_size=3000

for scale in ${scales[@]}; do
    train_sample_size=$((${base_size} * ${scale}))
    bash /data/ICCV2025/PaR/MMR1/local_scripts/train/grounding/train_qwen2_vl_perpo_scaling.sh ${train_sample_size}
done