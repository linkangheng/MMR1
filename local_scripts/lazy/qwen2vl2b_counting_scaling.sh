settings=(
    "2 2 2"
    "4 4 2"
    "8 8 2"
    "16 16 2"
    "32 32 2"
    "63 63 2"
    "126 18 14"
    "252 36 14"
    "378 54 14"
    "441 63 14"
)
for ((i=0; i<${#settings[@]}; i++)); do
    # 将字符串分割为数组
    params=(${settings[$i]})
    roll_num=${params[0]}
    batch_size=${params[1]}
    accumulation_steps=${params[2]}
    bash /data/ICCV2025/PaR/MMR1/local_scripts/train/lucas_counting/train_qwen2_vl_2b_lucas_couting_min20box_1k+scaling.sh $roll_num $batch_size $accumulation_steps
done