model_path="/mnt/jfs-test/models/Qwen2-VL-2B-Instruct"
eval_code="/data/ICCV2025/PaR/MMR1/eval/evaluate_counting_bon.py"
world_size=8
temperature=1
top_p=0.5
top_k=10
batchsize=8

for pass_n in 256; do
    # for i in $(seq 0 $world_size); do
    #     export CUDA_VISIBLE_DEVICES=$i
    #     python $eval_code --model_path $model_path \
    #     --world_size $world_size --rank $i --pass_n $pass_n --stage "infer" \
    #     --temperature $temperature --top_p $top_p --baseline True --batchsize $batchsize &
    # done
    # wait
    python $eval_code --model_path $model_path --world_size $world_size --pass_n $pass_n --stage "eval" --baseline True
done