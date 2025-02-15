session_name="gpu"
machine_num=1
for i in $(seq 0 $((machine_num - 1))); do
    tmux new-session -d -s "${session_name}-${i}"
    tmux send-keys -t "${session_name}-${i}" "rlaunch --gpu 8 --cpu 64 --memory=$((1024*400)) --charged-group pretrain2  -e MACHINE_ID=$i --private-machine=yes --positive-tags feature/gpfs=yes --mount=gpfs://gpfs1/basemind-hypertext/:/mnt/shared-storage/groups/hypertext/ --i-know-i-am-wasting-resource  -- bash " C-m
    tmux send-keys -t "${session_name}-${i}" "cd /data/LLaVA/" C-m
    tmux send-keys -t "${session_name}-${i}" "conda activate spavl" C-m
    tmux send-keys -t "${session_name}-${i}" "bash test.sh" C-m
    tmux send-keys -t "${session_name}-${i}" "exit" C-m
done