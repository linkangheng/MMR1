#!/usr/bin/bash
#set -x
SCRIPT_PATH=$(dirname $(realpath "${BASH_SOURCE[0]}"))
BASE_DIR=$(realpath -s --relative-base=. $(dirname "${SCRIPT_PATH}"))

source $SCRIPT_PATH/worker_init.sh

set -x
# TODO: prefix $BASE_DIR/python/rank_run.py before user command to hook per rank initialization logic
torchrun --nproc_per_node 7 --master_addr $MASTER_ADDR --master_port ${MASTER_PORT:-5678} --nnodes $NODE_COUNT --node_rank $NODE_RANK -- ${@:1}
LAST_EXIT_CODE=$?
{ set +x; } 2>/dev/null
exit ${LAST_EXIT_CODE}
