#!/bin/bash
# single.sh
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0


export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=source /public/anaconda3/bin/activate/lib:$LD_LIBRARY_PATH # Change it to your own path
rank=$OMPI_COMM_WORLD_RANK
local_rank=$OMPI_COMM_WORLD_LOCAL_RANK
world_size=$OMPI_COMM_WORLD_SIZE
node_rank=$OMPI_COMM_WORLD_NODE_RANK
export CUDA_VISIBLE_DEVICES=$local_rank

# 随机端口，避免冲突
port=33666
(
while true; do
    nvidia-smi >> log/nvidia-smi.out
    sleep 60
done
) &

python main_api1.py --dist-url tcp://${1}:33666 \
    --dist-backend nccl --world-size=${comm_size} --rank=${comm_rank} --local-rank=${lrank}\
    --config api_tracing.conf \
    --outfile log/api_tracing.out \
    --eval
