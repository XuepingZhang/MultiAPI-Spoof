#!/bin/bash
# single.sh
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0


export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=/dkucc/home/xz464/anaconda3/envs/Antispoofing/lib:$LD_LIBRARY_PATH
rank=$OMPI_COMM_WORLD_RANK
local_rank=$OMPI_COMM_WORLD_LOCAL_RANK
world_size=$OMPI_COMM_WORLD_SIZE
node_rank=$OMPI_COMM_WORLD_NODE_RANK
export CUDA_VISIBLE_DEVICES=$local_rank

# 随机端口，避免冲突
port=33661
(
while true; do
    nvidia-smi >> /work/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/job/xlsr2_aasisit_few_dataset/log/nvidia-smi.out
    sleep 60
done
) &
python /work/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/main1.py \
  --dist-url tcp://$1:${port} \
  --dist-backend nccl \
  --world-size $world_size \
  --rank $rank \
  --local-rank $local_rank \
  --node-rank $node_rank \
  --config /work/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/job/xlsr2_aasisit_few_dataset/few_dkucc.conf \
  --outfile /work/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/job/xlsr2_aasisit_few_dataset/log/xlsr2_aasisit_few_dataset.out \
  --eval