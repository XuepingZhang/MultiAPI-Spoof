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


port=33661
(
while true; do
    nvidia-smi >> log/nvidia-smi.out
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
  --config few_dkucc.conf \
  --outfile log/xlsr2_aasisit_few_dataset.out