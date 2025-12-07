#!/bin/bash
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=/dkucc/home/xz464/anaconda3/envs/Antispoofing/lib:$LD_LIBRARY_PATH
export NCCL_IB_HCA=mlx5_0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMBA_DISABLE_CACHE=1
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_TIMEOUT=18000



# DDP 进程信息
lrank=$OMPI_COMM_WORLD_LOCAL_RANK
comm_rank=$OMPI_COMM_WORLD_RANK
comm_size=$OMPI_COMM_WORLD_SIZE
node_rank=${SLURM_NODEID}

myhost=$(hostname)
echo lrank is $lrank at $myhost
echo comm_rank is $comm_rank at $myhost
echo comm_size is $comm_size at $myhost

echo disturl is tcp://${1}:33667



APP="python -u /dkucc/home/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/main_whisper_lora1.py --dist-url tcp://${1}:33667 \
    --dist-backend nccl --world-size=${comm_size} --rank=${comm_rank} --local-rank=${lrank}\
    --config /dkucc/home/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/job/whisper_aasist_with_new/eval_more/IW/eval_IW.conf \
    --outfile /dkucc/home/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/job/whisper_aasist_with_new/eval_more/IW/log/whisper_aasist_with_new.out\
    --eval"

case ${lrank} in
0)
  export CUDA_VISIBLE_DEVICES=0
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  numactl --physcpubind=0-7 --membind=0 ${APP}
  ;;
1)
  export CUDA_VISIBLE_DEVICES=1
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  numactl --physcpubind=8-15 --membind=0 ${APP}
  ;;
2)
  export CUDA_VISIBLE_DEVICES=2
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  numactl --physcpubind=16-23 --membind=0 ${APP}
  ;;
3)
  export CUDA_VISIBLE_DEVICES=3
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  numactl --physcpubind=24-31 --membind=0 ${APP}
  ;;
4)
  export CUDA_VISIBLE_DEVICES=4
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  numactl --physcpubind=32-39 --membind=1 ${APP}
  ;;
5)
  export CUDA_VISIBLE_DEVICES=5
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  numactl --physcpubind=40-47 --membind=1 ${APP}
  ;;
6)
  export CUDA_VISIBLE_DEVICES=6
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  numactl --physcpubind=48-55 --membind=1 ${APP}
  ;;
7)
  export CUDA_VISIBLE_DEVICES=7
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  numactl --physcpubind=56-63 --membind=1 ${APP}
  ;;
esac
