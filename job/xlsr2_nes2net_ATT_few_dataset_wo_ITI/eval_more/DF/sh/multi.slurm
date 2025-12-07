#!/bin/bash
#SBATCH -e /dkucc/home/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/job/whisper_aasist_with_new/eval_more/DF/log/whisper_aasist_with_new.err
#SBATCH -o /dkucc/home/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/job/whisper_aasist_with_new/eval_more/DF/log/whisper_aasist_with_new.out
#SBATCH -J eval_DF_whisper_aasist_with_new
#SBATCH -p l20-gpu
#SBATCH -N 3                    # 使用两个节点
#SBATCH --ntasks-per-node=8      # 每节点 8 个任务
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --export=ALL

module load OpenMPI/4.1.8-gpu
module load cuda/12.6
module load gcc/12.4.0
module use $HOME/my_modules
module load numactl/1.0
source /dkucc/home/xz464/anaconda3/bin/activate
conda activate Antispoofing




hostfile=./$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
num_node=$(cat $hostfile|sort|uniq | wc -l)

num_GPU=$(($num_node*8))

nodename=$(cat $hostfile | sed -n "1p")	# one p
dist_url=`echo $nodename | awk '{print $1}'`


rm `pwd`/hostfile-dl -f

cat $hostfile | sort | uniq > `pwd`/tmp

for i in `cat ./tmp`
do
  echo ${i} slots=8 >> `pwd`/hostfile-dl
done


mpirun -np $num_GPU --allow-run-as-root --bind-to none -hostfile `pwd`/hostfile-dl \
  sh /dkucc/home/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/job/whisper_aasist_with_new/eval_more/DF/sh/single.sh $dist_url


