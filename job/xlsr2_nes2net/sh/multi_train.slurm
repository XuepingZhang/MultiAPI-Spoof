#!/bin/bash
#SBATCH -e log/xlsr2_nes2net_few_dataset.err
#SBATCH -o log/xlsr2_nes2net_few_dataset.out
#SBATCH -J xlsr2_nes2net_few_dataset
#SBATCH -p l20-gpu
#SBATCH -N 1
#SBATCH --mem=256g
#SBATCH --ntasks-per-node=8      # 每节点 8 个任务
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --export=ALL

module load OpenMPI/4.1.8-gpu
module load cuda/12.6
module load gcc/12.4.0
module use $HOME/my_modules
module load numactl/1.0

source /public/anaconda3/bin/activate # Change it to your own path
conda activate multiapi-spoof




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
  sh sh/single.sh $dist_url


