#!/bin/bash

# Parameters
#SBATCH --array=0-271%4
#SBATCH --cpus-per-task=2
#SBATCH --error=/mnt/nfs/home/yunxingl/self-supervised-learning/multirun/sbatch/%A_%a_0_log.err
#SBATCH --gres=gpu:1080_ti:1
#SBATCH --job-name=byol
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/mnt/nfs/home/yunxingl/self-supervised-learning/multirun/sbatch/1_%A_%a_0_log.out
#SBATCH --time=60

# set stuff here 
# change the array! 
eval_config="save_embeddings.yaml"
ckpt_config=( 20 20 50 50 33 33 33 33 )

declare -a train_dirs=(
    "/mnt/nfs/home/yunxingl/self-supervised-learning/outputs/2022-01-11/12-56-49"
    "/mnt/nfs/home/yunxingl/self-supervised-learning/outputs/2022-01-14/08-05-55"
    "/mnt/nfs/home/yunxingl/self-supervised-learning/outputs/2021-11-15/10-24-32"
    "/mnt/nfs/home/yunxingl/self-supervised-learning/outputs/2021-11-15/10-20-24"
    "/mnt/nfs/home/yunxingl/self-supervised-learning/outputs/2022-02-01/14-08-35"
    "/mnt/nfs/home/yunxingl/self-supervised-learning/outputs/2022-02-01/14-44-38"
    "/mnt/nfs/home/yunxingl/self-supervised-learning/outputs/2022-02-08/14-23-58"
    "/mnt/nfs/home/yunxingl/self-supervised-learning/outputs/2022-02-11/00-00-07"
)
#####


get_directory_index () {
    run_sum=0
    idx=0
    local  __idx_dir=$1
    local  __idx_start=$2
    for i in "${ckpt_config[@]}"
    do
        next_val=$((run_sum + i))
        if [ $SLURM_ARRAY_TASK_ID -ge $next_val ]
        then 
            run_sum=$((next_val))
            idx=$((idx + 1))
        else 
            eval $__idx_dir=$idx
            eval $__idx_start=$run_sum
            break 
        fi 
    done
}

get_directory_index dir_idx idx_start 
train_dir=${train_dirs[dir_idx]}
ckpt_dir="$train_dir/checkpoints"
all_files=( $(ls $ckpt_dir | sort) )
ckpt_idx=$((SLURM_ARRAY_TASK_ID - idx_start))
FILE=${all_files[ckpt_idx]}
srun python3 eval_main.py +sweep_configs=$eval_config weight_path=$FILE train_dir=$train_dir
