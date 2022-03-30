#!/bin/bash

# Parameters
#SBATCH --array=0-5%1
#SBATCH --cpus-per-gpu=4
#SBATCH --error=/mnt/nfs/home/yunxingl/self-supervised-learning/multirun/sbatch/%A_%a_0_log.err
#SBATCH --gres=gpu:1080_ti:4
#SBATCH --job-name=pretrain
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/mnt/nfs/home/yunxingl/self-supervised-learning/multirun/sbatch/1_%A_%a_0_log.out
#SBATCH --time=2-00:00:00

declare -a train_dsets=(
    '"concat_datasets\(\[\"train_dataset\/imagenette.yaml\"\,\ subset_dataset\(\"train_dataset\/places.yaml\"\,\ 1000\)\]\)"'
    '"concat_datasets\(\[\"train_dataset\/imagenette.yaml\"\,\ subset_dataset\(\"train_dataset\/places.yaml\"\,\ 5000\)\]\)"'
    '"concat_datasets\(\[\"train_dataset\/imagenette.yaml\"\,\ subset_dataset\(\"train_dataset\/places.yaml\"\,\ 10000\)\]\)"'
    '"concat_datasets\(\[\"train_dataset\/imagewoof.yaml\"\,\ subset_dataset\(\"train_dataset\/places.yaml\"\,\ 1000\)\]\)"'
    '"concat_datasets\(\[\"train_dataset\/imagewoof.yaml\"\,\ subset_dataset\(\"train_dataset\/places.yaml\"\,\ 5000\)\]\)"'
    '"concat_datasets\(\[\"train_dataset\/imagewoof.yaml\"\,\ subset_dataset\(\"train_dataset\/places.yaml\"\,\ 10000\)\]\)"'
)

declare -a val_dsets=(
    "imagenette_val.yaml"
    "imagenette_val.yaml"
    "imagenette_val.yaml"
    "imagewoof_val.yaml"
    "imagewoof_val.yaml"
    "imagewoof_val.yaml"
)
#####


train_dset=${train_dsets[SLURM_ARRAY_TASK_ID]}
val_dset=${val_dsets[SLURM_ARRAY_TASK_ID]}
srun python3 main.py +experiment=pretrain_simclr train_dataset=function train_dataset.func_object="$train_dset" val_dataset=$val_dset
