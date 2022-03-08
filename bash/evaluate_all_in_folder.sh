#!/bin/bash

BASEPATH="/mnt/nfs/home/yunxingl/self-supervised-learning/outputs/2022-02-08/14-23-58"
f="$BASEPATH/checkpoints/epoch_449.ckpt" #459/449 and 999 for byol models 
REPRESENTATION_SIZE=2048
fnew=$(echo $f | sed -e s/=/'\\='/g)
fname=$(basename -s .ckpt $fnew)
weight_file=$(basename $fnew)
srun -n 1 -c 2 --time=2-00:00:00 --pty python3 main.py -m +experiment=linear_fit sweep_configs=sweep_linear_fit model.kwargs.weight_file=\"$weight_file\" model.kwargs.train_dir=$BASEPATH model.kwargs.representation_size=$REPRESENTATION_SIZE callbacks.model_checkpoint.dirpath="$BASEPATH/linear_checkpoint" callbacks.model_checkpoint.filename="$fname-\{epoch\}-\{val_loss\}"
