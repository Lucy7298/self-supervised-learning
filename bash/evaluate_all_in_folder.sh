#!/bin/bash

FILES="/mnt/nfs/home/yunxingl/self-supervised-learning/outputs/2021-11-11/21-47-12/checkpoints/*.ckpt"
for f in $FILES; 
do 
    fnew=$(echo $f | sed -e s/=/'\\='/g)
    srun -n 1 -c 2 --time=2-00:00:00 --pty python3 main.py -m experiment=linear_fit sweep_configs=sweep_linear_fit model.kwargs.weight_path=\"$fnew\" 
done