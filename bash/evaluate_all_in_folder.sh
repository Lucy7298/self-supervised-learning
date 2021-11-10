#!/bin/bash

FILES="/mnt/nfs/home/yunxingl/self-supervised-learning/outputs/2021-11-04/22-23-23/BYOL/znaicf73/checkpoints/*.ckpt"
for f in $FILES; 
do 
    fnew=$(echo $f | sed -e s/=/'\\='/g)
    srun -n 1 -c 2 --time=2-00:00:00 --pty python3 main.py -m experiment=linear_fit sweep_configs=sweep_linear_fit model.kwargs.weight_path=\"$fnew\"
done