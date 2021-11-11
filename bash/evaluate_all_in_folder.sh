#!/bin/bash

FILES="/mnt/nfs/home/yunxingl/byol-pytorch/examples/lightning/byol-lucidrains/3dxbi8tn/checkpoints/*.ckpt"
for f in $FILES; 
do 
    fnew=$(echo $f | sed -e s/=/'\\='/g)
    srun -n 1 -c 2 --time=2-00:00:00 --pty python3 main.py -m experiment=linear_fit sweep_configs=sweep_linear_fit model.kwargs.weight_path=\"$fnew\"
done