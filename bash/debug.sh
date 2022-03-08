#!/bin/bash

BASEPATH="/mnt/nfs/home/yunxingl/self-supervised-learning/outputs/2022-02-01/14-08-35"
FILES="$BASEPATH/checkpoints/*.ckpt"
REPRESENTATION_SIZE=2048
for f in $FILES; 
do 
    fnew=$(echo $f | sed -e s/=/'\\='/g)
    fname=$(basename -s .ckpt $fnew)
    filename="$fname\_\{epoch\}"
    echo $fname
    echo $filename 
done