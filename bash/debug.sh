#!/bin/bash

# BASEPATH="/mnt/nfs/home/yunxingl/self-supervised-learning/outputs/2022-02-01/14-08-35"
# FILES="$BASEPATH/checkpoints/*.ckpt"
# REPRESENTATION_SIZE=2048
# for f in $FILES; 
# do 
#     fnew=$(echo $f | sed -e s/=/'\\='/g)
#     fname=$(basename -s .ckpt $fnew)
#     filename="$fname\_\{epoch\}"
#     echo $fname
#     echo $filename 
# done

declare -a train_dsets=(
    "imagenette.yaml"
    "imagewoof.yaml"
    "concat_datasets(['train_dataset/imagenette.yaml', subset_dataset('train_dataset/places.yaml', 1000)])"
    "concat_datasets(['train_dataset/imagenette.yaml', subset_dataset('train_dataset/places.yaml', 5000)])"
    "concat_datasets(['train_dataset/imagenette.yaml', subset_dataset('train_dataset/places.yaml', 10000)])"
    "concat_datasets(['train_dataset/imagewoof.yaml', subset_dataset('train_dataset/places.yaml', 1000)])"
    "concat_datasets(['train_dataset/imagewoof.yaml', subset_dataset('train_dataset/places.yaml', 5000)])"
    "concat_datasets(['train_dataset/imagewoof.yaml', subset_dataset('train_dataset/places.yaml', 10000)])"
)

declare -a val_dsets=(
    "imagenette_val.yaml"
    "imagewoof_val.yaml"
    "imagenette_val.yaml"
    "imagenette_val.yaml"
    "imagenette_val.yaml"
    "imagewoof_val.yaml"
    "imagewoof_val.yaml"
    "imagewoof_val.yaml"
)
#####

SLURM_ARRAY_TASK_ID=3
train_dset=${train_dsets[SLURM_ARRAY_TASK_ID]}
val_dset=${val_dsets[SLURM_ARRAY_TASK_ID]}
echo $train_dset 
echo $val_dset


["70f.tar",
"7b5.tar",
"d2b.tar",
"f6c.tar",
"9a8.tar",
"669.tar",
"208.tar",
"3d4.tar",
"eab.tar",
"86c.tar",
"c2f.tar",
"31e.tar",
"ddd.tar",
"41a.tar",
"c63.tar",
"826.tar",
"d8a.tar",
"77d.tar",
"454.tar",
"d15.tar",
"4c2.tar",
"ee9.tar",
"792.tar",
"7ab.tar",
"46c.tar",
"5a8.tar",
"485.tar",
"743.tar",
"c5d.tar",
"c8e.tar",
"b49.tar",
"dcf.tar",
"0af.tar",
"426.tar",
"ccb.tar",
"3c6.tar",
"d67.tar",
"854.tar"]