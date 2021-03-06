#!/bin/bash

alias debuggpu='srun -n 1 -c 2 --gres=gpu:1080_ti:2 --pty zsh'
alias debugcpu='srun -n 1 -c 2 --pty zsh'
alias debugeval='python3 -m pdb main.py experiment=linear_fit trainer=debug_cpu'
alias debugpretrain='python3 -m pdb main.py experiment=pretrain trainer=debug_cpu'
alias debugpretrain_gpu='python3 main.py experiment=pretrain'
alias pretrain='srun --time=2-00:00:00 --gres=gpu:1080_ti:4 --cpus-per-gpu 4 -n 1 --pty python3 main.py +experiment=pretrain'
alias evalone='srun --time=2-00:00:00 --gres=gpu:1080_ti:2 --cpus-per-gpu 4 -n 1 --pty python3 main.py +experiment=linear_fit'
alias getnotebook='
cd /mnt/nfs/home/yunxingl/self-supervised-learning/notebook
nohup srun -w deep-gpu-4 -n 1 -c 2 --time=2-00:00:00 --resv-ports=1 bash -c "jupyter notebook --no-browser --port=\$SLURM_STEP_RESV_PORTS --ip 0.0.0.0" > test_nb.log &!
cat test_nb.log'

pretrain_file () {
  echo "running experiment on $1"
  srun --time=2-00:00:00 --gres=gpu:1080_ti:4 --cpus-per-gpu 4 -n 1 --pty python3 main.py +experiment=$1
}

debug_file () {
  echo "debugging config file $1"
  srun -n 1 -c 2 --pty python3 main.py +experiment=$1 trainer=debug_cpu
}

debug_file_gpu () {
  echo "debugging config file $1"
  srun --gres=gpu:1080_ti:4 --cpus-per-gpu 4 -n 1 --pty python3 main.py +experiment=$1 trainer=debug_gpu
}

alias debug_eval='srun -n 1 -c 2 --pty python3 eval_main.py trainer=debug_cpu'
alias evaluate_main='srun -n 1 -c 2  --gres=gpu:1080_ti:1 --pty python3 eval_main.py +sweep_configs=eval_invariance'

debug_eval_cpu () {
  echo "debugging evaluation file $1"
  srun -n 1 -c 2  --pty python3 eval_main.py trainer=cpu +sweep_configs=$1 output_dir=debug
}

debug_eval_gpu () {
  echo "debugging evaluation file $1"
  srun -n 1 -c 2 --gres=gpu:1080_ti:1 --pty python3 eval_main.py +sweep_configs=$1 output_dir=debug
}

alias go_to_batch_err='cd /mnt/nfs/home/yunxingl/self-supervised-learning/multirun/sbatch'
alias refresh_shortcut='source ~/self-supervised-learning/bash/useful_command.bashrc'
alias myqueue='squeue -u yunxingl'
