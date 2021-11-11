alias debugshell='srun -n 1 -c 2 --pty zsh'
alias debugeval='python3 -m pdb main.py experiment=linear_fit trainer=debug_cpu'
alias debugtrain='python3 -m pdb main.py experiment=pretrain trainer=debug_cpu'
alias pretrain='srun --time=2-00:00:00 --gres=gpu:1080_ti:4 --cpus-per-gpu 4 -n 1 --pty python3 main.py experiment=pretrain'