#!/bin/bash

#
#SBATCH --partition=jobs-gpu
#SBATCH -w srvdrai1
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=8G
#SBATCH -J multi-gpu-training
#SBATCH -o logs/multi-gpu-training.out
#SBATCH -e logs/multi-gpu-training.err
#SBATCH -A core-mum

singularity exec --pwd $(pwd) --nv \
  -B /data/core-mum/myovision:/mnt \
  /data/common/radler/radler_pytorch_v3.7.0 \
  bash -c "cd /mnt && torchrun --standalone --nproc_per_node=gpu test.py"
