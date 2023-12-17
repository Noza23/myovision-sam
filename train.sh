#!/bin/bash
#
#SBATCH --partition=jobs-gpu
#SBATCH -w srvdrai1
#SBATCH --gpus=6
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=192GB
#SBATCH -J multi-gpu-training
#SBATCH -o logs/multi-gpu-training_%A_%a.out
#SBATCH -e logs/multi-gpu-training_%A_%a.err
#SBATCH -A core-mum
#SBATCH --array=0-15%1

TORCH_DISTRIBUTED_DEBUG=INFO

singularity exec --pwd $(pwd) --nv \
  -B /data/core-mum/myovision:/mnt \
  /data/common/radler/radler_pytorch_v3.7.0 \
  bash -c "cd /mnt/myovision-sam && torchrun --standalone --nproc_per_node=gpu main.py"