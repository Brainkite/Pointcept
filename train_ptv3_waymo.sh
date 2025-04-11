#!/bin/bash
GPUS=$(( $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) ))

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointcept
wandb login
sh scripts/train.sh -g $GPUS -d waymo -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base