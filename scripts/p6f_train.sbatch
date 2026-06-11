#!/bin/bash
#SBATCH --job-name=mib_p6f
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/p6f_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/p6f_%j.err
set -e
source activate project
cd /home/users/ybi3/MultiViT2/geometric_multivit
python scripts/train_custom_split.py --config configs/multivit_default.yaml \
    --out-dir /home/users/ybi3/MultiViT2/outputs/p6f_custom_split \
    --test-size 0.3 --seed 42 --device cuda
echo "=== DONE ==="
