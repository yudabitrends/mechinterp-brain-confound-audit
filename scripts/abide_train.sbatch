#!/bin/bash
#SBATCH --job-name=mib_abtr
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/abtr_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/abtr_%j.err
set -e
source activate project
cd /home/users/ybi3/mechinterp_brain
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src
python scripts/train_abide.py --device cuda
echo "=== DONE ==="
