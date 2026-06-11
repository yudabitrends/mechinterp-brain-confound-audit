#!/bin/bash
#SBATCH --job-name=mib_p3c
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/p3c_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/p3c_%j.err
set -e
source activate project
cd /home/users/ybi3/mechinterp_brain
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src
python scripts/phase3c_multilayer.py --auc-thresh 0.15 --device cuda
echo "=== DONE ==="
