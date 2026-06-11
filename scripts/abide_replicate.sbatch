#!/bin/bash
#SBATCH --job-name=mib_abrep
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/abrep_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/abrep_%j.err
set -e
source activate project
cd /home/users/ybi3/mechinterp_brain
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src
python scripts/abide_replicate.py --min-site-n 10
echo "=== DONE ==="
