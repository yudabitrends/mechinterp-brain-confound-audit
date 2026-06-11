#!/bin/bash
#SBATCH --job-name=mib_p3
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/p3_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/p3_%j.err

# Phase 3: causal scanner-feature ablation at sMRI_encoder.norm (mechanistic harmonization).
set -e
source activate project
cd /home/users/ybi3/mechinterp_brain
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src
python scripts/phase3_ablate.py --target sMRI_encoder.norm --k 30 --device cuda
echo "=== DONE ==="
