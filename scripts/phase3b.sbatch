#!/bin/bash
#SBATCH --job-name=mib_p3b
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/p3b_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/p3b_%j.err

# Phase 3b: capture fused decision rep (GPU) -> scanner-direction erasure harmonization (CPU).
set -e
source activate project
cd /home/users/ybi3/mechinterp_brain
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src
python scripts/extract_fused.py
python scripts/phase3b_erase.py --rounds 15
echo "=== DONE ==="
