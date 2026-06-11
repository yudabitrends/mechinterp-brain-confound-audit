#!/bin/bash
#SBATCH --job-name=mib_p5b
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/p5b_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/p5b_%j.err
set -e
source /home/users/ybi3/anaconda3/etc/profile.d/conda.sh
conda activate project
cd /home/users/ybi3/mechinterp_brain
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src
python scripts/phase5b_das_earlier.py
echo "=== DONE ==="
