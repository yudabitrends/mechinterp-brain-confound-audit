#!/bin/bash
#SBATCH --job-name=mib_sweep
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/sweep_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/sweep_%j.err
set -e
source activate project
cd /home/users/ybi3/mechinterp_brain
export PYTHONPATH=src
python scripts/phase3b_sweep.py
echo "=== DONE ==="
