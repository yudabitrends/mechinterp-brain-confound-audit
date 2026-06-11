#!/bin/bash
#SBATCH --job-name=mib_structmap
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/structmap_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/structmap_%j.err
set -e
source /home/users/ybi3/anaconda3/etc/profile.d/conda.sh
conda activate project
cd /home/users/ybi3/mechinterp_brain
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src
export OMP_NUM_THREADS=6
python scripts/compute_struct_maps.py
echo "=== DONE ==="
