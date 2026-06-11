#!/bin/bash
#SBATCH --job-name=mib_p5c_depth
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/p5c_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/p5c_%j.err
set -e
source /home/users/ybi3/anaconda3/etc/profile.d/conda.sh
conda activate project
cd /home/users/ybi3/mechinterp_brain
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/phase5c_das_depth.py --steps 120 --out outputs/sae_ckpts/phase5c_das_depth.csv
echo "=== DONE ==="
