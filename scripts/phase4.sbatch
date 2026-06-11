#!/bin/bash
#SBATCH --job-name=mib_p4
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/p4_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/p4_%j.err
set -e
source activate project
cd /home/users/ybi3/mechinterp_brain
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src
# cap BLAS threads so the sklearn probes don't fan out across the node (engineering guard)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python scripts/phase4_patch_fusion.py --out outputs/sae_ckpts
echo "=== DONE ==="
