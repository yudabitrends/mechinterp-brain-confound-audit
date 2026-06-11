#!/bin/bash
#SBATCH --job-name=mib_p6sweep
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/p6sweep_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/p6sweep_%j.err
set -e
source /home/users/ybi3/anaconda3/etc/profile.d/conda.sh
conda activate project
cd /home/users/ybi3/mechinterp_brain
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/phase6b_count_sweep.py --out outputs/sae_ckpts/phase6_count_sweep.csv
echo "=== DONE ==="
