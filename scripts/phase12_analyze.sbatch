#!/bin/bash
#SBATCH --job-name=mib_p12
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/p12_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/p12_%j.err

# Runs after the token SAE training finishes (submit with --dependency=afterok:<jobid>).
# Phase-1 gate (seed-stability) + Phase-2 scanner-vs-disease probing depth profile.
set -e
source activate project
cd /home/users/ybi3/mechinterp_brain
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src

echo "=== PHASE 1 GATE: seed-stable features per hook ==="
python scripts/phase1_gate.py --tag ALLtok --threshold 0.7

echo ""
echo "=== PHASE 2: scanner-vs-disease probing (seed 0) ==="
python scripts/phase2_probe.py --tag ALLtok --seed 0 --min-site-n 30

echo "=== DONE: phase1_gate_ALLtok.csv + phase2_probe_ALLtok_seed0.csv ==="
