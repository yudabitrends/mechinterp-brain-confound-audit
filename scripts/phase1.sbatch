#!/bin/bash
#SBATCH --job-name=mib_phase1
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/phase1_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/phase1_%j.err

# Phase 1: extract MultiViT2 activations (canonical pooled model M*) across 4 cohorts,
# concat into one cross-site set, then train a multi-seed SAE atlas (TopK, k=32, x16).
set -e
source activate project
cd /home/users/ybi3/mechinterp_brain
mkdir -p outputs/logs outputs/activations outputs/sae_ckpts outputs/cache
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src

CANON=/home/users/ybi3/MultiViT2/outputs/p6c_t4_site_loso/FBIRN_site12/best.pt
COHORTS="COBRE FBIRN ChineseSZ PK_MPRC"

echo "=== [1/3] extract activations through canonical model M* ==="
for c in $COHORTS; do
  python scripts/extract_activations.py --cohort $c --ckpt $CANON \
      --position cls --device cuda --batch-size 16 --out outputs/activations
done

echo "=== [2/3] concat cohorts -> cross-site activation set ==="
python - <<'PY'
import glob, torch, pandas as pd, os
A="outputs/activations"
cohorts=["COBRE","FBIRN","ChineseSZ","PK_MPRC"]
acc={}; metas=[]
for c in cohorts:
    a=torch.load(f"{A}/acts_{c}_cls.pt", weights_only=True)
    metas.append(pd.read_csv(f"{A}/labels_{c}.csv"))
    for k,v in a.items():
        acc.setdefault(k,[]).append(v)
acts={k:torch.cat(v,0) for k,v in acc.items()}
torch.save(acts, f"{A}/acts_ALL_cls.pt")
pd.concat(metas, ignore_index=True).to_csv(f"{A}/labels_ALL.csv", index=False)
n=next(iter(acts.values())).shape[0]
print(f"concat: {len(acts)} hook points x {n} subjects -> acts_ALL_cls.pt")
PY

echo "=== [3/3] train multi-seed SAE atlas (TopK k=32, expansion 16) ==="
python scripts/train_sae.py --acts outputs/activations/acts_ALL_cls.pt --cohort ALL \
    --arch topk --k 32 --expansion 16 --seeds 0 1 2 --epochs 200 --batch 256 \
    --device cuda --out outputs/sae_ckpts

echo "=== DONE: metrics at outputs/sae_ckpts/metrics_ALL.csv ==="
