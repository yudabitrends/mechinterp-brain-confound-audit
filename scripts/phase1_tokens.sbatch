#!/bin/bash
#SBATCH --job-name=mib_p1tok
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/p1tok_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/p1tok_%j.err

# Phase 1 (token-level): the data-rich SAE atlas. Extract ALL token positions through M*
# (sMRI 253 tok/subj, FNC 54) -> ~333k / ~71k vectors per hook -> train multi-seed TopK SAE.
# CLS activations (acts_*_cls.pt) are retained separately for Phase-2 probing.
set -e
source activate project
cd /home/users/ybi3/mechinterp_brain
mkdir -p outputs/logs outputs/activations outputs/sae_ckpts outputs/cache
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src

CANON=/home/users/ybi3/MultiViT2/outputs/p6c_t4_site_loso/FBIRN_site12/best.pt
COHORTS="COBRE FBIRN ChineseSZ PK_MPRC"

echo "=== [1/3] extract TOKEN activations through M* ==="
for c in $COHORTS; do
  python scripts/extract_activations.py --cohort $c --ckpt $CANON \
      --position tokens --device cuda --batch-size 16 --out outputs/activations
done

echo "=== [2/3] concat token activations (per-hook, acts only) ==="
python - <<'PY'
import torch
A="outputs/activations"
cohorts=["COBRE","FBIRN","ChineseSZ","PK_MPRC"]
acc={}
for c in cohorts:
    a=torch.load(f"{A}/acts_{c}_tokens.pt", weights_only=True)
    for k,v in a.items():
        acc.setdefault(k,[]).append(v)
acts={k:torch.cat(v,0) for k,v in acc.items()}
torch.save(acts, f"{A}/acts_ALL_tokens.pt")
for k in sorted(acts)[:2]+sorted(acts)[-1:]:
    print(f"  {k:30s} {tuple(acts[k].shape)}")
print("hooks:", len(acts))
PY

echo "=== [3/3] train multi-seed TopK SAE atlas on token activations ==="
python scripts/train_sae.py --acts outputs/activations/acts_ALL_tokens.pt --cohort ALLtok \
    --arch topk --k 32 --expansion 16 --seeds 0 1 2 --epochs 30 --batch 1024 \
    --device cuda --out outputs/sae_ckpts

echo "=== DONE: metrics at outputs/sae_ckpts/metrics_ALLtok.csv ==="
