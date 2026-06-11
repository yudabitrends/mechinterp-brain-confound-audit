#!/bin/bash
#SBATCH --job-name=mib_hold
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/users/ybi3/mechinterp_brain/outputs/logs/hold_%j.out
#SBATCH --error=/home/users/ybi3/mechinterp_brain/outputs/logs/hold_%j.err
set -e
source activate project
cd /home/users/ybi3/mechinterp_brain
export PYTHONPATH=src:/home/users/ybi3/MultiViT2/geometric_multivit/src
CK=/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt
SI=/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/split_info.json
A=outputs/activations
COH="COBRE FBIRN ChineseSZ PK_MPRC"

echo "=== [1] extract TRAIN tokens (for SAE) + ALL cls (for probing) through M_hold ==="
for c in $COH; do
  python scripts/extract_activations.py --cohort $c --ckpt $CK --position tokens \
      --tag HOLDtr_ --split-info $SI --which train --device cuda --out $A
  python scripts/extract_activations.py --cohort $c --ckpt $CK --position cls \
      --tag HOLD_ --split-info $SI --which all --device cuda --out $A
done

echo "=== [2] concat ==="
python - <<'PY'
import torch, pandas as pd
A="outputs/activations"; COH=["COBRE","FBIRN","ChineseSZ","PK_MPRC"]
def cat(tag,pos):
    acc={}
    for c in COH:
        a=torch.load(f"{A}/acts_{tag}{c}_{pos}.pt", weights_only=True)
        for k,v in a.items(): acc.setdefault(k,[]).append(v)
    out={k:torch.cat(v,0) for k,v in acc.items()}
    torch.save(out,f"{A}/acts_{tag}ALL_{pos}.pt"); return out
cat("HOLDtr_","tokens")
cat("HOLD_","cls")
pd.concat([pd.read_csv(f"{A}/labels_HOLD_{c}.csv") for c in COH], ignore_index=True)\
  .to_csv(f"{A}/labels_HOLD_ALL.csv", index=False)
print("concat done")
PY

echo "=== [3] train SAE atlas on TRAIN tokens ==="
python scripts/train_sae.py --acts $A/acts_HOLDtr_ALL_tokens.pt --cohort HOLDtrain \
    --arch topk --k 32 --expansion 16 --seeds 0 1 2 --epochs 30 --batch 1024 --device cuda

echo "=== [4] Phase-1 gate ==="
python scripts/phase1_gate.py --tag HOLDtrain --threshold 0.7

echo "=== [5] Phase-2 probing: held-out TEST and TRAIN ==="
python scripts/phase2_probe.py --cls-acts $A/acts_HOLD_ALL_cls.pt --labels $A/labels_HOLD_ALL.csv \
    --tag HOLDtrain --seed 0 --min-site-n 20 --split test --out-suffix _test
python scripts/phase2_probe.py --cls-acts $A/acts_HOLD_ALL_cls.pt --labels $A/labels_HOLD_ALL.csv \
    --tag HOLDtrain --seed 0 --min-site-n 20 --split train --out-suffix _train

echo "=== [6] extract fused (M_hold, with split) ==="
python scripts/extract_fused.py --ckpt $CK --tag HOLD_ --split-info $SI

echo "=== [7] Stage B harmonization comparison (held-out) ==="
python scripts/harmonize_compare.py --fused $A/fused_HOLD_ALL.pt

echo "=== [8] Stage C stratified analysis ==="
python scripts/stratified_analysis.py --fused $A/fused_HOLD_ALL.pt

echo "=== DONE ==="
