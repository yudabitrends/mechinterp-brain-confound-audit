#!/usr/bin/env python
"""Reviewer #3: does the low-dimensional causal compression transfer to an unseen scanner on the PRIMARY
multimodal ViT (not just the cheap FNC net)? We reuse the already-trained site-LOSO multimodal checkpoints
(outputs/p6c_t4_site_loso/<S>/best.pt, each = the full sMRI+FNC cross-attention ViT trained leaving out site S)
-- no retraining needed. For held-out site S: extract that model's fused decision reps (done by extract_fused
with --tag LOSO_<S>_), fit the scanner probe + k=1 DAS on the SEEN sites only, and evaluate the interchange-IIA
on pairs whose unseen side is a never-trained site-S subject (paired against seen opposite-population subjects).
High IIA + disease preserved + low random-rotation floor => the causal axis transfers to a scanner the MAIN model
never saw. Run in `project`. Appends outputs/sae_ckpts/multimodal_loso_iia.csv.
"""
import os, sys, argparse
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mib import das as D
from ctrl_das_null import load_head

P6C = "/home/users/ybi3/MultiViT2/outputs/p6c_t4_site_loso"


def dpairs(base_pool, src_pool, n, seed):
    g = torch.Generator().manual_seed(seed)
    return base_pool[torch.randint(len(base_pool), (n,), generator=g)], src_pool[torch.randint(len(src_pool), (n,), generator=g)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--site", required=True); args = ap.parse_args()
    S = args.site
    d = torch.load(f"outputs/activations/fused_LOSO_{S}_ALL.pt", weights_only=True)
    h = d["fused"].float(); pop = np.asarray(d["population"]); site = np.asarray(d["site"])
    yp = (pop == "China").astype(int); scanner = torch.tensor(yp)
    hW, hb = load_head(f"{P6C}/{S}/best.pt")
    R = h.numpy()
    seen = site != S
    held = np.where(site == S)[0]
    if len(held) < 12:
        print(f"{S}: too few held-out ({len(held)})"); return
    held_pop = "China" if yp[held][0] == 1 else "US"
    # fit scanner probe on SEEN subjects only
    seenidx = np.where(seen)[0]
    lr = LogisticRegression(max_iter=3000).fit(R[seenidx], yp[seenidx])
    sw = torch.tensor(lr.coef_.ravel(), dtype=torch.float32); sb = torch.tensor(float(lr.intercept_[0]))
    # seen pools by opposite population to pair against the held-out site
    opp = 0 if held_pop == "China" else 1                       # seen source is the OPPOSITE population
    seen_opp = np.where(seen & (yp == opp))[0]
    seen_same = np.where(seen & (yp == (1 - opp)))[0]
    # decodability of population on the unseen site vs seen opposite (frozen probe)
    deci = np.concatenate([held, seen_opp])
    dec = roc_auc_score(yp[deci], (h[torch.tensor(deci)] @ sw + sb).numpy()); dec = max(dec, 1 - dec)
    # DAS trained on SEEN cross-population pairs; eval on UNSEEN-site pairs
    bt, st = dpairs(torch.tensor(seen_same), torch.tensor(seen_opp), 3000, 0)
    heldT, oppT = torch.tensor(held), torch.tensor(seen_opp)
    bv, sv = dpairs(heldT, oppT, 3000, 100)                     # base = unseen site, source = seen opposite pop
    hh = torch.tensor(R, dtype=torch.float32)
    das = D.train_das(hh, scanner, hW, hb, sw, sb, bt, st, k=1, steps=500, lr=5e-3, lam=1.0, seed=0, device="cpu")
    m = D.eval_iia(das, hh, scanner, hW, hb, sw, sb, bv, sv, device="cpu")
    # random-rotation floor on the same unseen-site eval pairs (untrained DASRotation)
    fl = D.eval_iia(D.DASRotation(hh.shape[1], 1, seed=1), hh, scanner, hW, hb, sw, sb, bv, sv, device="cpu")
    row = {"held_out_site": S, "held_pop": held_pop, "n_unseen": int(len(held)),
           "unseen_decode_auc": round(dec, 3), "unseen_iia": round(m["scanner_iia"], 3),
           "disease_preserved": round(m["disease_preserved"], 3), "rand_floor": round(fl["scanner_iia"], 3)}
    print(row, flush=True)
    p = "outputs/sae_ckpts/multimodal_loso_iia.csv"
    df = pd.DataFrame([row])
    df.to_csv(p, mode="a", header=not os.path.exists(p), index=False)


if __name__ == "__main__":
    main()
