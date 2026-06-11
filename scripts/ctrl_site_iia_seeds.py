#!/usr/bin/env python
"""Seed/placebo interval on the small-n within-country site-IIA (reviewer P1 round-2: the headline
within-country causal number lacked an uncertainty interval, unlike the population axis).

For US COBRE-vs-Scanner2 and China GZ-vs-ZMD: run the k=1 site-axis DAS over several seeds (mean±SD)
and a placebo (site labels permuted within the contrast -> expect the random/no-op floor). Cached fused
+ frozen head; no model forward. Writes outputs/sae_ckpts/site_iia_seeds.csv.
"""
import os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mib import das as D
from ctrl_das_null import load_head, make_pairs
MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"


def iia_for(h, axis, hW, hb, tri, tei, seed, steps, pairs):
    lr = LogisticRegression(max_iter=3000).fit(h[tri].numpy(), axis[tri].numpy())
    aw = torch.tensor(lr.coef_.ravel(), dtype=torch.float32); ab = torch.tensor(float(lr.intercept_[0]))
    bt, st = make_pairs(axis, tri, pairs, seed); bv, sv = make_pairs(axis, tei, pairs, seed + 100)
    das = D.train_das(h, axis, hW, hb, aw, ab, bt, st, k=1, steps=steps, lr=5e-3, lam=1.0, seed=seed, device="cpu")
    return D.eval_iia(das, h, axis, hW, hb, aw, ab, bv, sv, device="cpu")


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--pairs", type=int, default=3000); args = ap.parse_args()
    d = torch.load("outputs/activations/fused_HOLD_ALL.pt", weights_only=True)
    h = d["fused"].float(); pop = np.asarray(d["population"]); site = np.asarray(d["site"]); split = np.asarray(d["split"])
    hW, hb = load_head(MHOLD)
    rows = []
    for name, sites, posv in [("US_COBRE_vs_Scanner2", ["COBRE", "Scanner2"], "Scanner2"),
                              ("China_GZ_vs_ZMD", ["GZ", "ZMD"], "ZMD")]:
        keep = np.isin(site, sites); tr = (split == "train") & keep; te = (split == "test") & keep
        axis = torch.tensor((site == posv).astype(int))
        tri = torch.tensor(np.where(tr)[0]); tei = torch.tensor(np.where(te)[0])
        iias, press = [], []
        for s in [int(x) for x in args.seeds.split(",")]:
            m = iia_for(h, axis, hW, hb, tri, tei, s, args.steps, args.pairs)
            iias.append(m["scanner_iia"]); press.append(m["disease_preserved"])
            print(f"{name} seed={s} IIA={m['scanner_iia']:.3f} pres={m['disease_preserved']:.3f}", flush=True)
        # placebo: permute site labels within the contrast (balanced), expect floor
        g = torch.Generator().manual_seed(99); perm = torch.randperm(int(keep.sum()), generator=g)
        plac = axis.clone(); idxk = np.where(keep)[0]; plac[idxk] = axis[idxk][perm]
        mp = iia_for(h, plac, hW, hb, tri, tei, 0, args.steps, args.pairs)
        rows.append({"contrast": name, "iia_mean": round(np.mean(iias), 3), "iia_sd": round(np.std(iias), 3),
                     "disease_pres_mean": round(np.mean(press), 3), "placebo_iia": round(mp["scanner_iia"], 3),
                     "n_seeds": len(iias)})
        print(rows[-1], flush=True); pd.DataFrame(rows).to_csv("outputs/sae_ckpts/site_iia_seeds.csv", index=False)
    print("wrote outputs/sae_ckpts/site_iia_seeds.csv")


if __name__ == "__main__":
    main()
