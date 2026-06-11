#!/usr/bin/env python
"""A2 multi-seed replication with CIs (reviewer #7): the cross-architecture IIA values were single-run, so
"consistent across models" was informal and the Transformer 0.76 had no uncertainty. Re-run the k=1 DAS
interchange over 3 seeds (model-init seed + DAS seed) per model; report mean +- SD and a 95% normal CI, and
whether each model's IIA CI excludes its random-rotation floor. Reuses cross_arch + ctrl_crossarch_das + mib.das.
Run in `project`. Writes outputs/sae_ckpts/multiseed_crossarch.csv.
"""
import os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mib import das as D
from cross_arch import build_fnc, MLP, BrainNetCNN, TransformerBaseline
from ctrl_crossarch_das import train_with_head, make_pairs

SEEDS = [0, 1, 2]


def main():
    mats, X, ydx, split, pop, site = build_fnc()
    Xs = StandardScaler().fit(X[split == "train"]).transform(X).astype(np.float32)
    Ms = ((mats - mats[split == "train"].mean(0)) / (mats[split == "train"].std(0) + 1e-6)).astype(np.float32)[:, None]
    tr = split == "train"; keep = np.isin(pop, ["US", "China"]); yp = (pop == "China").astype(int)
    scanner = torch.tensor(yp); trk = np.where(tr & keep)[0]; tek = np.where((~tr) & keep)[0]
    specs = {"MLP": lambda: (MLP(X.shape[1]), Xs), "BrainNetCNN": lambda: (BrainNetCNN(), Ms),
             "Transformer": lambda: (TransformerBaseline(), Ms[:, 0])}
    rows = []
    for name, mk in specs.items():
        iias, flrs = [], []
        for s in SEEDS:
            net, feat = mk()
            R, hW, hb = train_with_head(net, feat, ydx, trk if False else np.where(tr)[0], seed=s)
            h = torch.tensor(R, dtype=torch.float32)
            lr = LogisticRegression(max_iter=3000).fit(R[trk], yp[trk])
            sw = torch.tensor(lr.coef_.ravel(), dtype=torch.float32); sb = torch.tensor(float(lr.intercept_[0]))
            bt, st = make_pairs(scanner, torch.tensor(trk), 3000, s)
            bv, sv = make_pairs(scanner, torch.tensor(tek), 3000, s + 100)
            das = D.train_das(h, scanner, hW, hb, sw, sb, bt, st, k=1, steps=600, lr=5e-3, lam=1.0, seed=s, device="cpu")
            iias.append(D.eval_iia(das, h, scanner, hW, hb, sw, sb, bv, sv, device="cpu")["scanner_iia"])
            flrs.append(D.eval_iia(D.DASRotation(h.shape[1], k=1, seed=s + 7), h, scanner, hW, hb, sw, sb, bv, sv,
                                   device="cpu")["scanner_iia"])
        a = np.array(iias); f = np.array(flrs); ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a))
        rows.append({"model": name, "iia_mean": round(a.mean(), 3), "iia_sd": round(a.std(ddof=1), 3),
                     "iia_ci95_lo": round(a.mean() - ci, 3), "iia_ci95_hi": round(a.mean() + ci, 3),
                     "floor_mean": round(f.mean(), 3), "ci_excludes_floor": bool(a.mean() - ci > f.mean())})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv("outputs/sae_ckpts/multiseed_crossarch.csv", index=False)
    print("\nAll models: IIA CI lower bound vs floor ->", [(r["model"], r["ci_excludes_floor"]) for r in rows])


if __name__ == "__main__":
    main()
