#!/usr/bin/env python
"""Per-edge scanner (US-vs-China) and disease (SZ-vs-HC) importance on THIS paper's held-out FNC,
for the functional brain panel. |coef| of an L2-logistic on standardized 1378-edge vectors (multivariate
importance, accounting for collinearity); top-100 each. Saves to outputs/fnc_importance/ in the same
schema the brain-atlas renderer expects. Run in the `project` conda env. CPU."""
import os, sys, json, shutil
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from cross_arch import build_fnc

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "fnc_importance")
COORDS_SRC = "/home/users/ybi3/MultiViT2/outputs/p_fnc_edge_importance/icn_mni_coords.npy"


def importance(X, y):
    w = LogisticRegression(solver="liblinear", max_iter=4000, C=1.0).fit(X, y).coef_.ravel()
    return np.abs(w)


def main():
    os.makedirs(OUT, exist_ok=True)
    mats, X, ydx, split, pop, site = build_fnc()
    tr = split == "train"; keep = np.isin(pop, ["US", "China"])
    yscan = (pop == "China").astype(int)
    Xs = StandardScaler().fit(X[tr]).transform(X)             # standardize on train, apply all
    trk = tr & keep
    scanner = importance(Xs[trk], yscan[trk])                 # US vs China
    disease = importance(Xs[tr], ydx[tr])                     # SZ vs HC
    np.save(f"{OUT}/scanner_coef_abs.npy", scanner.astype(np.float32))
    np.save(f"{OUT}/disease_coef_abs.npy", disease.astype(np.float32))
    np.save(f"{OUT}/scanner_top100.npy", np.argsort(-scanner)[:100].astype(np.int32))
    np.save(f"{OUT}/disease_top100.npy", np.argsort(-disease)[:100].astype(np.int32))
    shutil.copy(COORDS_SRC, f"{OUT}/icn_mni_coords.npy")
    iu = np.triu_indices(53, 1)
    inter = len(set(np.argsort(-scanner)[:100]) & set(np.argsort(-disease)[:100]))
    json.dump({"n_subjects": int(trk.sum()), "n_icns": 53, "n_edges": int(len(scanner)),
               "scanner_axis": "US_vs_China (binary)", "disease_axis": "SZ_vs_HC",
               "edge_convention": "np.triu_indices(53, k=1)", "penalty": "l2",
               "jaccard_top100_l2": inter / (200 - inter)}, open(f"{OUT}/meta.json", "w"), indent=2)
    print(f"scanner top edge |coef| max={scanner.max():.3f}; disease max={disease.max():.3f}; "
          f"top100 overlap={inter}; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
