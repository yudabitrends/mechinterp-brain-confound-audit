#!/usr/bin/env python
"""Per-edge importance for the within-country (between-scanner) site axes, for the brain figures: US
COBRE-vs-Scanner2 and China GZ-vs-ZMD. |L2-logistic coef| on standardized 1378-edge vectors, fit on train
subjects of each pair. Same schema as compute_fnc_importance. Run in `project`. Writes outputs/fnc_importance_site/.
"""
import os, sys, json, shutil
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from cross_arch import build_fnc

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "fnc_importance_site")
COORDS = "/home/users/ybi3/MultiViT2/outputs/p_fnc_edge_importance/icn_mni_coords.npy"
PAIRS = {"US_COBRE_vs_Scanner2": ("COBRE", "Scanner2"), "China_GZ_vs_ZMD": ("GZ", "ZMD")}


def importance(X, y):
    return np.abs(LogisticRegression(solver="liblinear", max_iter=4000, C=1.0).fit(X, y).coef_.ravel())


def main():
    os.makedirs(OUT, exist_ok=True)
    mats, X, ydx, split, pop, site = build_fnc()
    tr = split == "train"
    meta = {}
    for name, (a, b) in PAIRS.items():
        m = np.isin(site, [a, b])
        y = (site == b).astype(int)
        Xs = StandardScaler().fit(X[m & tr]).transform(X)
        imp = importance(Xs[m & tr], y[m & tr])
        np.save(f"{OUT}/{name}_coef_abs.npy", imp.astype(np.float32))
        np.save(f"{OUT}/{name}_top100.npy", np.argsort(-imp)[:100].astype(np.int32))
        meta[name] = {"n_train": int((m & tr).sum()), "n_total": int(m.sum()),
                      "max_coef": round(float(imp.max()), 3)}
        print(f"{name}: n_train={meta[name]['n_train']} max|coef|={meta[name]['max_coef']}", flush=True)
    shutil.copy(COORDS, f"{OUT}/icn_mni_coords.npy")
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
