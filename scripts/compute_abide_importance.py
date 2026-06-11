#!/usr/bin/env python
"""Cross-disorder (autism, ABIDE) scanner/site-edge importance for the brain figures: |L2-logistic coef| on the
raw ABIDE Neuromark FNC for the two largest ABIDE sites (a within-disorder between-scanner contrast, parallel to
the SZ within-country axes) and for disease (autism vs HC). Descriptive localization. Run in `project`.
Writes outputs/fnc_importance_abide/.
"""
import os, sys, json, shutil
import numpy as np, scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mib.abide_data import build_abide_manifest, A1, A2

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "fnc_importance_abide")
COORDS = "/home/users/ybi3/MultiViT2/outputs/p_fnc_edge_importance/icn_mni_coords.npy"


def importance(X, y):
    return np.abs(LogisticRegression(solver="liblinear", max_iter=4000, C=1.0).fit(X, y).coef_.ravel())


def main():
    os.makedirs(OUT, exist_ok=True)
    df = build_abide_manifest()
    fnc = {"ABIDE_I": np.asarray(sio.loadmat(A1["mat"])["sFNC"]),
           "ABIDE_II": np.asarray(sio.loadmat(A2["mat"])["sFNC"])}
    iu = np.triu_indices(53, 1)
    X = np.stack([fnc[r.cohort][r.sfnc_idx][iu] for _, r in df.iterrows()]).astype(np.float64)
    site = df.site.to_numpy(); dx = df.Diagnosis.to_numpy().astype(int)
    rng = np.random.RandomState(0); tr = rng.rand(len(df)) < 0.7

    # scanner axis = two largest sites (binary, within-disorder between-scanner)
    top2 = [s for s, _ in sorted(((s, (site == s).sum()) for s in set(site)), key=lambda t: -t[1])[:2]]
    m = np.isin(site, top2); ysc = (site == top2[1]).astype(int)
    Xs = StandardScaler().fit(X[m & tr]).transform(X)
    sc = importance(Xs[m & tr], ysc[m & tr])
    Xd = StandardScaler().fit(X[tr]).transform(X)
    di = importance(Xd[tr], dx[tr])
    np.save(f"{OUT}/scanner_coef_abs.npy", sc.astype(np.float32))
    np.save(f"{OUT}/disease_coef_abs.npy", di.astype(np.float32))
    np.save(f"{OUT}/scanner_top100.npy", np.argsort(-sc)[:100].astype(np.int32))
    np.save(f"{OUT}/disease_top100.npy", np.argsort(-di)[:100].astype(np.int32))
    shutil.copy(COORDS, f"{OUT}/icn_mni_coords.npy")
    json.dump({"n": int(len(df)), "scanner_sites": top2, "n_scanner_pair": int(m.sum()),
               "scanner_max": round(float(sc.max()), 3), "disease_max": round(float(di.max()), 3)},
              open(f"{OUT}/meta.json", "w"), indent=2)
    print(f"ABIDE N={len(df)} scanner-pair={top2} ({int(m.sum())}); scanner max|coef|={sc.max():.3f}, "
          f"disease max={di.max():.3f}; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
