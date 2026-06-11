#!/usr/bin/env python
"""Bootstrap 95% CIs (1000 resamples) for the headline held-out AUCs: the harmonization
comparison (Table III) and the modality-dissociation headline. Transforms/probes are fit once
on TRAIN; only the held-out TEST evaluation is resampled (standard AUC bootstrap). Writes
outputs/sae_ckpts/bootstrap_ci.csv.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
import importlib.util, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
_spec = importlib.util.spec_from_file_location("hc", os.path.join(os.path.dirname(__file__), "harmonize_compare.py"))
hc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(hc)   # inlp, random_proj, combat


def boot_ci(ytr, ptr_X, yte, pte_X, n=1000, seed=0):
    """Fit a probe on (ptr_X, ytr), score pte_X, bootstrap test AUC. Returns (point, lo, hi)."""
    sc = StandardScaler().fit(ptr_X)
    clf = LogisticRegression(solver="liblinear", max_iter=2000).fit(sc.transform(ptr_X), ytr)
    p = clf.predict_proba(sc.transform(pte_X))[:, 1]
    point = roc_auc_score(yte, p)
    rng = np.random.default_rng(seed); m = len(yte); a = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        if len(np.unique(yte[idx])) == 2:
            a.append(roc_auc_score(yte[idx], p[idx]))
    a = np.array(a)
    return point, float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    rows = []
    # ---------- harmonization (Table III) on the fused rep ----------
    d = torch.load("outputs/activations/fused_HOLD_ALL.pt", weights_only=True)
    X = d["fused"].numpy(); ydx = np.asarray(d["y_dx"]); pop = np.asarray(d["population"]); site = np.asarray(d["site"])
    split = np.asarray(d["split"]); tr, te = split == "train", split == "test"
    yp = (pop == "China").astype(int)
    Xtr, Xte = X[tr], X[te]
    methods = {"raw": (Xtr, Xte)}
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(site[tr].reshape(-1, 1))
    lr = LinearRegression().fit(enc.transform(site[tr].reshape(-1, 1)), Xtr)
    methods["site-regression"] = (Xtr - lr.predict(enc.transform(site[tr].reshape(-1, 1))),
                                  Xte - lr.predict(enc.transform(site[te].reshape(-1, 1))))
    P = hc.inlp(Xtr, yp[tr], 60)   # operating point where scanner is near-chance AND disease preserved
    methods["INLP"] = (Xtr @ P, Xte @ P)
    Pr = hc.random_proj(Xtr.shape[1], 20, 0)
    methods["random"] = (Xtr @ Pr, Xte @ Pr)
    try:
        methods["ComBat"] = hc.combat(Xtr, Xte, site[tr], site[te], ydx[tr], ydx[te])
    except Exception as e:
        print("[ComBat] failed", e); methods["ComBat"] = None
    for name, pair in methods.items():
        if pair is None:
            continue
        a, b = pair
        for tgt, ytr_, yte_ in [("scanner", yp[tr], yp[te]), ("disease", ydx[tr], ydx[te])]:
            pt, lo, hi = boot_ci(ytr_, a, yte_, b)
            rows.append({"group": "harmonization", "method": name, "metric": tgt,
                         "auc": round(pt, 3), "lo": round(lo, 3), "hi": round(hi, 3)})
            print(f"harm {name:16s} {tgt:8s} {pt:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)

    # ---------- modality-dissociation headline (SAE-feature probe on CLS) ----------
    from mib.sae import SAEConfig, SparseAutoencoder
    from mib import probe as P_
    acts = torch.load("outputs/activations/acts_HOLD_ALL_cls.pt", weights_only=True)
    lab = pd.read_csv("outputs/activations/labels_HOLD_ALL.csv")
    trm = (lab["split"] == "train").to_numpy(); tem = (lab["split"] == "test").to_numpy()
    ydx2 = lab["label"].to_numpy(); yp2 = (lab["population"] == "China").astype(int).to_numpy()

    def load_sae(hook):
        c = torch.load(f"outputs/sae_ckpts/sae_HOLDtrain_{hook}_seed0.pt", weights_only=True, map_location="cpu")
        s = SparseAutoencoder(SAEConfig(**c["cfg"])); s.load_state_dict(c["state_dict"]); return s.eval()

    for hook, br in [("sMRI_encoder.norm", "sMRI"), ("sFNC_encoder.norm", "FNC")]:
        F = P_.encode_features(load_sae(hook), acts[hook].numpy())
        for tgt, y in [("scanner", yp2), ("disease", ydx2)]:
            pt, lo, hi = boot_ci(y[trm], F[trm], y[tem], F[tem])
            rows.append({"group": "dissociation", "method": br, "metric": tgt,
                         "auc": round(pt, 3), "lo": round(lo, 3), "hi": round(hi, 3)})
            print(f"diss {br:6s} {tgt:8s} {pt:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)

    pd.DataFrame(rows).to_csv("outputs/sae_ckpts/bootstrap_ci.csv", index=False)
    print("wrote bootstrap_ci.csv")


if __name__ == "__main__":
    main()
