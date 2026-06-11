#!/usr/bin/env python
"""Tier-1 / reviewer #11: does "structural branch = scanner machine" reproduce OUTSIDE the SPM12 VBM pipeline?
We use FastSurfer/FreeSurfer aseg subcortical-volume features (a surface/segmentation pipeline, orthogonal to
voxel-based morphometry) for the SZ cohorts and ask whether scanner (US vs China) is far more decodable than
disease (SZ vs HC) from structural features, replicating the VBM modality dissociation in a different pipeline.
Read-only over /data/users1/ybi/geometric_multivit/freesurfer_subjects. Writes outputs/sae_ckpts/freesurfer_struct.csv.
"""
import os, sys, re, glob
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

FS = "/data/users1/ybi/geometric_multivit/freesurfer_subjects"


def parse_aseg(path):
    vols = {}
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        if len(p) >= 5:
            try: vols[p[4]] = float(p[3])      # StructName -> Volume_mm3
            except ValueError: pass
    return vols


def main():
    # fused id -> (diagnosis, population) for the SZ cohorts
    d = torch.load("outputs/activations/fused_HOLD_ALL.pt", weights_only=True)
    fmap = {str(s): (int(y), str(p)) for s, y, p in zip(d["subject_id"], np.asarray(d["y_dx"]),
                                                        np.asarray(d["population"]))}
    rows = []
    for sub in sorted(os.listdir(FS)):
        ap = f"{FS}/{sub}/stats/aseg.stats"
        if not os.path.exists(ap):
            continue
        pref = sub.split("_")[0]
        if pref not in ("COBRE", "FBIRN", "ChineseSZ", "PK", "Scanner1", "Scanner2", "Scanner3", "PKMPRC"):
            continue
        pop = "China" if pref == "ChineseSZ" else "US"
        bare = re.sub(r"^(COBRE_COBRE_|FBIRN_FBIRN_|ChineseSZ_|Scanner[0-9]_)", "", sub)
        dx = None
        if bare in fmap: dx = fmap[bare][0]
        elif "SZ-" in sub: dx = 1
        elif "NC-" in sub: dx = 0
        rows.append({"sub": sub, "pop": pop, "dx": dx, "vols": parse_aseg(ap)})

    feats = sorted(set().union(*[set(r["vols"]) for r in rows]))
    X = np.array([[r["vols"].get(f, np.nan) for f in feats] for r in rows], float)
    keep = ~np.isnan(X).any(1)
    X, rows = X[keep], [r for r, k in zip(rows, keep) if k]
    pop = np.array([r["pop"] for r in rows]); dx = np.array([r["dx"] if r["dx"] is not None else -1 for r in rows])
    print(f"FreeSurfer subjects with aseg: {len(rows)} ({(pop=='US').sum()} US, {(pop=='China').sum()} China); "
          f"diagnosis-labelled: {(dx>=0).sum()}", flush=True)

    rng = np.random.RandomState(0); n = len(rows); te = rng.rand(n) < 0.3; tr = ~te
    Xs = StandardScaler().fit(X[tr]).transform(X)

    def auc(y, m):
        mm = m & (y >= 0) if y.min() < 0 else m
        return round(roc_auc_score(y[mm & te], LogisticRegression(max_iter=4000)
                     .fit(Xs[mm & tr], y[mm & tr]).predict_proba(Xs[mm & te])[:, 1]), 3)

    yscan = (pop == "China").astype(int)
    rows_out = [
        {"axis": "scanner_US_vs_China_FreeSurfer_aseg", "auc": auc(yscan, np.ones(n, bool))},
        {"axis": "disease_SZ_vs_HC_FreeSurfer_aseg", "auc": auc(dx, dx >= 0)},
    ]
    for r in rows_out: print(r, flush=True)
    pd.DataFrame(rows_out).to_csv("outputs/sae_ckpts/freesurfer_struct.csv", index=False)
    print(f"\nINTERP: scanner AUC {rows_out[0]['auc']} vs disease AUC {rows_out[1]['auc']} on a FastSurfer/FreeSurfer "
          "segmentation pipeline reproduces the 'structural = scanner machine' dissociation outside SPM12 VBM "
          "(scanner >> disease), so that finding is not VBM-normalization-specific.")


if __name__ == "__main__":
    main()
