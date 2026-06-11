#!/usr/bin/env python
"""Stage B+ : cross-attention path-patching 'crime-scene' test (held-out, M_hold).

Which architectural path carries the scanner confound into the disease decision? We
mean-patch a branch's CLS token (overwrite with the TRAIN global-mean CLS -> lesion that
path's between-subject signal) and measure what the FUSED decision rep loses on held-out
test. If lesioning the sMRI path collapses scanner-AUC while sparing disease, the confound
enters the decision through the structural branch -> a causal localization (activation
patching), complementing the linear readout erasure. Writes outputs/sae_ckpts/path_patch.csv.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
import argparse, json, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from geomultivit.data.dataset import MultiModalH5Dataset
from geomultivit.data.multicohort import load_geometric_cohort
from mib.extract import load_model
from mib import patch

H5 = "/home/users/ybi3/data/szdataset_modified.h5"
COH = ["COBRE", "FBIRN", "ChineseSZ", "PK_MPRC"]


def cv_auc(X, y, k=5, seed=0):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    Xs = StandardScaler().fit_transform(X)
    proba = cross_val_predict(LogisticRegression(solver="liblinear", max_iter=2000),
                              Xs, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                              method="predict_proba")
    return float(roc_auc_score(y, proba[:, 1]))


def loader_for(df, ids, bs=16):
    sub = df[df["SubjectID"].isin(set(ids))].copy()
    ds = MultiModalH5Dataset(sub, volume_shape=(96, 112, 96), num_icns=53,
                             require_sMRI=True, cache_dir="outputs/cache", augment=False)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt")
    ap.add_argument("--split-info", default="/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/split_info.json")
    ap.add_argument("--out", default="outputs/sae_ckpts/path_patch.csv")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    si = json.load(open(args.split_info))
    gdf = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False).df
    gdf = gdf[gdf["cohort"].isin(COH)]
    model = load_model(args.ckpt, device=dev)
    tr_loader = loader_for(gdf, si["train_ids"])
    te_loader = loader_for(gdf, si["test_ids"])

    # TRAIN global-mean CLS at each branch output -> the "lesion" value
    SM, FN = "sMRI_encoder.norm", "sFNC_encoder.norm"
    sm_tok, _ = patch.capture_hook_tokens(model, tr_loader, dev, SM)   # (N,T,256)
    fn_tok, _ = patch.capture_hook_tokens(model, tr_loader, dev, FN)
    sm_cls, fn_cls = sm_tok.mean(0)[0], fn_tok.mean(0)[0]              # (256,) CLS-only
    sm_full, fn_full = sm_tok.mean(0), fn_tok.mean(0)                  # (T,256) whole branch

    # CLS-only patch = is the confound bottlenecked through the CLS token? (expected: no)
    # full-sequence patch = lesion the whole branch's between-subject variance (the real path test)
    conds = {
        "baseline": [],
        "patch_sMRI_cls":  [(SM, patch.make_write_hook(sm_cls, token_idx=0))],
        "patch_FNC_cls":   [(FN, patch.make_write_hook(fn_cls, token_idx=0))],
        "patch_sMRI_full": [(SM, patch.make_write_hook(sm_full, token_idx=None))],
        "patch_FNC_full":  [(FN, patch.make_write_hook(fn_full, token_idx=None))],
        "patch_both_full": [(SM, patch.make_write_hook(sm_full, token_idx=None)),
                            (FN, patch.make_write_hook(fn_full, token_idx=None))],
    }
    d = gdf.drop_duplicates("SubjectID").set_index("SubjectID")
    rows = []
    for name, eh in conds.items():
        r = patch.run_model(model, te_loader, dev, extra_hooks=eh)
        pop = np.array([d.loc[s, "population"] if s in d.index else "NA" for s in r["subject_id"]])
        keep = np.isin(pop, ["US", "China"])
        Xf = r["fused"].numpy()
        rows.append({"condition": name,
                     "scanner_pop_auc": cv_auc(Xf[keep], (pop[keep] == "China").astype(int)),
                     "disease_auc": cv_auc(Xf, r["y"])})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n{df.round(3).to_string(index=False)}\nwrote {args.out}")


if __name__ == "__main__":
    main()
