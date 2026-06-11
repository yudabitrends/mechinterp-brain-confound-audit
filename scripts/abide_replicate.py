#!/usr/bin/env python
"""ABIDE replication: does the scanner-vs-disease mechanism generalize to autism?

Scanner axis = SITE (no US/China split). On held-out test of M_abide: (1) per-branch
dissociation — probe sMRI-CLS vs FNC-CLS for disease (dx) and site; (2) harmonization on
the fused rep (fit-train/eval-test): raw / ComBat / site-regression / random, reporting
site AUC + disease AUC. Writes outputs/sae_ckpts/abide_{dissociation,harmonize}.csv.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
import argparse, importlib.util, json, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, label_binarize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mib.abide_data import build_abide_manifest, ABIDEDataset
from mib.extract import load_model, ActivationExtractor, ExtractConfig
from mib import patch
_spec = importlib.util.spec_from_file_location("hc", os.path.join(os.path.dirname(__file__), "harmonize_compare.py"))
hc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(hc)

CKPT = "/home/users/ybi3/MultiViT2/outputs/abide_split/best.pt"
SI = "/home/users/ybi3/MultiViT2/outputs/abide_split/split_info.json"


def cv_auc(X, y, k=5, seed=0):
    y = np.asarray(y)
    if y.dtype.kind not in "iu":
        y = pd.factorize(y)[0]
    cl = np.unique(y)
    if len(cl) < 2:
        return np.nan
    kk = int(min(k, np.bincount(y)[np.bincount(y) > 0].min()))
    if kk < 2:
        return np.nan
    multi = len(cl) > 2
    Xs = StandardScaler().fit_transform(X)
    proba = cross_val_predict(LogisticRegression(solver="lbfgs" if multi else "liblinear", max_iter=2000),
                              Xs, y, cv=StratifiedKFold(kk, shuffle=True, random_state=seed),
                              method="predict_proba")
    if multi:
        return float(roc_auc_score(label_binarize(y, classes=cl), proba, average="macro", multi_class="ovr"))
    return float(roc_auc_score(y, proba[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-site-n", type=int, default=15)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    si = json.load(open(SI))
    df = build_abide_manifest()
    tr_df = df[df.SubjectID.isin(set(si["train_ids"]))].reset_index(drop=True)
    te_df = df[df.SubjectID.isin(set(si["test_ids"]))].reset_index(drop=True)
    model = load_model(CKPT, device=dev)

    def loader(d):
        return DataLoader(ABIDEDataset(d, cache_dir="outputs/cache_abide"), batch_size=16, shuffle=False, num_workers=4)

    # --- fused (train + test) for harmonization ---
    def fused(d):
        r = patch.run_model(model, loader(d), dev)
        meta = d.drop_duplicates("SubjectID").set_index("SubjectID")
        site = np.array([meta.loc[s, "site"] for s in r["subject_id"]])
        return r["fused"].numpy(), r["y"], site
    Xtr, ytr, str_tr = fused(tr_df)
    Xte, yte, str_te = fused(te_df)

    # --- per-branch CLS (test) for dissociation ---
    cfg = ExtractConfig(ckpt_path=CKPT, position="cls", device=dev,
                        hook_points=["sMRI_encoder.norm", "sFNC_encoder.norm"])
    with ActivationExtractor(model, cfg) as ext:
        acts, labels = ext.run(loader(te_df))
    meta = te_df.drop_duplicates("SubjectID").set_index("SubjectID")
    site_te = np.array([meta.loc[s, "site"] for s in labels["subject_id"]])
    y_te = labels["label"]

    diss = []
    for hook, name in [("sMRI_encoder.norm", "sMRI"), ("sFNC_encoder.norm", "FNC")]:
        Xb = acts[hook].numpy()
        diss.append({"branch": name, "disease_auc": round(cv_auc(Xb, y_te), 3),
                     "site_auc": round(cv_auc(Xb, site_te), 3)})
        print(diss[-1], flush=True)
    pd.DataFrame(diss).to_csv("outputs/sae_ckpts/abide_dissociation.csv", index=False)

    # --- harmonization on fused: scanner axis = site ---
    okv = [s for s in np.unique(str_tr)
           if (str_tr == s).sum() >= args.min_site_n and (str_te == s).sum() >= args.min_site_n]
    smt, sme = np.isin(str_tr, okv), np.isin(str_te, okv)

    def ev(Xa, Xb):
        return {"site_auc": hc.auc_fit_eval(Xa[smt], str_tr[smt], Xb[sme], str_te[sme], multiclass=True),
                "disease_auc": hc.auc_fit_eval(Xa, ytr, Xb, yte)}
    methods = {"raw": (Xtr, Xte)}
    import sklearn.linear_model as sklm
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(str_tr.reshape(-1, 1))
    lr = sklm.LinearRegression().fit(enc.transform(str_tr.reshape(-1, 1)), Xtr)
    methods["site_regression"] = (Xtr - lr.predict(enc.transform(str_tr.reshape(-1, 1))),
                                  Xte - lr.predict(enc.transform(str_te.reshape(-1, 1))))
    Pr = hc.random_proj(Xtr.shape[1], 20, 0)
    methods["random_erasure"] = (Xtr @ Pr, Xte @ Pr)
    try:
        methods["ComBat"] = hc.combat(Xtr, Xte, str_tr, str_te, ytr, yte)
    except Exception as e:
        print("[ComBat] failed", e); methods["ComBat"] = None

    rows = []
    for name, pair in methods.items():
        try:
            r = ev(*pair) if pair else {"site_auc": np.nan, "disease_auc": np.nan}
        except Exception as e:                       # e.g. ComBat OOS "in development" -> bad data
            print(f"[{name}] eval failed: {e!r}"); r = {"site_auc": np.nan, "disease_auc": np.nan}
        rows.append({"method": name, **r}); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("outputs/sae_ckpts/abide_harmonize.csv", index=False)
    print("wrote abide_dissociation.csv + abide_harmonize.csv")


if __name__ == "__main__":
    main()
