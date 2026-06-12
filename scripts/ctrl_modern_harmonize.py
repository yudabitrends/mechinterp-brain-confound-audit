#!/usr/bin/env python
"""Reviewer M6: vanilla ComBat cannot apply to an unseen site, but covariate-preserving ComBat (neuroHarmonize)
is designed to apply a learned model to new data. We test whether it removes scanner from a HELD-OUT (unseen) US
cohort when fit only on the seen cohorts (with age/sex/diagnosis covariates), in the same leave-one-US-cohort-out
protocol as the decision-level correction. Honest test of whether a MODERN harmonizer escapes the unseen-site
limitation. Cached fused rep; demographics from the szdataset HDF5. Writes outputs/sae_ckpts/modern_harmonize.csv.
"""
import os, sys
import numpy as np, pandas as pd, torch
sys.modules.setdefault("numpy._core", np.core)               # HDF5 pickled under numpy>=2; alias for numpy<2
for _s in (".multiarray", ".numeric", ".umath", "._multiarray_umath"):
    try: sys.modules.setdefault("numpy._core" + _s, __import__("numpy.core" + _s, fromlist=["x"]))
    except Exception: pass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from ctrl_offsite_closedloop import site_to_cohort, US_COHORTS
from neuroHarmonize import harmonizationLearn, harmonizationApply

H5 = "/home/users/ybi3/data/szdataset_modified.h5"


def demo_maps():
    age, sex = {}, {}
    for grp in ("train", "test"):
        df = pd.read_hdf(H5, grp)
        for s, a, g in zip(df["SubjectID"].astype(str), df["Age"], df["Gender"]):
            try: age[s] = float(a); sex[s] = int(float(g))
            except (ValueError, TypeError): pass
    return age, sex


def scanner_auc(X, yp, tr, te):
    lr = LogisticRegression(max_iter=2000).fit(X[tr], yp[tr])
    a = roc_auc_score(yp[te], lr.predict_proba(X[te])[:, 1]); return max(a, 1 - a)


def main():
    d = torch.load("outputs/activations/fused_HOLD_ALL.pt", weights_only=True)
    X = d["fused"].numpy(); y = np.asarray(d["y_dx"]); pop = np.asarray(d["population"])
    site = np.asarray(d["site"]); split = np.asarray(d["split"]); sid = np.asarray(d["subject_id"])
    cohort = site_to_cohort(site); yp = (pop == "China").astype(int)
    agem, sexm = demo_maps()
    age = np.array([agem.get(str(s), np.nan) for s in sid]); sex = np.array([sexm.get(str(s), np.nan) for s in sid])
    have = ~np.isnan(age) & ~np.isnan(sex)
    rows = []
    for H in US_COHORTS:
        seen = (cohort != H) & have
        tr = np.where(seen & (split == "train"))[0]
        te = np.where(have & ((cohort == H) | ((cohort != H) & (pop == "China") & (split == "test"))))[0]
        before = scanner_auc(X, yp, tr, te)
        cov_tr = pd.DataFrame({"SITE": site[tr], "age": age[tr], "sex": sex[tr], "dx": y[tr]})
        try:
            model, _ = harmonizationLearn(X[tr], cov_tr)
            cov_te = pd.DataFrame({"SITE": site[te], "age": age[te], "sex": sex[te], "dx": y[te]})
            Xte_h = harmonizationApply(X[te], cov_te, model)
            # refit scanner probe on harmonized train (apply model to train too for a fair probe)
            Xtr_h = harmonizationApply(X[tr], cov_tr, model)
            lr = LogisticRegression(max_iter=2000).fit(Xtr_h, yp[tr])
            after = max(roc_auc_score(yp[te], lr.predict_proba(Xte_h)[:, 1]), 1 - roc_auc_score(yp[te], lr.predict_proba(Xte_h)[:, 1]))
            note = "applied to unseen site"
        except Exception as e:
            after = np.nan; note = f"FAILED on unseen batch: {repr(e)[:80]}"
        rows.append({"held_out": H, "scanner_auc_before": round(before, 3),
                     "scanner_auc_after_neuroHarmonize": (round(after, 3) if after == after else "NA"), "note": note})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("outputs/sae_ckpts/modern_harmonize.csv", index=False)
    print("\nM6: does covariate-preserving ComBat (neuroHarmonize), fit on seen cohorts, remove scanner from the "
          "UNSEEN cohort? Compare scanner_auc_after to the decision-level INLP off-site result (0.92->0.64).")


if __name__ == "__main__":
    main()
