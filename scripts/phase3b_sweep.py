#!/usr/bin/env python
"""Lean INLP rounds sweep for the harmonization tradeoff curve (scanner vs disease AUC
as a function of erased rank). Uses liblinear (stable on post-projection collinear data)
and thread caps. Writes outputs/sae_ckpts/phase3b_rounds_sweep.csv.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def auc(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=200).fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def inlp(X, y, rounds):
    d = X.shape[1]; P = np.eye(d); Xc = X.copy()
    for _ in range(rounds):
        w = LogisticRegression(solver="liblinear", C=1.0, max_iter=200).fit(Xc, y).coef_.ravel()
        n = np.linalg.norm(w)
        if n < 1e-9:
            break
        w /= n; Pw = np.eye(d) - np.outer(w, w); Xc = Xc @ Pw; P = P @ Pw
    return P


def main():
    d = torch.load("outputs/activations/fused_ALL.pt", weights_only=True)
    X = d["fused"].numpy(); y = np.asarray(d["y_dx"]); pop = np.asarray(d["population"])
    keep = np.isin(pop, ["US", "China"]); X, y, pop = X[keep], y[keep], pop[keep]
    yp = (pop == "China").astype(int)
    strata = y.astype(str) + "_" + yp.astype(str)
    tr, te = train_test_split(np.arange(len(X)), test_size=0.3, random_state=0, stratify=strata)
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

    rows = [{"rounds": 0, "scanner_pop_auc": auc(Xtr, yp[tr], Xte, yp[te]),
             "disease_auc": auc(Xtr, y[tr], Xte, y[te])}]
    for R in [10, 20, 40, 60, 100]:
        P = inlp(Xtr, yp[tr], R)
        rows.append({"rounds": R, "scanner_pop_auc": auc(Xtr @ P, yp[tr], Xte @ P, yp[te]),
                     "disease_auc": auc(Xtr @ P, y[tr], Xte @ P, y[te])})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("outputs/sae_ckpts/phase3b_rounds_sweep.csv", index=False)
    print("wrote phase3b_rounds_sweep.csv")


if __name__ == "__main__":
    main()
