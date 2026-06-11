#!/usr/bin/env python
"""Reviewer #14: does the optimal closed-form linear eraser (LEACE, Belrose et al. 2023) drive scanner to
chance, or does it also floor near 0.55 like iterative INLP? This settles whether the 0.55 INLP floor is a
genuine linear-irreducibility (redundancy lower bound) or merely INLP sub-optimality.

LEACE eraser (binary concept): r' = r - W^+ (u u^T) W (r - mu), with whitening W = Sigma_xx^{-1/2},
W^+ = Sigma_xx^{1/2}, and u = normalize(W Sigma_xz) the whitened class-mean-difference direction. Fit on TRAIN,
applied to held-out TEST; a fresh scanner / disease probe is then fit on erased-train and scored on erased-test.
Cached fused_HOLD_ALL.pt; CPU. Writes outputs/sae_ckpts/leace.csv.
"""
import os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def auc(Xtr, ytr, Xte, yte):
    return roc_auc_score(yte, LogisticRegression(max_iter=4000, C=1.0)
                         .fit(Xtr, ytr).predict_proba(Xte)[:, 1])


def main():
    d = torch.load("outputs/activations/fused_HOLD_ALL.pt", weights_only=True)
    X = d["fused"].numpy().astype(np.float64); y = np.asarray(d["y_dx"])
    pop = np.asarray(d["population"]); split = np.asarray(d["split"])
    keep = np.isin(pop, ["US", "China"]); z = (pop == "China").astype(float)
    trk = np.where((split == "train") & keep)[0]; tek = np.where((split == "test") & keep)[0]

    Xtr = X[trk]; ztr = z[trk]; mu = Xtr.mean(0); Xc = Xtr - mu
    Sxx = np.cov(Xc, rowvar=False)
    ev, V = np.linalg.eigh(Sxx); ev = np.clip(ev, 1e-6 * ev.max(), None)
    W = V @ np.diag(ev ** -0.5) @ V.T          # Sigma_xx^{-1/2}  (whitening)
    Winv = V @ np.diag(ev ** 0.5) @ V.T        # Sigma_xx^{ 1/2}  (= W^+)
    Sxz = (Xc * (ztr - ztr.mean())[:, None]).mean(0)
    u = W @ Sxz; u = u / (np.linalg.norm(u) + 1e-12)   # whitened mean-difference dir (rank-1 for binary)
    WinvU = Winv @ u

    def erase(A):
        Ac = A - mu
        coef = (Ac @ W) @ u                    # u^T W (a-mu)  (W symmetric)
        return A - np.outer(coef, WinvU)

    Xe = erase(X)
    rows = [
        {"rep": "raw",   "scanner_pop_auc": round(auc(X[trk],  z[trk],  X[tek],  z[tek]), 3),
                          "disease_auc":     round(auc(X[trk],  y[trk],  X[tek],  y[tek]), 3)},
        {"rep": "LEACE", "scanner_pop_auc": round(auc(Xe[trk], z[trk],  Xe[tek], z[tek]), 3),
                          "disease_auc":     round(auc(Xe[trk], y[trk],  Xe[tek], y[tek]), 3)},
    ]
    for r in rows: print(r, flush=True)
    pd.DataFrame(rows).to_csv("outputs/sae_ckpts/leace.csv", index=False)
    sc = rows[1]["scanner_pop_auc"]
    print(f"\nLEACE scanner AUC = {sc} (raw {rows[0]['scanner_pop_auc']}; INLP@60 floor ~0.55-0.57).")
    print("INTERP: if LEACE ~0.50, the 0.55 INLP floor was sub-optimality and scanner IS fully linearly "
          "removable (redundancy shows up as FEATURE-ablation failure, not linear-subspace incompleteness); "
          "if LEACE also ~0.55, the residual is a genuine linear-irreducibility. Disease must stay ~baseline.")


if __name__ == "__main__":
    main()
