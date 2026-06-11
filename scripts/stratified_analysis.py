#!/usr/bin/env python
"""Stage C: stratified / nuisance-controlled checks on the held-out test fused rep.

Shows the scanner-vs-disease dissociation and erasure-harmonization survive when the
disease x population confound is removed: within-US, within-China, on a dx x population
balanced subset, and controlling the nuisance (scanner decodable within each diagnosis;
disease decodable within each population). Writes outputs/sae_ckpts/stratified.csv.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
import argparse, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, label_binarize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def cv_auc(X, y, multiclass=False, k=5, seed=0):
    """CV AUC; binary vs multiclass auto-detected from the actual #classes (a 2-class
    'multiclass' label must use the binary path, else label_binarize gives 1 col)."""
    y = np.asarray(y)
    if y.dtype.kind not in "iu":
        y = pd.factorize(y)[0]
    classes = np.unique(y)
    if len(classes) < 2:
        return np.nan
    kk = int(min(k, np.bincount(y)[np.bincount(y) > 0].min()))
    if kk < 2:
        return np.nan
    is_multi = len(classes) > 2
    Xs = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(kk, shuffle=True, random_state=seed)
    clf = LogisticRegression(solver="lbfgs" if is_multi else "liblinear", max_iter=2000)
    proba = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")
    if is_multi:
        Y = label_binarize(y, classes=classes)
        return float(roc_auc_score(Y, proba, average="macro", multi_class="ovr"))
    return float(roc_auc_score(y, proba[:, 1]))


def inlp(X, y, rounds=20):
    d = X.shape[1]; P = np.eye(d); Xc = X.copy()
    for _ in range(rounds):
        w = LogisticRegression(solver="liblinear", max_iter=200).fit(Xc, y).coef_.ravel()
        n = np.linalg.norm(w)
        if n < 1e-9:
            break
        w /= n; Pw = np.eye(d) - np.outer(w, w); Xc = Xc @ Pw; P = P @ Pw
    return P


def site_codes(site, min_n):
    vc = pd.Series(site).value_counts()
    keep = vc[vc >= min_n].index
    m = np.isin(site, keep)
    return m, pd.factorize(site[m])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="outputs/activations/fused_HOLD_ALL.pt")
    ap.add_argument("--min-site-n", type=int, default=15)
    ap.add_argument("--out", default="outputs/sae_ckpts/stratified.csv")
    args = ap.parse_args()

    d = torch.load(args.fused, weights_only=True)
    X = d["fused"].numpy(); y = np.asarray(d["y_dx"])
    pop = np.asarray(d["population"]); site = np.asarray(d["site"]); split = np.asarray(d["split"])
    te = split == "test"
    X, y, pop, site = X[te], y[te], pop[te], site[te]
    yp = (pop == "China").astype(int)
    rows = []

    # within-population strata: scanner axis = SITE (pop is constant), disease = dx
    for popname in ["US", "China"]:
        m = pop == popname
        sm, scodes = site_codes(site[m], args.min_site_n)
        Xm = X[m]
        scan = cv_auc(Xm[sm], scodes, multiclass=True)
        dis = cv_auc(Xm, y[m])
        # erase site subspace, recheck
        if sm.sum() > 10 and not np.isnan(scan):
            P = inlp(Xm[sm], (scodes == scodes.max()).astype(int))  # erase a dominant site dir (proxy)
            scan_e = cv_auc((Xm[sm]) @ P, scodes, multiclass=True)
            dis_e = cv_auc(Xm @ P, y[m])
        else:
            scan_e = dis_e = np.nan
        rows.append({"stratum": f"within_{popname}", "scanner_axis": "site", "n": int(m.sum()),
                     "scanner_auc": scan, "disease_auc": dis,
                     "scanner_auc_post_erase": scan_e, "disease_auc_post_erase": dis_e})

    # balanced dx x population subset: equal n per (dx,pop) cell
    rng = np.random.default_rng(0)
    cells = [(dx, pp) for dx in [0, 1] for pp in [0, 1]]
    ncell = min((( (y == dx) & (yp == pp) ).sum() for dx, pp in cells))
    idx = np.concatenate([rng.choice(np.where((y == dx) & (yp == pp))[0], ncell, replace=False)
                          for dx, pp in cells])
    rows.append({"stratum": "balanced_dx_x_pop", "scanner_axis": "population", "n": len(idx),
                 "scanner_auc": cv_auc(X[idx], yp[idx]), "disease_auc": cv_auc(X[idx], y[idx]),
                 "scanner_auc_post_erase": np.nan, "disease_auc_post_erase": np.nan})

    # nuisance-controlled: scanner(pop) within each diagnosis; disease within each population
    pop_within_dx = np.nanmean([cv_auc(X[y == dx], yp[y == dx]) for dx in [0, 1]])
    dis_within_pop = np.nanmean([cv_auc(X[yp == pp], y[yp == pp]) for pp in [0, 1]])
    rows.append({"stratum": "control_dx (pop|within each dx)", "scanner_axis": "population", "n": len(X),
                 "scanner_auc": pop_within_dx, "disease_auc": np.nan,
                 "scanner_auc_post_erase": np.nan, "disease_auc_post_erase": np.nan})
    rows.append({"stratum": "control_pop (dx|within each pop)", "scanner_axis": "-", "n": len(X),
                 "scanner_auc": np.nan, "disease_auc": dis_within_pop,
                 "scanner_auc_post_erase": np.nan, "disease_auc_post_erase": np.nan})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
