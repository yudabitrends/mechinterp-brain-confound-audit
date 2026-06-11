#!/usr/bin/env python
"""Stage B: harmonization head-to-head on the held-out (model-unseen) test set.

All methods FIT on M_hold's TRAIN fused rep, EVALUATE on its held-out TEST fused rep, so
disease AUC is honest (the model never saw test subjects -> no 0.99 ceiling). Reports
scanner(pop) / site / disease AUC for: raw, ComBat, site-regression, INLP scanner-erasure,
random erasure. Writes outputs/sae_ckpts/harmonize_compare.csv.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
import argparse, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def auc_fit_eval(Xtr, ytr, Xte, yte, multiclass=False):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = LogisticRegression(solver="liblinear" if not multiclass else "lbfgs",
                             max_iter=2000, C=1.0).fit(Xtr, ytr)
    if multiclass:
        classes = np.unique(ytr)
        Y = label_binarize(yte, classes=classes)
        return float(roc_auc_score(Y, clf.predict_proba(Xte), average="macro", multi_class="ovr"))
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def inlp(X, y, rounds=20):
    d = X.shape[1]; P = np.eye(d); Xc = X.copy()
    for _ in range(rounds):
        w = LogisticRegression(solver="liblinear", max_iter=200).fit(Xc, y).coef_.ravel()
        n = np.linalg.norm(w)
        if n < 1e-9:
            break
        w /= n; Pw = np.eye(d) - np.outer(w, w); Xc = Xc @ Pw; P = P @ Pw
    return P


def random_proj(d, rank, seed):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, rank)))
    return np.eye(d) - Q @ Q.T


def combat(Xtr, Xte, site_tr, site_te, dx_tr, dx_te):
    """ComBat fit on train, applied out-of-sample to test via neuroCombatFromTraining."""
    from neuroCombat import neuroCombat, neuroCombatFromTraining
    cov_tr = pd.DataFrame({"site": site_tr, "dx": dx_tr})
    res = neuroCombat(dat=Xtr.T, covars=cov_tr, batch_col="site",
                      categorical_cols=["dx"])  # dat = features x samples
    Xtr_h = res["data"].T
    out = neuroCombatFromTraining(dat=Xte.T, batch=np.asarray(site_te), estimates=res["estimates"])
    return Xtr_h, out["data"].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="outputs/activations/fused_HOLD_ALL.pt")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--min-site-n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/sae_ckpts/harmonize_compare.csv")
    args = ap.parse_args()

    d = torch.load(args.fused, weights_only=True)
    X = d["fused"].numpy(); y = np.asarray(d["y_dx"])
    pop = np.asarray(d["population"]); site = np.asarray(d["site"]); split = np.asarray(d["split"])
    tr, te = split == "train", split == "test"
    yp = (pop == "China").astype(int)
    print(f"train={tr.sum()} test={te.sum()} | test dx {np.bincount(y[te])}", flush=True)

    # sites present with enough support in BOTH splits (for multiclass site AUC)
    okv = [s for s in np.unique(site)
           if (site[tr] == s).sum() >= args.min_site_n and (site[te] == s).sum() >= args.min_site_n]
    sm_tr, sm_te = np.isin(site[tr], okv), np.isin(site[te], okv)

    def evaluate(Xtr_t, Xte_t):
        return {
            "scanner_pop_auc": auc_fit_eval(Xtr_t, yp[tr], Xte_t, yp[te]),
            "site_auc": auc_fit_eval(Xtr_t[sm_tr], site[tr][sm_tr], Xte_t[sm_te], site[te][sm_te], multiclass=True),
            "disease_auc": auc_fit_eval(Xtr_t, y[tr], Xte_t, y[te]),
        }

    Xtr, Xte = X[tr], X[te]
    methods = {}
    methods["raw"] = (Xtr, Xte)
    # site-regression residualization (fit encoder+linreg on train)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(site[tr].reshape(-1, 1))
    lr = LinearRegression().fit(enc.transform(site[tr].reshape(-1, 1)), Xtr)
    methods["site_regression"] = (Xtr - lr.predict(enc.transform(site[tr].reshape(-1, 1))),
                                  Xte - lr.predict(enc.transform(site[te].reshape(-1, 1))))
    P = inlp(Xtr, yp[tr], args.rounds)
    methods["INLP_scanner_erasure"] = (Xtr @ P, Xte @ P)
    Pr = random_proj(X.shape[1], args.rounds, args.seed)
    methods["random_erasure"] = (Xtr @ Pr, Xte @ Pr)
    try:
        methods["ComBat"] = combat(Xtr, Xte, site[tr], site[te], y[tr], y[te])
    except Exception as e:
        print(f"[ComBat] failed ({e!r}) -> recording NaN", flush=True)
        methods["ComBat"] = None

    rows = []
    for name, pair in methods.items():
        if pair is None:
            rows.append({"method": name, "scanner_pop_auc": np.nan, "site_auc": np.nan, "disease_auc": np.nan})
        else:
            rows.append({"method": name, **evaluate(*pair)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)[["method", "scanner_pop_auc", "site_auc", "disease_auc"]]
    df.to_csv(args.out, index=False)
    print(f"\n{df.round(3).to_string(index=False)}\nwrote {args.out}")


if __name__ == "__main__":
    main()
