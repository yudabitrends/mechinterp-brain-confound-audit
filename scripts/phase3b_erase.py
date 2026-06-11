#!/usr/bin/env python
"""Phase 3b: mechanistic harmonization by scanner-direction erasure (held-out).

On the fused decision representation: iteratively project out the linearly-decodable scanner
(population) subspace (INLP), fit on a TRAIN split, evaluate on a held-out TEST split. If the
scanner readout is near-orthogonal to disease (Phase-3 |cos|=0.06), erasure should drop scanner
decodability toward chance while preserving disease -> a causal harmonization. Control: erase a
random subspace of equal rank (scanner should NOT drop).

Writes outputs/sae_ckpts/phase3b_erase.csv.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def fit_eval_auc(Xtr, ytr, Xte, yte, multiclass=False):
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    if multiclass:
        classes = np.unique(ytr)
        proba = clf.predict_proba(Xte)
        Y = label_binarize(yte, classes=classes)
        return float(roc_auc_score(Y, proba, average="macro", multi_class="ovr"))
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def inlp_projection(X, y, rounds):
    """Return projection P (d,d) removing the linearly-decodable directions for label y."""
    d = X.shape[1]
    P = np.eye(d)
    Xc = X.copy()
    for _ in range(rounds):
        w = LogisticRegression(max_iter=1000).fit(Xc, y).coef_.ravel()
        n = np.linalg.norm(w)
        if n < 1e-9:
            break
        w /= n
        Pw = np.eye(d) - np.outer(w, w)
        Xc = Xc @ Pw
        P = P @ Pw
    return P


def random_projection(d, rank, seed):
    g = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(g.standard_normal((d, rank)))
    return np.eye(d) - Q @ Q.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="outputs/activations/fused_ALL.pt")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/sae_ckpts")
    args = ap.parse_args()

    d = torch.load(args.fused, weights_only=True)   # cache holds only tensors + lists/str
    X = d["fused"].numpy()
    y_dx = np.asarray(d["y_dx"])
    pop = np.asarray(d["population"]); site = np.asarray(d["site"])
    keep = np.isin(pop, ["US", "China"])
    X, y_dx, pop, site = X[keep], y_dx[keep], pop[keep], site[keep]
    y_pop = (pop == "China").astype(int)

    # held-out split stratified by disease x population
    strata = y_dx.astype(str) + "_" + y_pop.astype(str)
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=strata)

    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

    # site classes with enough train support (for the multiclass scanner readout)
    vc = pd.Series(site[tr]).value_counts(); keep_site = vc[vc >= 20].index
    sm_tr = np.isin(site[tr], keep_site); sm_te = np.isin(site[te], keep_site)

    def panel(Xtr_, Xte_, tag):
        return {
            "variant": tag,
            "scanner_pop_auc": fit_eval_auc(Xtr_, y_pop[tr], Xte_, y_pop[te]),
            "scanner_site_auc": fit_eval_auc(Xtr_[sm_tr], site[tr][sm_tr], Xte_[sm_te], site[te][sm_te], multiclass=True),
            "disease_auc": fit_eval_auc(Xtr_, y_dx[tr], Xte_, y_dx[te]),
        }

    rows = [panel(Xtr, Xte, "pre_erase")]

    P = inlp_projection(Xtr, y_pop[tr], args.rounds)
    rows.append(panel(Xtr @ P, Xte @ P, f"erase_scanner_INLP_r{args.rounds}"))

    Pr = random_projection(X.shape[1], args.rounds, args.seed)
    rows.append(panel(Xtr @ Pr, Xte @ Pr, f"erase_random_r{args.rounds}"))

    df = pd.DataFrame(rows)
    out = f"{args.out}/phase3b_erase.csv"
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print(f"\nN_train={len(tr)} N_test={len(te)} | wrote {out}")
    pre, era = df.iloc[0], df.iloc[1]
    print(f"\nHARMONIZATION: scanner(pop) {pre.scanner_pop_auc:.3f}->{era.scanner_pop_auc:.3f}, "
          f"site {pre.scanner_site_auc:.3f}->{era.scanner_site_auc:.3f}, "
          f"disease {pre.disease_auc:.3f}->{era.disease_auc:.3f} (cost {pre.disease_auc-era.disease_auc:+.3f})")


if __name__ == "__main__":
    main()
