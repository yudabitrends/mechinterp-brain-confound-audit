#!/usr/bin/env python
"""C4 (threat T6) — does the DAS/INLP scanner subspace actually EXPLAIN why ComBat works, or is
that an unproven rhetorical bridge?

We test the bridge mechanistically on the fused decision rep:
  (1) Fit the k=1 DAS scanner direction u (and the unit linear-probe direction w, |cos|~0.99 from
      C2). Project the fused rep onto u; measure the VARIANCE of that projection on raw vs ComBat-
      corrected reps. If ComBat works by collapsing exactly this axis, var should fall sharply.
  (2) Collinearity: the principal direction that ComBat (and INLP) REMOVE from the rep — i.e. the
      top eigenvector of the (raw - corrected) difference covariance — should be ~collinear with u
      if the linear correction and the causal subspace are the same object. Report |cos|.
  (3) Sanity: scanner-probe AUC raw vs ComBat vs INLP (held-out), to anchor that all three collapse
      scanner while preserving disease.
Cached fused rep; ComBat via neuroCombat (train-fit, OOS-applied), matching harmonize_compare.
Writes outputs/sae_ckpts/ctrl_combat_bridge.csv.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mib import das as D
from ctrl_das_null import load_head, make_pairs
from harmonize_compare import inlp, combat


def auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    c = LogisticRegression(max_iter=2000, solver="liblinear").fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, c.predict_proba(sc.transform(Xte))[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="outputs/activations/fused_HOLD_ALL.pt")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/sae_ckpts/ctrl_combat_bridge.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    d = torch.load(args.fused, weights_only=True)
    X = d["fused"].numpy(); h = d["fused"].float()
    pop = np.asarray(d["population"]); site = np.asarray(d["site"]); split = np.asarray(d["split"])
    y = np.asarray(d["y_dx"]); keep = np.isin(pop, ["US", "China"])
    tr = (split == "train") & keep; te = (split == "test") & keep
    yp = (pop == "China").astype(int)
    hW, hb = load_head("/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt")
    scanner = torch.tensor(yp)
    tri = torch.tensor(np.where(tr)[0]); tei = torch.tensor(np.where(te)[0])

    # (0) DAS k=1 direction u + probe direction w
    lr = LogisticRegression(max_iter=3000).fit(X[tr], yp[tr])
    scan_w = torch.tensor(lr.coef_.ravel(), dtype=torch.float32)
    scan_b = torch.tensor(float(lr.intercept_[0]), dtype=torch.float32)
    bt, st = make_pairs(scanner, tri, args.pairs, args.seed)
    das = D.train_das(h, scanner, hW, hb, scan_w, scan_b, bt, st, k=1,
                      steps=args.steps, lr=5e-3, lam=1.0, seed=args.seed, device=dev)
    u = das._W()[0].detach().cpu().numpy(); u = u / np.linalg.norm(u)
    w = lr.coef_.ravel(); w = w / np.linalg.norm(w)

    # (1) ComBat-correct fused (train-fit, OOS to test)
    try:
        Xtr_c, Xte_c = combat(X[tr], X[te], site[tr], site[te], y[tr], y[te])
        combat_ok = True
    except Exception as e:
        print(f"[ComBat] failed ({e!r})", flush=True)
        Xtr_c = Xte_c = None; combat_ok = False

    rows = []

    def add(metric, value, note=""):
        rows.append({"metric": metric, "value": value, "note": note})
        print(f"{metric:34s} {value if isinstance(value,str) else round(value,4)}  {note}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    add("cos(DAS_u, probe_w)", float(abs(u @ w)), "C2 cross-check")

    if combat_ok:
        # variance of projection onto u, raw vs ComBat (held-out test)
        var_raw = float(np.var(X[te] @ u))
        var_cmb = float(np.var(Xte_c @ u))
        add("var_proj_u_raw", var_raw, "test")
        add("var_proj_u_combat", var_cmb, "test")
        add("var_collapse_ratio", var_cmb / (var_raw + 1e-12), "ComBat/raw along DAS axis")
        # collinearity of ComBat-removed direction with u
        Dlt = (X[te] - Xte_c)                                  # what ComBat removed (test)
        # top eigenvector of removed-signal covariance
        Dc = Dlt - Dlt.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Dc, full_matrices=False)
        v_removed = Vt[0]
        add("cos(combat_removed_dir, DAS_u)", float(abs(v_removed @ u)), "top PC of (raw-ComBat)")
        # scanner/disease AUC anchors
        add("scanner_auc_raw", auc(X[tr], yp[tr], X[te], yp[te]))
        add("scanner_auc_combat", auc(Xtr_c, yp[tr], Xte_c, yp[te]))
        add("disease_auc_raw", auc(X[tr], y[tr], X[te], y[te]))
        add("disease_auc_combat", auc(Xtr_c, y[tr], Xte_c, y[te]))

    # INLP reference (always available)
    P = inlp(X[tr], yp[tr], args.rounds)
    add("scanner_auc_inlp", auc(X[tr] @ P, yp[tr], X[te] @ P, yp[te]))
    add("disease_auc_inlp", auc(X[tr] @ P, y[tr], X[te] @ P, y[te]))
    # collinearity of INLP-removed subspace with u: 1 - ||P u|| (fraction of u erased)
    frac_u_erased = float(1.0 - np.linalg.norm(P @ u))
    add("frac_DAS_u_erased_by_INLP", frac_u_erased, "1-||P u|| (1=fully in INLP nullspace)")

    print(f"\nwrote {args.out}")
    print("\nINTERPRETATION: a large var-collapse-ratio drop + high cos(ComBat-removed, DAS_u) +"
          "\nhigh frac_DAS_u_erased_by_INLP would mean ComBat/INLP literally act on the causal DAS"
          "\naxis -> the bridge is mechanistic. Weak collapse -> retreat to 'linear correction at the"
          "\ndecision rep succeeds; ComBat is one instance' (no causal-bridge overclaim).")


if __name__ == "__main__":
    main()
