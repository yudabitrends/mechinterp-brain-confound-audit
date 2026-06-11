#!/usr/bin/env python
"""So-What payoff (reviewer P0-1) — turn the paper's untested 'subspace-targeted harmonization'
motivation into a tested deployment result.

The audit identifies a low-dimensional causal scanner subspace (DAS). The actionable claim is: you
can harmonize by editing ONLY that audit-identified subspace, instead of blindly correcting all
features (ComBat) or erasing a high-rank nullspace (full INLP). We test it on the held-out fused
decision rep:
  - DAS-subspace erasure: zero the top-k DAS rotated coordinates (k=1,2,4,8,16), measure scanner +
    disease (probe) + disease via the model HEAD logits (post-intervention disease AUC, answering the
    'argmax not AUC' objection).
  - vs full INLP (rounds 10..60), ComBat, and a rank-matched RANDOM-subspace erasure control.
The payoff: if low-rank DAS-subspace erasure reaches ComBat/INLP-level scanner removal with disease
preserved at a fraction of the edited rank, the audit gives a minimal, targeted correction. Cached
fused + frozen head; no model forward. Writes outputs/sae_ckpts/subspace_harmonize.csv.
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

MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"


def auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    c = LogisticRegression(max_iter=2000, solver="liblinear").fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, c.predict_proba(sc.transform(Xte))[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="outputs/activations/fused_HOLD_ALL.pt")
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/sae_ckpts/subspace_harmonize.csv")
    args = ap.parse_args()
    dev = "cpu"

    d = torch.load(args.fused, weights_only=True)
    h = d["fused"].float(); X = h.numpy()
    pop = np.asarray(d["population"]); site = np.asarray(d["site"]); split = np.asarray(d["split"])
    y = np.asarray(d["y_dx"]).astype(int); keep = np.isin(pop, ["US", "China"])
    tr = (split == "train") & keep; te = (split == "test") & keep
    yp = (pop == "China").astype(int)
    hW, hb = load_head(MHOLD)
    scanner = torch.tensor(yp)
    tri = torch.tensor(np.where(tr)[0]); tei = torch.tensor(np.where(te)[0])

    # post-intervention disease AUC via the model HEAD logit (answers 'argmax not AUC')
    def disease_head_auc(Xt):
        ht = torch.tensor(Xt, dtype=torch.float32)
        logit = (ht @ hW.T + hb)
        s = (logit[:, 1] - logit[:, 0]).numpy()
        return float(roc_auc_score(y[te], s[te]))

    rows = []
    # We operate on a full transformed array Xall and index by tr/te.
    def evaluate(Xall, method, rank):
        s = auc(Xall[tr], yp[tr], Xall[te], yp[te])
        dz = auc(Xall[tr], y[tr], Xall[te], y[te])
        dh = disease_head_auc(Xall)
        rows.append({"method": method, "edited_rank": rank, "scanner_auc": round(s, 3),
                     "disease_auc": round(dz, 3), "disease_head_auc": round(dh, 3)})
        print(f"{method:22s} rank={rank:<4} scanner={s:.3f} disease={dz:.3f} disease_head={dh:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    evaluate(X, "raw", 0)

    # --- DAS-subspace erasure: zero top-k rotated coords of the audit-identified causal subspace ---
    bt, st = make_pairs(scanner, tri, args.pairs, args.seed)
    for k in [int(x) for x in args.ks.split(",")]:
        das = D.train_das(h, scanner, hW, hb,
                          torch.tensor(LogisticRegression(max_iter=3000).fit(X[tr], yp[tr]).coef_.ravel(), dtype=torch.float32),
                          torch.tensor(0.0), bt, st, k=k, steps=args.steps, lr=5e-3, lam=1.0, seed=args.seed, device=dev)
        with torch.no_grad():
            z = das.rotate(h)            # (N,d) rotated
            z[:, :k] = 0.0               # erase the causal subspace
            Xk = das.unrotate(z).numpy()
        evaluate(Xk, "DAS_subspace_erase", k)

    # --- full INLP (high-rank nullspace) ---
    for r in [10, 20, 40, 60]:
        P = inlp(X[tr], yp[tr], r)
        evaluate(X @ P, "INLP", r)

    # --- ComBat (edits ALL features) ---
    try:
        Xtr_c, Xte_c = combat(X[tr], X[te], site[tr], site[te], y[tr], y[te])
        Xc = X.copy(); Xc[tr] = Xtr_c; Xc[te] = Xte_c
        evaluate(Xc, "ComBat", X.shape[1])
    except Exception as e:
        print(f"[ComBat] failed: {e!r}", flush=True)

    # --- rank-matched random-subspace erasure control (k=4) ---
    rng = np.random.default_rng(args.seed)
    Br, _ = np.linalg.qr(rng.standard_normal((X.shape[1], 4)))
    evaluate(X @ (np.eye(X.shape[1]) - Br @ Br.T), "random_subspace_erase", 4)

    print(f"\nwrote {args.out}")
    print("PAYOFF: if DAS_subspace_erase reaches ComBat/INLP-level scanner removal with disease preserved"
          " at edited_rank << ComBat's full-feature edit, the audit yields a minimal targeted correction.")


if __name__ == "__main__":
    main()
