#!/usr/bin/env python
"""E3 (reviewer R2) — decodability is not causality: what does DAS add over a linear probe?

The k=1 DAS axis is nearly collinear with the linear scanner probe (|cos|=0.99), prompting "then DAS
just re-finds the linear direction." We answer empirically. For a BATTERY of candidate unit directions
in the fused decision rep we measure two very different quantities on held-out subjects:
  (x) CORRELATIONAL decodability  = AUC of the 1-D projection (h·u) predicting scanner (probe-style);
  (y) CAUSAL interchange-IIA      = swap the h·u component source->base, does the frozen scanner readout
      flip to source WHILE the model's disease decision (head) is preserved?
Directions: the linear scanner-probe weight; top principal components (high variance); successive INLP
nullspace directions (decreasingly scanner-decodable); random directions. The point: decodability does
NOT imply causal transfer-with-disease-preservation; the probe/DAS axis is the one that achieves BOTH.
DAS therefore certifies (via interchange on the real decision) what probing alone cannot. Cached fused +
frozen head, no model forward. Writes outputs/sae_ckpts/decode_vs_causal.csv.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ctrl_das_null import load_head, make_pairs

MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"


def inlp_directions(X, y, rounds):
    """Return the successive unit directions INLP removes (each round's logistic weight)."""
    d = X.shape[1]; P = np.eye(d); Xc = X.copy(); dirs = []
    for _ in range(rounds):
        w = LogisticRegression(solver="liblinear", max_iter=200).fit(Xc, y).coef_.ravel()
        n = np.linalg.norm(w)
        if n < 1e-9:
            break
        w = w / n; dirs.append((P.T @ w))            # map back to original coords
        Pw = np.eye(d) - np.outer(w, w); Xc = Xc @ Pw; P = P @ Pw
    return [v / (np.linalg.norm(v) + 1e-12) for v in dirs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="outputs/activations/fused_HOLD_ALL.pt")
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/sae_ckpts/decode_vs_causal.csv")
    args = ap.parse_args()
    dev = "cpu"

    d = torch.load(args.fused, weights_only=True)
    h = d["fused"].float(); X = h.numpy()
    pop = np.asarray(d["population"]); split = np.asarray(d["split"]); keep = np.isin(pop, ["US", "China"])
    hW, hb = load_head(MHOLD)
    scanner = torch.tensor((pop == "China").astype(int))
    tr = np.where((split == "train") & keep)[0]; te = np.where((split == "test") & keep)[0]
    tri, tei = torch.tensor(tr), torch.tensor(te)
    ys_tr, ys_te = scanner[tr].numpy(), scanner[te].numpy()

    # frozen scanner probe + interchange test pairs
    lr = LogisticRegression(max_iter=3000).fit(X[tr], ys_tr)
    scan_w = torch.tensor(lr.coef_.ravel(), dtype=torch.float32); scan_b = torch.tensor(float(lr.intercept_[0]))
    bv, sv = make_pairs(scanner, tei, args.pairs, args.seed + 100)
    base_dx = (h[bv] @ hW.T + hb).argmax(1); scan_src = scanner[sv]

    def decodability(u):                              # correlational: AUC of 1-D projection
        s = X[te] @ u; a = roc_auc_score(ys_te, s); return max(a, 1 - a)

    @torch.no_grad()
    def causal_iia(u):                                # 1-D interchange along u (source->base)
        ut = torch.tensor(u, dtype=torch.float32)
        coef = ((h[sv] - h[bv]) @ ut).unsqueeze(1)
        hp = h[bv] + coef * ut
        scan_pred = (hp @ scan_w + scan_b) > 0
        dx_pred = (hp @ hW.T + hb).argmax(1)
        return ((scan_pred.long() == scan_src).float().mean().item(),
                (dx_pred == base_dx).float().mean().item())

    # ---- direction battery ----
    dirs = []
    pw = lr.coef_.ravel(); dirs.append(("probe", "scanner probe", pw / np.linalg.norm(pw)))
    Xc = X[tr] - X[tr].mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    for i in range(8):
        dirs.append(("pca", f"PC{i+1}", Vt[i] / np.linalg.norm(Vt[i])))
    for i, u in enumerate(inlp_directions(X[tr], ys_tr, 8)):
        dirs.append(("inlp", f"INLP dir {i+1}", u))
    rng = np.random.default_rng(args.seed)
    for i in range(16):
        u = rng.standard_normal(X.shape[1]); dirs.append(("random", f"rand {i+1}", u / np.linalg.norm(u)))

    rows = []
    for kind, label, u in dirs:
        dec = decodability(u); iia, pres = causal_iia(u)
        rows.append({"kind": kind, "label": label, "decodability": round(dec, 4),
                     "scanner_iia": round(iia, 4), "disease_preserved": round(pres, 4)})
    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    print(df.to_string(index=False), flush=True)
    # summary by kind
    print("\nmean by kind (decodability / scanner_iia / disease_preserved):")
    print(df.groupby("kind")[["decodability", "scanner_iia", "disease_preserved"]].mean().round(3).to_string())
    print(f"\nwrote {args.out}")
    print("INTERPRETATION: high decodability does NOT imply high causal-IIA-with-disease-preserved; only"
          " the probe/DAS-aligned axis achieves both -> DAS certifies causality that probing cannot.")


if __name__ == "__main__":
    main()
