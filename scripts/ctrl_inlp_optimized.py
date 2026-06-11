#!/usr/bin/env python
"""R3 symmetric control (reviewer): close the 'DAS is optimized, INLP directions are not' asymmetry in
the decodability != causality result. We OPTIMIZE a 1-D causal-transfer direction CONFINED to the span of
the strongly-decodable-but-causally-inert INLP nullspace directions (rounds 2..8), by gradient through the
frozen head, and report the best scanner-IIA it can reach. If even an optimized search inside the decodable
subspace stays at the floor, the dissociation is not an optimization artifact.

Positive control: the same optimization confined to the span of ALL INLP directions (incl. round 1, which is
the probe/DAS-aligned causal axis) should recover the high IIA -- confirming the optimizer works and the one
causal handle lives in round-1's direction, not the redundant decodable remainder.

Cached fused_HOLD_ALL.pt + frozen head; no model forward. Writes outputs/sae_ckpts/inlp_optimized.csv.
"""
import os, sys, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from ctrl_das_null import load_head, make_pairs
from ctrl_decode_vs_causal import inlp_directions

MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"


def optimize_in_subspace(h, B, scanner, hW, hb, sw, sb, bt, st, bv, sv, base_dx_te, scan_src_te,
                         steps=400, lr=5e-2, lam=1.0):
    """Train a unit u in the m-dim subspace B (d,m); the 1-D interchange direction is v=B u (in span B).
    Returns best held-out scanner-IIA + disease-preservation."""
    Bt = torch.tensor(B, dtype=torch.float32)
    u = torch.nn.Parameter(torch.randn(B.shape[1])); opt = torch.optim.Adam([u], lr=lr)
    s_src = scanner[st].float()
    base_dx = (h[bt] @ hW.T + hb).argmax(1)
    for _ in range(steps):
        opt.zero_grad()
        v = Bt @ (u / (u.norm() + 1e-8))                       # (d,) unit-ish direction in span(B)
        coef = ((h[st] - h[bt]) @ v).unsqueeze(1)
        hp = h[bt] + coef * v
        scan_logit = hp @ sw + sb
        dx_logit = hp @ hW.T + hb
        loss = F.binary_cross_entropy_with_logits(scan_logit, s_src) + lam * F.cross_entropy(dx_logit, base_dx)
        loss.backward(); opt.step()
    with torch.no_grad():
        v = Bt @ (u / (u.norm() + 1e-8))
        coef = ((h[sv] - h[bv]) @ v).unsqueeze(1)
        hp = h[bv] + coef * v
        scan_pred = (hp @ sw + sb) > 0
        dx_pred = (hp @ hW.T + hb).argmax(1)
        iia = (scan_pred.long() == scan_src_te).float().mean().item()
        pres = (dx_pred == base_dx_te).float().mean().item()
    return iia, pres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="outputs/activations/fused_HOLD_ALL.pt")
    ap.add_argument("--rounds", type=int, default=8); ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/sae_ckpts/inlp_optimized.csv")
    args = ap.parse_args()
    d = torch.load(args.fused, weights_only=True)
    h = d["fused"].float(); X = h.numpy()
    pop = np.asarray(d["population"]); split = np.asarray(d["split"]); keep = np.isin(pop, ["US", "China"])
    hW, hb = load_head(MHOLD)
    scanner = torch.tensor((pop == "China").astype(int))
    tr = np.where((split == "train") & keep)[0]; te = np.where((split == "test") & keep)[0]
    tri, tei = torch.tensor(tr), torch.tensor(te)
    lr = LogisticRegression(max_iter=3000).fit(X[tr], scanner[tr].numpy())
    sw = torch.tensor(lr.coef_.ravel(), dtype=torch.float32); sb = torch.tensor(float(lr.intercept_[0]))
    bt, st = make_pairs(scanner, tri, args.pairs, args.seed)
    bv, sv = make_pairs(scanner, tei, args.pairs, args.seed + 100)
    base_dx_te = (h[bv] @ hW.T + hb).argmax(1); scan_src_te = scanner[sv]

    dirs = inlp_directions(X[tr], scanner[tr].numpy(), args.rounds)     # list of d-vectors, unit
    dirs = np.array(dirs)                                               # (R, d)

    def qr_basis(V):
        Q, _ = np.linalg.qr(V.T); return Q                              # (d, rank)

    rows = []
    def rec(name, B):
        iia, pres = optimize_in_subspace(h, B, scanner, hW, hb, sw, sb, bt, st, bv, sv,
                                         base_dx_te, scan_src_te, steps=args.steps)
        rows.append({"subspace": name, "dim": B.shape[1], "optimized_iia": round(iia, 3),
                     "disease_pres": round(pres, 3)})
        print(f"{name:34s} dim={B.shape[1]:<3} optimized scanner-IIA={iia:.3f}  disease_pres={pres:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    # (1) decodable-but-inert subspace: INLP rounds 2..R (skip round 1 = the probe/causal axis)
    rec("INLP_dirs_2to%d_decodable_inert" % args.rounds, qr_basis(dirs[1:]))
    # (2) positive control: span of ALL INLP dirs (incl round 1, the causal axis)
    rec("INLP_dirs_1to%d_incl_causal" % args.rounds, qr_basis(dirs))
    # (3) reference: the round-1 direction alone (the probe/causal axis)
    rec("INLP_dir_1_alone_causal", qr_basis(dirs[:1]))
    print(f"\nwrote {args.out}")
    print("INTERPRETATION: if (1) stays near the 0.12 floor while (2)/(3) reach ~0.88, then even an OPTIMIZED"
          " search inside the strongly-decodable INLP remainder finds no causal-transfer handle -- the"
          " decodability!=causality dissociation is not a DAS-optimization artifact.")


if __name__ == "__main__":
    main()
