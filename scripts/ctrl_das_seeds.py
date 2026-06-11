#!/usr/bin/env python
"""C2c (threat T7) — seed robustness of the k=1 scanner-IIA headline.

The hostile review flagged that the DAS/IIA number came from a single seed (rotation init +
interchange-pair sampling). Here we retrain the k=1 DAS over several seeds and report mean±SD of
held-out scanner-IIA, disease-preservation, and the random-rotation control. Cached fused rep +
frozen head, no model forward. Writes outputs/sae_ckpts/ctrl_das_seeds.csv.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mib import das as D
from ctrl_das_null import load_head, make_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="outputs/activations/fused_HOLD_ALL.pt")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--out", default="outputs/sae_ckpts/ctrl_das_seeds.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    d = torch.load(args.fused, weights_only=True)
    h = d["fused"].float()
    pop = np.asarray(d["population"]); split = np.asarray(d["split"])
    keep = np.isin(pop, ["US", "China"])
    hW, hb = load_head("/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt")
    scanner = torch.tensor((pop == "China").astype(int))
    tr = torch.tensor(np.where((split == "train") & keep)[0])
    te = torch.tensor(np.where((split == "test") & keep)[0])
    lr = LogisticRegression(max_iter=3000).fit(h[tr].numpy(), scanner[tr].numpy())
    scan_w = torch.tensor(lr.coef_.ravel(), dtype=torch.float32)
    scan_b = torch.tensor(float(lr.intercept_[0]), dtype=torch.float32)

    rows = []
    for sd in [int(x) for x in args.seeds.split(",")]:
        bt, st = make_pairs(scanner, tr, args.pairs, sd)
        bv, sv = make_pairs(scanner, te, args.pairs, sd + 100)
        das = D.train_das(h, scanner, hW, hb, scan_w, scan_b, bt, st, k=args.k,
                          steps=args.steps, lr=5e-3, lam=1.0, seed=sd, device=dev)
        m = D.eval_iia(das, h, scanner, hW, hb, scan_w, scan_b, bv, sv, device=dev)
        rnd = D.DASRotation(h.shape[1], k=args.k, seed=sd + 7).to(dev)
        rm = D.eval_iia(rnd, h, scanner, hW, hb, scan_w, scan_b, bv, sv, device=dev)
        rows.append({"seed": sd, "scanner_iia": m["scanner_iia"],
                     "disease_preserved": m["disease_preserved"],
                     "scanner_iia_rand": rm["scanner_iia"]})
        print(f"seed={sd}  IIA={m['scanner_iia']:.3f}  disease_pres={m['disease_preserved']:.3f}"
              f"  (rand {rm['scanner_iia']:.3f})", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print(f"\nk={args.k} over {len(df)} seeds:")
    print(f"  scanner-IIA      = {df.scanner_iia.mean():.3f} ± {df.scanner_iia.std():.3f}")
    print(f"  disease-preserved= {df.disease_preserved.mean():.3f} ± {df.disease_preserved.std():.3f}")
    print(f"  random-rot IIA   = {df.scanner_iia_rand.mean():.3f} ± {df.scanner_iia_rand.std():.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
