#!/usr/bin/env python
"""Stage 2 — DAS + Interchange Intervention Accuracy on the scanner ground-truth (M_hold, CPU).

Finds the minimal k-dim subspace of the fused decision rep whose INTERCHANGE (swap source->base)
causally transfers SCANNER while the model's own DISEASE decision (head(h')) is preserved.
Reports held-out scanner-IIA + disease-preservation vs a random-rotation control, swept over k.
This replaces correlational readout-cosine / INLP with a measured causal-faithfulness number.

Runs on cached fused_HOLD_ALL.pt + the model head weights (the cached `fused` IS the head input,
so head(h')=h'@W.T+b is the model's real disease logit). Writes outputs/sae_ckpts/phase5_das_iia.csv.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mib import das as D

MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"


def load_head(ckpt_path):
    # safe loader first (ckpt is config-dict + tensors); fall back only for trusted local ckpts
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    return sd["head.weight"].float(), sd["head.bias"].float()


def make_pairs(scanner, idx, n_pairs, seed):
    g = torch.Generator().manual_seed(seed)
    us = idx[scanner[idx] == 0]; ch = idx[scanner[idx] == 1]
    if len(us) == 0 or len(ch) == 0:
        raise SystemExit("need both scanner groups present")
    b = torch.cat([us[torch.randint(len(us), (n_pairs,), generator=g)],
                   ch[torch.randint(len(ch), (n_pairs,), generator=g)]])
    s = torch.cat([ch[torch.randint(len(ch), (n_pairs,), generator=g)],
                   us[torch.randint(len(us), (n_pairs,), generator=g)]])
    return b, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="outputs/activations/fused_HOLD_ALL.pt")
    ap.add_argument("--ks", default="1,2,4,8,16,32")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/sae_ckpts")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    d = torch.load(args.fused, weights_only=True)
    h = d["fused"].float()
    ydx = d["y_dx"].long()
    pop = np.asarray(d["population"]); split = np.asarray(d["split"])
    keep = np.isin(pop, ["US", "China"])
    hW, hb = load_head(MHOLD)

    # sanity: head(fused) must reproduce the cached logits
    rec = (h @ hW.T + hb)
    err = (rec - d["logits"].float()).abs().max().item()
    print(f"head(fused) vs cached logits max|Δ| = {err:.4f}  (should be ~0)")

    scanner = torch.tensor((pop == "China").astype(int))
    tr_mask = (split == "train") & keep
    te_mask = (split == "test") & keep
    tr_idx = torch.tensor(np.where(tr_mask)[0])
    te_idx = torch.tensor(np.where(te_mask)[0])
    print(f"N_train={len(tr_idx)} N_test={len(te_idx)} (model-unseen)")

    # frozen scanner readout: LogisticRegression on TRAIN fused -> scanner, as a torch linear
    lr = LogisticRegression(max_iter=3000).fit(h[tr_idx].numpy(), scanner[tr_idx].numpy())
    scan_w = torch.tensor(lr.coef_.ravel(), dtype=torch.float32)
    scan_b = torch.tensor(float(lr.intercept_[0]), dtype=torch.float32)
    print(f"frozen scanner probe held-out AUC = "
          f"{roc_auc_score(scanner[te_idx].numpy(), (h[te_idx] @ scan_w + scan_b).numpy()):.3f}")

    bt, st = make_pairs(scanner, tr_idx, n_pairs=args.pairs, seed=args.seed)
    bv, sv = make_pairs(scanner, te_idx, n_pairs=args.pairs, seed=args.seed + 100)

    os.makedirs(args.out, exist_ok=True)
    out = f"{args.out}/phase5_das_iia.csv"
    rows = []
    for k in [int(x) for x in args.ks.split(",")]:
        das = D.train_das(h, scanner, hW, hb, scan_w, scan_b, bt, st, k=k,
                          steps=args.steps, lr=5e-3, lam=1.0, seed=args.seed, device=dev)
        tr_m = D.eval_iia(das, h, scanner, hW, hb, scan_w, scan_b, bv, sv, device=dev)
        rnd = D.DASRotation(h.shape[1], k=k, seed=args.seed + 7).to(dev)
        rn_m = D.eval_iia(rnd, h, scanner, hW, hb, scan_w, scan_b, bv, sv, device=dev)
        rows.append({"k": k,
                     "scanner_iia": tr_m["scanner_iia"], "disease_preserved": tr_m["disease_preserved"],
                     "scanner_iia_rand": rn_m["scanner_iia"], "disease_preserved_rand": rn_m["disease_preserved"]})
        print(f"k={k:>2}  scanner-IIA {tr_m['scanner_iia']:.3f} (rand {rn_m['scanner_iia']:.3f})  "
              f"disease-preserved {tr_m['disease_preserved']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)   # incremental: survive a timeout

    df = pd.DataFrame(rows)
    hit = df[df.scanner_iia >= 0.9]
    kmin = int(hit.k.min()) if len(hit) else None
    print(f"\nminimal causal scanner dim (scanner-IIA≥0.90): k={kmin}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
