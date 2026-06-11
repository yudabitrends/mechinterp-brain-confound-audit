#!/usr/bin/env python
"""Phase 1b: train an SAE dictionary per hook point (multi-seed) on cached activations.

Reads acts_<cohort>_<pos>.pt, trains one SAE per hook point per seed, logs L0 /
explained-variance / dead-feature %, and writes:
  <out>/sae_<cohort>_<hook>_seed<S>.pt   : SAE state_dict + cfg
  <out>/metrics_<cohort>.csv             : one row per (hook, seed) with final metrics

Example:
  python scripts/train_sae.py --acts outputs/activations/acts_COBRE_cls.pt \
      --cohort COBRE --arch topk --k 32 --expansion 16 --seeds 0 1 2 --epochs 200 \
      --device cuda --out outputs/sae_ckpts
"""
import argparse, os, sys
import pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mib.sae import SAEConfig, SparseAutoencoder
from mib import metrics


def train_one(x, cfg, epochs, lr, batch, l1_warmup, device):
    sae = SparseAutoencoder(cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x), batch_size=batch, shuffle=True)
    for ep in range(epochs):
        # linear L1 warm-up prevents early feature death (standard arch only)
        w = min(1.0, (ep + 1) / max(1, l1_warmup)) if cfg.architecture == "standard" else 1.0
        sae.cfg.l1_coef = cfg.l1_coef * w
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            total, _ = sae.loss(xb)
            total.backward()
            opt.step()
            sae.normalize_decoder()
    sae.cfg.l1_coef = cfg.l1_coef
    with torch.no_grad():
        xd = x.to(device)
        x_hat, f = sae(xd)
        m = {"L0": metrics.l0(f), "explained_var": metrics.explained_variance(xd, x_hat),
             "dead_frac": metrics.dead_feature_fraction(f)}
    return sae, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--arch", default="topk", choices=["topk", "standard"])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--expansion", type=int, default=16)
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--l1-warmup", type=int, default=50)
    ap.add_argument("--hooks", nargs="*", default=None, help="subset of hook names; default all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/sae_ckpts")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    acts = torch.load(args.acts, weights_only=True)
    hooks = args.hooks or sorted(acts.keys())

    rows = []
    for hook in hooks:
        x = acts[hook].float()
        d_in = x.shape[-1]
        for seed in args.seeds:
            cfg = SAEConfig(d_in=d_in, expansion=args.expansion, architecture=args.arch,
                            k=args.k, l1_coef=args.l1, seed=seed)
            sae, m = train_one(x, cfg, args.epochs, args.lr, args.batch, args.l1_warmup, dev)
            torch.save({"state_dict": sae.state_dict(), "cfg": cfg.__dict__, "metrics": m},
                       f"{args.out}/sae_{args.cohort}_{hook}_seed{seed}.pt")
            rows.append({"cohort": args.cohort, "hook": hook, "seed": seed,
                         "d_in": d_in, "d_sae": cfg.d_sae, "arch": args.arch, **m})
            print(f"[{args.cohort}|{hook}|s{seed}] "
                  f"L0={m['L0']:.1f} EV={m['explained_var']:.3f} dead={m['dead_frac']:.3f}",
                  flush=True)
    pd.DataFrame(rows).to_csv(f"{args.out}/metrics_{args.cohort}.csv", index=False)
    print(f"wrote metrics_{args.cohort}.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
