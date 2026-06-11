#!/usr/bin/env python
"""E1 (reviewer R5) — feature-ablation COUNT SWEEP: is the ablation-null an artifact of deleting
too many features at once?

Reviewer concern: "you ablate 5,726 features, of course disease breaks." We answer by sweeping the
NUMBER of ablated scanner features N in {100, 500, 1000, 2000, 4000, 5726}, always choosing the
globally most scanner-important features (by |attribution|) across all hooks, and propagating the
ablation through the model. Prediction of the distributed-redundancy thesis: held-out scanner AUC
stays ~flat at every N (you cannot localize/remove the confound by deleting features), while disease
decodability collapses monotonically as you delete more. A small-N ablation that already breaks
disease without moving scanner is the strongest possible form of the null.

Reuses phase6_sfc.py machinery verbatim: attribution-patched node importance (mib.edge_attribution),
SAE-decoded subtractive ablation propagated through the model (mib.patch.run_model). GPU.
Writes outputs/sae_ckpts/phase6_count_sweep.csv.
"""
import argparse, os, sys, json
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from geomultivit.data.dataset import MultiModalH5Dataset
from geomultivit.data.multicohort import load_geometric_cohort
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from mib.extract import load_model
from mib.sae import SAEConfig, SparseAutoencoder
from mib import patch, edge_attribution as EA

H5 = "/home/users/ybi3/data/szdataset_modified.h5"
MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"
SPLIT = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/split_info.json"
HOOKS = [f"sMRI_encoder.blocks.{i}" for i in range(6)] + ["sMRI_encoder.norm"] \
      + [f"sFNC_encoder.blocks.{i}" for i in range(4)] + ["sFNC_encoder.norm"] \
      + ["cross_blocks.0.ca_b", "cross_blocks.1.ca_b"]


def load_sae(hook, device, tag="HOLDtrain", seed=0):
    d = torch.load(f"outputs/sae_ckpts/sae_{tag}_{hook}_seed{seed}.pt", weights_only=True, map_location="cpu")
    sae = SparseAutoencoder(SAEConfig(**d["cfg"])); sae.load_state_dict(d["state_dict"])
    return sae.eval().to(device)


def build_loader():
    full = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False)
    sub = full.df[full.df["cohort"].isin(["COBRE", "FBIRN", "ChineseSZ", "PK_MPRC"])].copy()
    ds = MultiModalH5Dataset(sub, volume_shape=(96, 112, 96), num_icns=53,
                             require_sMRI=True, cache_dir="outputs/cache", augment=False)
    return DataLoader(ds, batch_size=4, shuffle=False, num_workers=4), sub


def held_out_auc(F, y, tr, te):
    sc = StandardScaler().fit(F[tr]); c = LogisticRegression(max_iter=2000).fit(sc.transform(F[tr]), y[tr])
    return float(roc_auc_score(y[te], c.predict_proba(sc.transform(F[te]))[:, 1]))


def select_global_topN(attr, N):
    """Pick the globally N most scanner-important features (|attribution|) across all hooks,
    returned as per-hook index tensors."""
    hooks = list(attr.keys())
    sizes = [attr[h].numel() for h in hooks]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    flat = torch.cat([attr[h].abs() for h in hooks])
    gi = torch.topk(flat, min(N, flat.numel())).indices.cpu().numpy()
    sel = {h: [] for h in hooks}
    hook_of = np.searchsorted(offsets, gi, side="right") - 1
    for g, hk in zip(gi, hook_of):
        sel[hooks[hk]].append(int(g - offsets[hk]))
    return {h: torch.tensor(v, dtype=torch.long) for h, v in sel.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="100,500,1000,2000,4000,5726")
    ap.add_argument("--attr-batches", type=int, default=60)
    ap.add_argument("--out", default="outputs/sae_ckpts/phase6_count_sweep.csv")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)

    model = load_model(MHOLD, device=dev)
    saes = {h: load_sae(h, dev) for h in HOOKS}
    loader, sub = build_loader()
    si = json.load(open(SPLIT))
    split_map = {**{s: "train" for s in si["train_ids"]}, **{s: "test" for s in si["test_ids"]}}

    # frozen scanner readout on the fused rep (cached fused train split), as a differentiable metric
    fc = torch.load("outputs/activations/fused_HOLD_ALL.pt", weights_only=True)
    Ff = fc["fused"].numpy(); popf = np.asarray(fc["population"]); splf = np.asarray(fc["split"])
    kf = np.isin(popf, ["US", "China"]); trf = (splf == "train") & kf
    yscan_f = (popf == "China").astype(int)
    scsc = StandardScaler().fit(Ff[trf])
    lr = LogisticRegression(max_iter=3000).fit(scsc.transform(Ff[trf]), yscan_f[trf])
    w = torch.tensor((lr.coef_.ravel() / scsc.scale_), dtype=torch.float32, device=dev)
    b = torch.tensor(float(lr.intercept_[0] - (lr.coef_.ravel() * scsc.mean_ / scsc.scale_).sum()),
                     dtype=torch.float32, device=dev)
    def metric_fn(fused):
        return fused @ w + b

    attr, _ = EA.node_attribution(model, loader, dev, HOOKS, saes, metric_fn, max_batches=args.attr_batches)
    total_avail = int(sum(a.numel() for a in attr.values()))
    print(f"total features available = {total_avail}", flush=True)

    d = sub.drop_duplicates("SubjectID").set_index("SubjectID")

    def eval_cond(ablations):
        r = patch.run_model(model, loader, dev, ablations=ablations)
        pop = np.array([d.loc[s, "population"] if s in d.index else "NA" for s in r["subject_id"]])
        split = np.array([split_map.get(s, "NA") for s in r["subject_id"]])
        keep = np.isin(pop, ["US", "China"]); tr = (split == "train") & keep; te = (split == "test") & keep
        F = r["fused"].numpy(); yscan = (pop == "China").astype(int); ydx = r["y"].astype(int)
        return {"scanner_pop_auc": held_out_auc(F, yscan, tr, te),
                "disease_auc": held_out_auc(F, ydx, tr, te),
                "disease_logits": float(roc_auc_score(ydx[te], (r["logits"][:, 1] - r["logits"][:, 0]).numpy()[te]))}

    rows = [{"n_ablated": 0, **eval_cond([])}]
    print(f"N=0 (baseline): {rows[-1]}", flush=True)
    for N in [int(x) for x in args.counts.split(",")]:
        sel = select_global_topN(attr, N)
        n_real = int(sum(len(v) for v in sel.values()))
        m = eval_cond([(h, saes[h], sel[h]) for h in HOOKS if len(sel[h])])
        rows.append({"n_ablated": n_real, **m})
        print(f"N={N} (real {n_real}): scanner {m['scanner_pop_auc']:.3f}  "
              f"disease {m['disease_auc']:.3f}  disease_logit {m['disease_logits']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    print(f"\nwrote {args.out}")
    print("INTERPRETATION: flat scanner across N + monotone disease collapse = the null is NOT an"
          " artifact of ablation size; the confound has no localized feature handle at any scale.")


if __name__ == "__main__":
    main()
