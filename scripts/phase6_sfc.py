#!/usr/bin/env python
"""Stage 3 — Sparse Feature Circuits: is the scanner confound DISTRIBUTED across SAE features,
and does a JOINT graph-ablation move scanner where single-layer SAE-atom ablation (the prior NULL,
RESULTS §6) did not?  (M_hold, GPU.)

(1) Attribution-patching node importance toward a frozen scanner readout on the fused rep, across
    the sMRI/FNC residual stack + the ca_b fusion write (`mib.edge_attribution.node_attribution`).
(2) Participation ratio per hook + pooled = how many features carry scanner (HIGH = distributed,
    the redundant-confound prediction).
(3) Joint graph-ablation: SAE-decoded subtraction of the top scanner features at EVERY hook at once
    (`mib.patch.run_model(ablations=...)`), scanner(pop)/disease AUC held-out vs baseline + vs a
    random same-size feature set. Writes phase6_sfc_nodes.csv + phase6_sfc_ablation.csv.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfrac", type=float, default=0.10, help="top-frac features per hook to ablate")
    ap.add_argument("--attr-batches", type=int, default=60, help="cap batches for attribution (memory/time)")
    ap.add_argument("--out", default="outputs/sae_ckpts")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(MHOLD, device=dev)
    saes = {h: load_sae(h, dev) for h in HOOKS}
    loader, sub = build_loader()
    si = json.load(open(SPLIT))
    split_map = {**{s: "train" for s in si["train_ids"]}, **{s: "test" for s in si["test_ids"]}}

    # frozen scanner readout on the fused rep (from the cached fused_HOLD train split)
    fc = torch.load("outputs/activations/fused_HOLD_ALL.pt", weights_only=True)
    Ff = fc["fused"].numpy(); popf = np.asarray(fc["population"]); splf = np.asarray(fc["split"])
    kf = np.isin(popf, ["US", "China"])
    trf = (splf == "train") & kf
    yscan_f = (popf == "China").astype(int)
    scsc = StandardScaler().fit(Ff[trf])
    lr = LogisticRegression(max_iter=3000).fit(scsc.transform(Ff[trf]), yscan_f[trf])
    w = torch.tensor((lr.coef_.ravel() / scsc.scale_), dtype=torch.float32, device=dev)
    b = torch.tensor(float(lr.intercept_[0] - (lr.coef_.ravel() * scsc.mean_ / scsc.scale_).sum()),
                     dtype=torch.float32, device=dev)
    def metric_fn(fused):                      # standardized scanner logit, differentiable
        return fused @ w + b

    # ---- (1) node attribution (capped batches: attribution is a gradient statistic) ----
    attr, totals = EA.node_attribution(model, loader, dev, HOOKS, saes, metric_fn,
                                       max_batches=args.attr_batches)
    pr = {h: EA.participation_ratio(attr[h]) for h in HOOKS}
    nlive = {h: int((attr[h].abs() > 1e-8).sum()) for h in HOOKS}
    nodes = pd.DataFrame([{"hook": h, "abs_total": totals[h], "participation_ratio": pr[h],
                           "n_features": attr[h].numel(), "n_nonzero": nlive[h]} for h in HOOKS])
    os.makedirs(args.out, exist_ok=True)
    nodes.to_csv(f"{args.out}/phase6_sfc_nodes.csv", index=False)
    print(nodes.round(3).to_string(index=False))
    print(f"\npooled participation ratio = "
          f"{EA.participation_ratio(torch.cat([attr[h] for h in HOOKS])):.1f}")

    # ---- (3) joint graph-ablation vs random same-size set ----
    sel = EA.top_features(attr, frac_per_hook=args.kfrac)
    rng = np.random.default_rng(0)
    rnd = {h: torch.tensor(rng.choice(attr[h].numel(), len(sel[h]), replace=False)) for h in HOOKS}

    def eval_cond(ablations, tag):
        r = patch.run_model(model, loader, dev, ablations=ablations)
        d = sub.drop_duplicates("SubjectID").set_index("SubjectID")
        pop = np.array([d.loc[s, "population"] if s in d.index else "NA" for s in r["subject_id"]])
        split = np.array([split_map.get(s, "NA") for s in r["subject_id"]])
        keep = np.isin(pop, ["US", "China"]); tr = (split == "train") & keep; te = (split == "test") & keep
        F = r["fused"].numpy(); yscan = (pop == "China").astype(int); ydx = r["y"].astype(int)
        return {"condition": tag, "n_feats": int(sum(len(v) for v in (sel if "graph" in tag else rnd).values())),
                "scanner_pop_auc": held_out_auc(F, yscan, tr, te),
                "disease_auc": held_out_auc(F, ydx, tr, te),
                "disease_logits": float(roc_auc_score(ydx[te], (r["logits"][:, 1] - r["logits"][:, 0]).numpy()[te]))}

    rows = [eval_cond([], "baseline")]
    rows.append(eval_cond([(h, saes[h], sel[h]) for h in HOOKS], "graph_ablate_scanner"))
    rows.append(eval_cond([(h, saes[h], rnd[h]) for h in HOOKS], "random_same_size_ctrl"))
    abl = pd.DataFrame(rows)
    abl.to_csv(f"{args.out}/phase6_sfc_ablation.csv", index=False)
    print("\n" + abl.round(4).to_string(index=False))
    print(f"\nwrote phase6_sfc_nodes.csv + phase6_sfc_ablation.csv")


if __name__ == "__main__":
    main()
