#!/usr/bin/env python
"""Phase 3: causal scanner-feature ablation (mechanistic harmonization test).

At a target sMRI layer, ablate the top scanner-tuned SAE features and measure the model's
own disease AUC (from logits) and downstream scanner decodability (population AUC on the
fused head input). Controls: ablate top disease features, ablate random features. Also
reports a dictionary-size-robust disjointness metric: |cos| between the disease and scanner
linear readouts in SAE-feature space.

GPU job (re-runs the model). Writes outputs/sae_ckpts/phase3_ablate_<target>.csv.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from scipy.spatial.distance import cosine as cos_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from geomultivit.data.dataset import MultiModalH5Dataset
from geomultivit.data.multicohort import load_geometric_cohort
from mib.sae import SAEConfig, SparseAutoencoder
from mib.extract import load_model
from mib import patch, probe

H5 = "/home/users/ybi3/data/szdataset_modified.h5"
CANON = "/home/users/ybi3/MultiViT2/outputs/p6c_t4_site_loso/FBIRN_site12/best.pt"


def load_sae(path, device):
    d = torch.load(path, weights_only=True, map_location="cpu")
    sae = SparseAutoencoder(SAEConfig(**d["cfg"]))
    sae.load_state_dict(d["state_dict"])
    return sae.eval().to(device)


def dx_auc_from_logits(logits, y):
    p = torch.softmax(logits, -1)[:, 1].numpy()
    return roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="sMRI_encoder.norm")
    ap.add_argument("--tag", default="ALLtok")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=30, help="# features to ablate per condition")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/sae_ckpts")
    args = ap.parse_args()
    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # --- model, SAE, data ---
    model = load_model(CANON, device=dev)
    sae = load_sae(f"{args.out}/sae_{args.tag}_{args.target}_seed{args.seed}.pt", dev)
    full = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False)
    gdf = full.df
    sub = gdf[gdf["cohort"].isin(["COBRE", "FBIRN", "ChineseSZ", "PK_MPRC"])].copy()
    ds = MultiModalH5Dataset(sub, volume_shape=(96, 112, 96), num_icns=53,
                             require_sMRI=True, cache_dir="outputs/cache", augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    # dedup: some SubjectIDs recur (multi-session) -> scalar dict map, not .loc (avoids Series)
    pop_map = sub.drop_duplicates("SubjectID").set_index("SubjectID")["population"].to_dict()

    # --- rank features by scanner / disease tuning (on cached CLS acts, aligned to labels) ---
    cls = torch.load(f"outputs/activations/acts_ALL_cls.pt", weights_only=True)[args.target]
    lab = pd.read_csv("outputs/activations/labels_ALL.csv")
    F = probe.encode_features(sae.cpu(), cls.numpy()); sae.to(dev)
    y_dx = lab["label"].to_numpy()
    pm = lab["population"].isin(["US", "China"]).to_numpy()
    y_pop = (lab["population"] == "China").astype(int).to_numpy()
    auc_scan = probe.per_feature_auc(F[pm], y_pop[pm])
    auc_dx = probe.per_feature_auc(F, y_dx)
    order_scan = np.argsort(-np.abs(auc_scan - 0.5))
    order_dx = np.argsort(-np.abs(auc_dx - 0.5))
    S_scan = order_scan[:args.k]
    S_dx = order_dx[:args.k]
    live = np.where(F.std(0) > 1e-9)[0]
    g = torch.Generator().manual_seed(args.seed)
    S_rand = live[torch.randperm(len(live), generator=g)[:args.k].numpy()]

    # robust, dictionary-size-independent disjointness: |cos| of dx vs scanner readouts
    Fz = StandardScaler().fit_transform(F)
    w_dx = LogisticRegression(max_iter=2000).fit(Fz, y_dx).coef_.ravel()
    w_pop = LogisticRegression(max_iter=2000).fit(Fz[pm], y_pop[pm]).coef_.ravel()
    readout_cos = 1.0 - cos_dist(w_dx, w_pop)

    # --- run conditions ---
    conds = {"baseline": None, "ablate_scanner": S_scan, "ablate_disease": S_dx,
             "ablate_random": S_rand}
    rows = []
    for name, idx in conds.items():
        r = patch.run_model(model, loader, dev, target_hook=args.target, sae=sae,
                            ablate_idx=idx)
        pop = np.array([pop_map.get(s, "NA") for s in r["subject_id"]])
        keep = np.isin(pop, ["US", "China"])
        y_pop_run = (pop[keep] == "China").astype(int)
        dxa = dx_auc_from_logits(r["logits"], r["y"])
        popa = probe.linear_probe_auc(r["fused"].numpy()[keep], y_pop_run)
        rows.append({"condition": name, "k": args.k, "target": args.target,
                     "disease_auc": round(dxa, 4), "scanner_auc_fused": round(popa, 4),
                     "n": len(r["y"])})
        print(f"{name:16s} disease_auc={dxa:.4f}  scanner_auc(fused)={popa:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df["readout_cos_dx_scanner"] = round(abs(readout_cos), 4)
    out = f"{args.out}/phase3_ablate_{args.target}.csv"
    df.to_csv(out, index=False)
    print(f"\n|cos| dx-vs-scanner readout (dict-size-robust disjointness) = {abs(readout_cos):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
