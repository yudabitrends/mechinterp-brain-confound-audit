#!/usr/bin/env python
"""Phase 3c: multi-layer SAE-feature ablation (honest SAE-route comparison to 3b).

Tests whether the SAE dictionary itself can harmonize: ablate ALL scanner-tuned features
(population-AUC beyond a threshold) across every sMRI residual layer simultaneously, and
measure disease vs scanner decodability. Compared against 3b's direction-erasure result,
this answers 'is a sparse dictionary a usable harmonization tool, or does it need the full
linear subspace?'. Writes outputs/sae_ckpts/phase3c_multilayer.csv.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from geomultivit.data.dataset import MultiModalH5Dataset
from geomultivit.data.multicohort import load_geometric_cohort
from mib.sae import SAEConfig, SparseAutoencoder
from mib.extract import load_model
from mib import patch, probe
from sklearn.metrics import roc_auc_score

H5 = "/home/users/ybi3/data/szdataset_modified.h5"
CANON = "/home/users/ybi3/MultiViT2/outputs/p6c_t4_site_loso/FBIRN_site12/best.pt"
SMRI_TARGETS = [f"sMRI_encoder.blocks.{i}" for i in range(6)] + ["sMRI_encoder.norm"]


def load_sae(path, device):
    d = torch.load(path, weights_only=True, map_location="cpu")
    sae = SparseAutoencoder(SAEConfig(**d["cfg"]))
    sae.load_state_dict(d["state_dict"])
    return sae.eval().to(device)


def dx_auc(logits, y):
    return roc_auc_score(y, torch.softmax(logits, -1)[:, 1].numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ALLtok")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auc-thresh", type=float, default=0.15, help="|pop_auc-0.5| to call a feature 'scanner'")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/sae_ckpts")
    args = ap.parse_args()
    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    model = load_model(CANON, device=dev)
    cls = torch.load("outputs/activations/acts_ALL_cls.pt", weights_only=True)
    lab = pd.read_csv("outputs/activations/labels_ALL.csv")
    pm = lab["population"].isin(["US", "China"]).to_numpy()
    y_pop = (lab["population"] == "China").astype(int).to_numpy()

    # per-layer: load SAE, pick scanner features by population-AUC threshold
    saes, scan_idx, rnd_idx, total = {}, {}, {}, 0
    g = torch.Generator().manual_seed(args.seed)
    for t in SMRI_TARGETS:
        sae = load_sae(f"{args.out}/sae_{args.tag}_{t}_seed{args.seed}.pt", dev)
        saes[t] = sae
        F = probe.encode_features(sae.cpu(), cls[t].numpy()); sae.to(dev)
        a = probe.per_feature_auc(F[pm], y_pop[pm])
        sel = np.where(np.abs(a - 0.5) >= args.auc_thresh)[0]
        scan_idx[t] = sel
        live = np.where(F.std(0) > 1e-9)[0]
        rnd_idx[t] = live[torch.randperm(len(live), generator=g)[:len(sel)].numpy()]
        total += len(sel)
        print(f"{t:30s} scanner-features={len(sel):4d} (live {len(live)})", flush=True)
    print(f"TOTAL scanner features ablated across sMRI = {total}", flush=True)

    full = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False).df
    sub = full[full["cohort"].isin(["COBRE", "FBIRN", "ChineseSZ", "PK_MPRC"])].copy()
    ds = MultiModalH5Dataset(sub, volume_shape=(96, 112, 96), num_icns=53,
                             require_sMRI=True, cache_dir="outputs/cache", augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    pop_map = sub.drop_duplicates("SubjectID").set_index("SubjectID")["population"].to_dict()

    conds = {
        "baseline": [],
        "ablate_scanner_multilayer": [(t, saes[t], scan_idx[t]) for t in SMRI_TARGETS],
        "ablate_random_multilayer": [(t, saes[t], rnd_idx[t]) for t in SMRI_TARGETS],
    }
    rows = []
    for name, abls in conds.items():
        r = patch.run_model(model, loader, dev, ablations=abls)
        pop = np.array([pop_map.get(s, "NA") for s in r["subject_id"]])
        keep = np.isin(pop, ["US", "China"])
        popa = probe.linear_probe_auc(r["fused"].numpy()[keep], (pop[keep] == "China").astype(int))
        dxa = dx_auc(r["logits"], r["y"])
        rows.append({"condition": name, "n_ablated": total, "disease_auc": round(dxa, 4),
                     "scanner_auc_fused": round(popa, 4)})
        print(f"{name:28s} disease={dxa:.4f} scanner(fused)={popa:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(f"{args.out}/phase3c_multilayer.csv", index=False)
    print(f"wrote {args.out}/phase3c_multilayer.csv")


if __name__ == "__main__":
    main()
