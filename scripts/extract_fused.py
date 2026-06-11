#!/usr/bin/env python
"""Phase 3b prep: one forward pass of M* over the 4 cohorts, saving the fused decision
representation (head input, 512-d) + logits + per-subject dx/population/site labels.

All downstream erasure/probing analysis (phase3b_erase.py) then runs on CPU off this cache.
Writes outputs/activations/fused_ALL.pt.
"""
import os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from geomultivit.data.dataset import MultiModalH5Dataset
from geomultivit.data.multicohort import load_geometric_cohort
from mib.extract import load_model
from mib import patch

H5 = "/home/users/ybi3/data/szdataset_modified.h5"
CANON = "/home/users/ybi3/MultiViT2/outputs/p6c_t4_site_loso/FBIRN_site12/best.pt"


def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CANON)
    ap.add_argument("--tag", default="", help="output prefix, e.g. HOLD_")
    ap.add_argument("--split-info", default=None)
    args = ap.parse_args()

    split_map = {}
    if args.split_info:
        si = json.load(open(args.split_info))
        split_map = {**{s: "train" for s in si["train_ids"]}, **{s: "test" for s in si["test_ids"]}}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device=dev)
    full = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False)
    gdf = full.df
    sub = gdf[gdf["cohort"].isin(["COBRE", "FBIRN", "ChineseSZ", "PK_MPRC"])].copy()
    sub = sub.drop_duplicates("SubjectID").reset_index(drop=True)  # one row/subject (see audit)
    ds = MultiModalH5Dataset(sub, volume_shape=(96, 112, 96), num_icns=53,
                             require_sMRI=True, cache_dir="outputs/cache", augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    r = patch.run_model(model, loader, dev)   # baseline, no ablation; captures head input
    d = sub.drop_duplicates("SubjectID").set_index("SubjectID")
    pop = np.array([d.loc[s, "population"] if s in d.index else "NA" for s in r["subject_id"]])
    site = np.array([d.loc[s, "site"] if s in d.index else "NA" for s in r["subject_id"]])
    split = [split_map.get(s, "NA") for s in r["subject_id"]]

    # save labels as plain lists so the cache loads with weights_only=True (safe unpickler)
    torch.save({"fused": r["fused"], "logits": r["logits"],
                "y_dx": torch.as_tensor(np.asarray(r["y"])),
                "population": pop.tolist(), "site": site.tolist(), "split": list(split),
                "subject_id": list(r["subject_id"])},
               f"outputs/activations/fused_{args.tag}ALL.pt")
    print(f"saved fused {tuple(r['fused'].shape)} for N={len(r['y'])} | "
          f"pop {pd.Series(pop).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
