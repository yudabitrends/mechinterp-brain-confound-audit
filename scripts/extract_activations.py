#!/usr/bin/env python
"""Phase 1a: extract + cache MultiViT2 activations for one cohort/checkpoint.

Saves:
  <out>/acts_<cohort>_<position>.pt   : dict {hook_name -> (N, d) float32 tensor}
  <out>/labels_<cohort>.csv           : subject_id, label, site, cohort, population

Example:
  python scripts/extract_activations.py --cohort COBRE \
      --ckpt /home/users/ybi3/MultiViT2/outputs/p6c_t4_site_loso/COBRE/best.pt \
      --position cls --device cuda --out outputs/activations
"""
import argparse, json, os, sys
import pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from geomultivit.data.dataset import MultiModalH5Dataset
from geomultivit.data.multicohort import load_geometric_cohort
from mib.extract import load_model, ActivationExtractor, ExtractConfig

H5 = "/home/users/ybi3/data/szdataset_modified.h5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, help="COBRE | FBIRN | ChineseSZ | PK_MPRC")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--position", default="cls", choices=["cls", "tokens"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", default="outputs/activations")
    ap.add_argument("--tag", default="", help="filename prefix, e.g. HOLD_ (default reproduces old names)")
    ap.add_argument("--split-info", default=None, help="path to split_info.json for train/test tagging")
    ap.add_argument("--which", default="all", choices=["all", "train", "test"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    full = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False)
    gdf = full.df
    sub = gdf[gdf["cohort"] == args.cohort].copy()

    split_map = {}
    if args.split_info:
        si = json.load(open(args.split_info))
        split_map = {**{s: "train" for s in si["train_ids"]}, **{s: "test" for s in si["test_ids"]}}
        keep = set(split_map) if args.which == "all" else {s for s, v in split_map.items() if v == args.which}
        sub = sub[sub["SubjectID"].isin(keep)].copy()

    # dedup multi-session rows -> one row per subject (training was deduped; keeps analysis
    # samples independent and N consistent with split_info). 63 dups in the 4-cohort set.
    sub = sub.drop_duplicates("SubjectID").reset_index(drop=True)
    ds = MultiModalH5Dataset(sub, volume_shape=(96, 112, 96), num_icns=53,
                             require_sMRI=True, cache_dir="outputs/cache", augment=False)
    print(f"[{args.cohort}|{args.which}] subjects with resolvable sMRI: {len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = load_model(args.ckpt, device=dev)
    cfg = ExtractConfig(ckpt_path=args.ckpt, position=args.position, device=dev,
                        max_batches=args.max_batches)
    with ActivationExtractor(model, cfg) as ext:
        acts, labels = ext.run(loader)

    torch.save(acts, f"{args.out}/acts_{args.tag}{args.cohort}_{args.position}.pt")

    # join phenotype metadata by subject_id (dedup multi-session dups -> scalar lookups)
    meta = sub.drop_duplicates("SubjectID").set_index("SubjectID")
    rows = [{"subject_id": s, "label": int(y),
             "site": meta.loc[s, "site"] if s in meta.index else "NA",
             "cohort": args.cohort,
             "population": meta.loc[s, "population"] if s in meta.index else "NA",
             "split": split_map.get(s, "NA")}
            for s, y in zip(labels["subject_id"], labels["label"])]
    pd.DataFrame(rows).to_csv(f"{args.out}/labels_{args.tag}{args.cohort}.csv", index=False)
    print(f"saved {len(acts)} hook points x {len(rows)} subjects -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
