"""Train MultiViT2 on ABIDE autism (stratified 70/30 held-out), for the replication.
Reuses run_one_fold + the default config; ABIDEDataset supplies the same batch contract.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/scripts")
from mib.abide_data import build_abide_manifest, ABIDEDataset
from train import run_one_fold, set_seed  # noqa: E402

CFG = "/home/users/ybi3/MultiViT2/geometric_multivit/configs/multivit_default.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, default=Path("/home/users/ybi3/MultiViT2/outputs/abide_split"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(CFG).read_text())
    epochs = int(args.epochs if args.epochs is not None else cfg["train"]["epochs"])
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = build_abide_manifest()
    print(f"[ABIDE] N={len(df)} dx={df.Diagnosis.value_counts().to_dict()} sites={df.site.nunique()}", flush=True)
    # stratify by Diagnosis x site, but merge any singleton stratum into a 'rare' bucket
    # (34 sites have several singleton dx x site cells); final fallback = Diagnosis only.
    strat = df["Diagnosis"].astype(str) + "|" + df["site"].astype(str)
    strat = strat.where(strat.map(strat.value_counts()) >= 2, "rare")
    try:
        tr, te = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=strat)
    except ValueError:
        tr, te = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df["Diagnosis"])
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    assert set(tr.SubjectID).isdisjoint(set(te.SubjectID))
    (args.out_dir / "split_info.json").write_text(json.dumps(
        {"seed": args.seed, "n_train": len(tr), "n_test": len(te),
         "train_ids": tr.SubjectID.tolist(), "test_ids": te.SubjectID.tolist()}, indent=2))
    print(f"[ABIDE] train={len(tr)} test={len(te)} | test dx {te.Diagnosis.value_counts().to_dict()}", flush=True)

    cache = "outputs/cache_abide"
    tr_ds = ABIDEDataset(tr, cache_dir=cache, augment=True)
    te_ds = ABIDEDataset(te, cache_dir=cache, augment=False)
    res = run_one_fold(cfg=cfg, train_ds=tr_ds, val_ds=te_ds, out_dir=args.out_dir,
                       device=torch.device(args.device), epochs=epochs)
    print(f"[ABIDE] held-out test best_auc = {res['best_auc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
