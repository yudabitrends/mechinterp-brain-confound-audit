#!/usr/bin/env python
"""#3 (more advanced method) — causal scrubbing of the scanner circuit (Chan et al. 2022).

The paper's circuit hypothesis: the scanner readout is a function of the SCANNER CLASS carried by the
branch CLS tokens (FNC and, redundantly, sMRI), re-injected through fusion. Causal scrubbing tests this
by resample-ablation: replace a node's activation with one from another input that AGREES on the
hypothesis-relevant feature (same scanner class) -- the output should be PRESERVED (the specific subject
is irrelevant, only its scanner class matters); replace with a DIFFERENT scanner class -- the output
should TRANSFER. A hypothesis that survives both is causally validated, a stronger statement than path
patching.

Per branch-CLS hook we measure, downstream (held-out, no grad): (a) same-scanner scrub -> scanner readout
agreement with the ORIGINAL scanner (expect high = scanner is class-level, subject-invariant); (b)
different-scanner scrub -> agreement with the SOURCE scanner (expect high = transfers); disease preservation.
Reuses phase5b IndexedWriter + run_model. GPU. Writes outputs/sae_ckpts/phase5d_scrub.csv.
"""
import json, os, sys, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from geomultivit.data.dataset import MultiModalH5Dataset
from geomultivit.data.multicohort import load_geometric_cohort
from mib.extract import load_model
from mib import patch

H5 = "/home/users/ybi3/data/szdataset_modified.h5"
MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"
SPLIT = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/split_info.json"
HOOKS = ["sFNC_encoder.norm", "sMRI_encoder.norm", "cross_blocks.1.ca_b"]


def loader_for(gdf, ids, bs=16):
    sub = gdf[gdf.SubjectID.isin(set(ids))].drop_duplicates("SubjectID").reset_index(drop=True)
    ds = MultiModalH5Dataset(sub, (96, 112, 96), 53, require_sMRI=True, cache_dir="outputs/cache")
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4), sub


class CLSWriter:
    """Overwrite CLS token t=0 with precomputed per-subject values, consumed in loader order."""
    def __init__(self, vals):
        self.v, self.i = vals, 0
    def __call__(self, _m, _i, out):
        x = (out[0] if isinstance(out, tuple) else out).clone()
        b = x.shape[0]
        x[:, 0] = self.v[self.i:self.i + b].to(x.device, x.dtype); self.i += b
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/sae_ckpts/phase5d_scrub.csv"); args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print(f"device={dev}", flush=True)
    si = json.load(open(SPLIT))
    gdf = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False).df
    gdf = gdf[gdf.cohort.isin(["COBRE", "FBIRN", "ChineseSZ", "PK_MPRC"])]
    model = load_model(MHOLD, device=dev)
    tr_loader, tr_sub = loader_for(gdf, si["train_ids"]); te_loader, te_sub = loader_for(gdf, si["test_ids"])

    def pop_of(sub, sids):
        m = sub.drop_duplicates("SubjectID").set_index("SubjectID")
        return np.array([m.loc[s, "population"] for s in sids])

    base_tr = patch.run_model(model, tr_loader, dev); base_te = patch.run_model(model, te_loader, dev)
    pop_tr = pop_of(tr_sub, base_tr["subject_id"]); pop_te = pop_of(te_sub, base_te["subject_id"])
    yscan_tr = (pop_tr == "China").astype(int); yscan_te = (pop_te == "China").astype(int)
    keep_te = np.isin(pop_te, ["US", "China"]); base_dx_te = base_te["logits"].argmax(1).numpy()
    # frozen scanner readout on train fused
    sc = StandardScaler().fit(base_tr["fused"].numpy()[np.isin(pop_tr, ["US", "China"])])
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(base_tr["fused"].numpy()[np.isin(pop_tr, ["US", "China"])]),
                                               yscan_tr[np.isin(pop_tr, ["US", "China"])])
    def scan_pred(F):
        return (lr.predict(sc.transform(F)))

    g = np.random.default_rng(0)
    rows = []
    for H in HOOKS:
        cls_te = patch.capture_hook_tokens(model, te_loader, dev, H)[0][:, 0]   # (Nte,d) loader order
        # same-scanner and different-scanner source assignment
        us = np.where((yscan_te == 0) & keep_te)[0]; ch = np.where((yscan_te == 1) & keep_te)[0]
        src_same = np.arange(len(yscan_te)); src_diff = np.arange(len(yscan_te))
        for i in range(len(yscan_te)):
            if not keep_te[i]:
                continue
            same_pool = us if yscan_te[i] == 0 else ch
            diff_pool = ch if yscan_te[i] == 0 else us
            src_same[i] = g.choice(same_pool); src_diff[i] = g.choice(diff_pool)

        def run_scrub(src_idx):
            hk = CLSWriter(cls_te[src_idx])
            hh = patch._resolve(model, H).register_forward_hook(hk)
            r = patch.run_model(model, te_loader, dev)
            hh.remove()
            sp = scan_pred(r["fused"].numpy()); dx = r["logits"].argmax(1).numpy()
            return sp, dx
        sp_same, dx_same = run_scrub(src_same)
        sp_diff, dx_diff = run_scrub(src_diff)
        # same-scanner scrub: scanner readout should match ORIGINAL scanner (subject-invariant)
        same_preserve = float((sp_same[keep_te] == yscan_te[keep_te]).mean())
        # different-scanner scrub: should TRANSFER to source scanner (= opposite of original here)
        diff_transfer = float((sp_diff[keep_te] == (1 - yscan_te)[keep_te]).mean())
        # disease preservation under same-scanner scrub (hypothesis: scanner-class swap shouldn't move disease much)
        dis_pres_same = float((dx_same[keep_te] == base_dx_te[keep_te]).mean())
        rows.append({"hook": H, "scrub_same_scanner_preserved": round(same_preserve, 3),
                     "scrub_diff_scanner_transferred": round(diff_transfer, 3),
                     "disease_pres_same_scrub": round(dis_pres_same, 3)})
        print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("INTERPRETATION: high same-scanner-preserved + high diff-scanner-transferred = the scanner readout"
          " depends on the CLS scanner CLASS, not the specific subject -> the scanner-circuit hypothesis is"
          " causally validated by scrubbing (stronger than path patching).")


if __name__ == "__main__":
    main()
