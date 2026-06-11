#!/usr/bin/env python
"""Reviewer #2: upgrade the within-country site-axis from linear sex-residualization to sex STRATIFICATION.
Re-run the k=1 site-IIA (US COBRE-vs-Scanner2, China GZ-vs-ZMD) within the male-only subset (the larger,
imbalance-driving stratum) over several seeds; if the IIA holds in a single-sex subset, sex cannot be driving it.
Sex from szdataset_modified.h5 (Gender), mapped to fused subject_id. Cached fused + frozen head; no model forward.
Writes outputs/sae_ckpts/site_iia_sexstrat.csv.
"""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mib import das as D
from ctrl_das_null import load_head, make_pairs

MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"
H5 = "/home/users/ybi3/data/szdataset_modified.h5"


def sex_map():
    m = {}
    for grp in ("train", "test"):
        df = pd.read_hdf(H5, grp)
        for sidv, g in zip(df["SubjectID"].astype(str), df["Gender"]):
            try: m[sidv] = int(float(g))
            except (ValueError, TypeError): pass
    return m


def iia_for(h, axis, hW, hb, tri, tei, seed, pairs=2500):
    aw = torch.zeros(h.shape[1]); ab = torch.zeros(())  # probe fit inside? -> use logistic
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=3000).fit(h[tri].numpy(), axis[tri].numpy())
    aw = torch.tensor(lr.coef_.ravel(), dtype=torch.float32); ab = torch.tensor(float(lr.intercept_[0]))
    bt, st = make_pairs(axis, tri, pairs, seed); bv, sv = make_pairs(axis, tei, pairs, seed + 100)
    das = D.train_das(h, axis, hW, hb, aw, ab, bt, st, k=1, steps=500, lr=5e-3, lam=1.0, seed=seed, device="cpu")
    return D.eval_iia(das, h, axis, hW, hb, aw, ab, bv, sv, device="cpu")


def main():
    d = torch.load("outputs/activations/fused_HOLD_ALL.pt", weights_only=True)
    h = d["fused"].float(); site = np.asarray(d["site"]); split = np.asarray(d["split"]); sid = np.asarray(d["subject_id"])
    sm = sex_map(); sex = np.array([sm.get(str(s), -1) for s in sid])
    hW, hb = load_head(MHOLD)
    rows = []
    for name, sites, posv, full in [("US_COBRE_vs_Scanner2", ["COBRE", "Scanner2"], "Scanner2", 0.87),
                                     ("China_GZ_vs_ZMD", ["GZ", "ZMD"], "ZMD", 0.92)]:
        base = np.isin(site, sites)
        for sx, tag in [(1, "male"), (0, "female")]:
            keep = base & (sex == sx)
            tr = np.where((split == "train") & keep)[0]; te = np.where((split == "test") & keep)[0]
            axis = torch.tensor((site == posv).astype(int))
            n_pos_tr = int(axis.numpy()[tr].sum()); n_neg_tr = len(tr) - n_pos_tr
            if len(te) < 12 or min(n_pos_tr, n_neg_tr) < 8:
                rows.append({"axis": name, "stratum": tag, "n_test": len(te), "iia": np.nan, "note": "too few"})
                print(rows[-1], flush=True); continue
            vals = [iia_for(h, axis, hW, hb, torch.tensor(tr), torch.tensor(te), s, pairs=1500)["scanner_iia"] for s in (0, 1)]
            rows.append({"axis": name, "stratum": tag, "n_test": len(te), "n_train": len(tr),
                         "iia_mean": round(float(np.mean(vals)), 3), "iia_sd": round(float(np.std(vals)), 3),
                         "full_mixed_iia": full})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("outputs/sae_ckpts/site_iia_sexstrat.csv", index=False)
    print("\nINTERP: if male-subset site-IIA ~ the full mixed-sex value, the within-country acquisition axis is not "
          "a sex artifact (stratification, stronger than linear residualization).")


if __name__ == "__main__":
    main()
