#!/usr/bin/env python
"""Gap-closer: forward-propagated causal interchange at an EARLIER hook (sFNC_encoder.norm).

Rebuts "DAS-on-fused = INLP++". Take the scanner direction at the FNC branch output (pre-fusion,
frozen probe), do a 1-D counterfactual interchange (replace each base subject's component along it
with the source subject's), then propagate through the rest of the model. Measure DOWNSTREAM:
does the model's fused-scanner flip to source (scanner-IIA) while its disease decision is preserved?
Control = a random direction. Held-out test of M_hold. Writes outputs/sae_ckpts/phase5b_das_earlier.csv.
"""
import json, os, sys
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
HOOK = "sFNC_encoder.norm"


def loader_for(gdf, ids):
    sub = gdf[gdf.SubjectID.isin(set(ids))].drop_duplicates("SubjectID").reset_index(drop=True)
    return DataLoader(MultiModalH5Dataset(sub, (96, 112, 96), 53, require_sMRI=True,
                      cache_dir="outputs/cache"), batch_size=16, shuffle=False, num_workers=4), sub


class IndexedWriter:
    """Forward hook overwriting token `t` of each row with precomputed per-subject values,
    consumed in loader order (shuffle=False keeps it aligned with the capture order)."""
    def __init__(self, values, t=0):
        self.v, self.t, self.i = values, t, 0
    def __call__(self, _m, _i, out):
        x = (out[0] if isinstance(out, tuple) else out).clone()
        b = x.shape[0]
        x[:, self.t] = self.v[self.i:self.i + b].to(x.device, x.dtype); self.i += b
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    si = json.load(open(SPLIT))
    gdf = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False).df
    gdf = gdf[gdf.cohort.isin(["COBRE", "FBIRN", "ChineseSZ", "PK_MPRC"])]
    model = load_model(MHOLD, device=dev)
    tr_loader, tr_sub = loader_for(gdf, si["train_ids"])
    te_loader, te_sub = loader_for(gdf, si["test_ids"])

    def pop_of(sub, sids):
        m = sub.drop_duplicates("SubjectID").set_index("SubjectID")
        return np.array([m.loc[s, "population"] for s in sids])

    # CLS at the earlier hook (train for probe, test for interchange)
    tr_tok, tr_sids = patch.capture_hook_tokens(model, tr_loader, dev, HOOK)
    te_tok, te_sids = patch.capture_hook_tokens(model, te_loader, dev, HOOK)
    cls_tr, cls_te = tr_tok[:, 0].numpy(), te_tok[:, 0].numpy()           # (N,256)
    pop_tr, pop_te = pop_of(tr_sub, tr_sids), pop_of(te_sub, te_sids)
    keep_tr = np.isin(pop_tr, ["US", "China"]); keep_te = np.isin(pop_te, ["US", "China"])
    yscan_tr = (pop_tr == "China").astype(int); yscan_te = (pop_te == "China").astype(int)

    # frozen scanner direction at the earlier hook (unit vector in raw CLS space)
    sc = StandardScaler().fit(cls_tr[keep_tr])
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(cls_tr[keep_tr]), yscan_tr[keep_tr])
    u = (lr.coef_.ravel() / sc.scale_); u = u / np.linalg.norm(u)         # de-standardized, unit
    g = np.random.default_rng(0)
    u_rand = g.standard_normal(u.shape); u_rand /= np.linalg.norm(u_rand)

    # baseline forward (test): real fused + logits, in te_loader order
    base = patch.run_model(model, te_loader, dev)
    base_dx = base["logits"].argmax(1).numpy()
    Xf_tr = patch.run_model(model, tr_loader, dev)["fused"].numpy()       # train fused for readout
    rdr = LogisticRegression(max_iter=2000)
    rsc = StandardScaler().fit(Xf_tr); rdr.fit(rsc.transform(Xf_tr), yscan_tr)

    # pair each test subject (base) with a random source of OPPOSITE scanner
    idx = np.where(keep_te)[0]
    us, ch = idx[yscan_te[idx] == 0], idx[yscan_te[idx] == 1]
    src = np.array([g.choice(ch) if yscan_te[i] == 0 else g.choice(us) for i in range(len(cls_te))])
    src_scan = 1 - yscan_te                                               # source = opposite group

    def run_interchange(direction):
        proj_b = cls_te @ direction
        proj_s = cls_te[src] @ direction
        patched = cls_te + np.outer(proj_s - proj_b, direction)          # 1-D interchange
        writer = IndexedWriter(torch.tensor(patched, dtype=torch.float32), t=0)
        r = patch.run_model(model, te_loader, dev, extra_hooks=[(HOOK, writer)])
        fused_scan = rdr.predict(rsc.transform(r["fused"].numpy()))      # downstream scanner pred
        dx = r["logits"].argmax(1).numpy()
        iia = float((fused_scan[keep_te] == src_scan[keep_te]).mean())
        pres = float((dx[keep_te] == base_dx[keep_te]).mean())
        return iia, pres

    rows = []
    for name, d in [("scanner_direction", u), ("random_direction(ctrl)", u_rand)]:
        iia, pres = run_interchange(d)
        rows.append({"intervention": name, "hook": HOOK, "scanner_IIA_downstream": round(iia, 3),
                     "disease_preserved": round(pres, 3)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("outputs/sae_ckpts/phase5b_das_earlier.csv", index=False)
    print("wrote phase5b_das_earlier.csv")


if __name__ == "__main__":
    main()
