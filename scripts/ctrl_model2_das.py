#!/usr/bin/env python
"""#4 follow-up — DAS/IIA on a SECOND, independently-trained multimodal model (seed 123, its own
split). Closes single-model at the MULTIMODAL level: does the k=1 causal compression replicate?

Loads model2, runs its own held-out split, fits a frozen scanner readout on its train fused, and runs
the identical k=1 DAS interchange test on its fused decision rep. GPU forward + CPU DAS.
Writes outputs/sae_ckpts/model2_das.csv.
"""
import json, os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from geomultivit.data.dataset import MultiModalH5Dataset
from geomultivit.data.multicohort import load_geometric_cohort
from mib.extract import load_model
from mib import patch, das as D

H5 = "/home/users/ybi3/data/szdataset_modified.h5"
M2 = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split_seed123/best.pt"
SPLIT2 = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split_seed123/split_info.json"


def loader_for(gdf, ids, bs=16):
    sub = gdf[gdf.SubjectID.isin(set(ids))].drop_duplicates("SubjectID").reset_index(drop=True)
    return DataLoader(MultiModalH5Dataset(sub, (96, 112, 96), 53, require_sMRI=True, cache_dir="outputs/cache"),
                      batch_size=bs, shuffle=False, num_workers=4), sub


def make_pairs(label, idx, n, seed):
    g = torch.Generator().manual_seed(seed)
    a = idx[label[idx] == 0]; b = idx[label[idx] == 1]
    base = torch.cat([a[torch.randint(len(a), (n,), generator=g)], b[torch.randint(len(b), (n,), generator=g)]])
    src = torch.cat([b[torch.randint(len(b), (n,), generator=g)], a[torch.randint(len(a), (n,), generator=g)]])
    return base, src


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print(f"device={dev}", flush=True)
    si = json.load(open(SPLIT2))
    gdf = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False).df
    gdf = gdf[gdf.cohort.isin(["COBRE", "FBIRN", "ChineseSZ", "PK_MPRC"])]
    model = load_model(M2, device=dev)
    hW = model.head.weight.detach().float().cpu(); hb = model.head.bias.detach().float().cpu()
    tr_loader, tr_sub = loader_for(gdf, si["train_ids"]); te_loader, te_sub = loader_for(gdf, si["test_ids"])

    def pop_of(sub, sids):
        m = sub.drop_duplicates("SubjectID").set_index("SubjectID")
        return np.array([m.loc[s, "population"] for s in sids])

    base_tr = patch.run_model(model, tr_loader, dev); base_te = patch.run_model(model, te_loader, dev)
    Ftr = base_tr["fused"].float(); Fte = base_te["fused"].float()
    pop_tr = pop_of(tr_sub, base_tr["subject_id"]); pop_te = pop_of(te_sub, base_te["subject_id"])
    yscan_tr = (pop_tr == "China").astype(int); yscan_te = (pop_te == "China").astype(int)
    ydx_te = base_te["y"].astype(int)
    keep_tr = np.isin(pop_tr, ["US", "China"]); keep_te = np.isin(pop_te, ["US", "China"])
    # sanity: head(fused) reproduces logits
    err = ((Fte @ hW.T + hb) - base_te["logits"].float()).abs().max().item()
    print(f"head(fused) vs logits max|Δ| = {err:.2e}; model2 held-out disease AUC = "
          f"{roc_auc_score(ydx_te, (base_te['logits'][:,1]-base_te['logits'][:,0]).numpy()):.3f}", flush=True)

    h = torch.cat([Ftr, Fte], 0)               # combined, index train then test
    ntr = len(Ftr)
    scanner = torch.tensor(np.concatenate([yscan_tr, yscan_te]))
    keep = np.concatenate([keep_tr, keep_te])
    tri = torch.tensor(np.where(np.arange(len(h)) < ntr)[0][keep_tr])
    tei = torch.tensor((np.where(np.arange(len(h)) >= ntr)[0])[keep_te])
    lr = LogisticRegression(max_iter=3000).fit(h[tri].numpy(), scanner[tri].numpy())
    sw = torch.tensor(lr.coef_.ravel(), dtype=torch.float32); sb = torch.tensor(float(lr.intercept_[0]))
    probe_auc = roc_auc_score(scanner[tei].numpy(), (h[tei] @ sw + sb).numpy())
    bt, st = make_pairs(scanner, tri, 4000, 0); bv, sv = make_pairs(scanner, tei, 4000, 100)
    das = D.train_das(h, scanner, hW, hb, sw, sb, bt, st, k=1, steps=800, lr=5e-3, lam=1.0, seed=0, device="cpu")
    m = D.eval_iia(das, h, scanner, hW, hb, sw, sb, bv, sv, device="cpu")
    rnd = D.DASRotation(h.shape[1], k=1, seed=7); rm = D.eval_iia(rnd, h, scanner, hW, hb, sw, sb, bv, sv, device="cpu")
    u = das._W()[0].detach().numpy(); cos = abs(float(u @ lr.coef_.ravel() / (np.linalg.norm(u) * np.linalg.norm(lr.coef_.ravel()) + 1e-12)))
    row = {"model": "multimodal_seed123", "scanner_probe_auc": round(float(probe_auc), 3),
           "scanner_iia_k1": round(m["scanner_iia"], 3), "disease_preserved": round(m["disease_preserved"], 3),
           "scanner_iia_random": round(rm["scanner_iia"], 3), "cos_das_probe": round(cos, 3)}
    print(row, flush=True)
    pd.DataFrame([row]).to_csv("outputs/sae_ckpts/model2_das.csv", index=False)
    print("wrote outputs/sae_ckpts/model2_das.csv")


if __name__ == "__main__":
    main()
