#!/usr/bin/env python
"""Tier-1 / reviewer #3: a REAL framewise-displacement (FD) control, not the mean-|edge| proxy. PK_MPRC carries
per-subject realignment parameters (Headmotion.txt: 6 rigid-body params x ~135 frames); all 306 of this paper's
PK_MPRC subjects map. We compute Power-2012 mean FD, ask whether FD is confounded with the acquisition (scanner)
axis, and then residualize FD out of the fused decision representation and re-measure the within-PK_MPRC scanner
signal (multiclass site decodability + a binary-scanner DAS-IIA). If the scanner axis survives FD-residualization,
it is not explained by head motion. Cached fused_HOLD_ALL.pt + frozen head. Writes outputs/sae_ckpts/fd_residualize.csv.
"""
import os, sys, re, glob
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mib import das as D
from ctrl_das_null import load_head, make_pairs

HMDIR = "/data/qneuromark/Data/PK_MPRC/ZN_Neuromark/ZN_Prep_fMRI"
MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"


def mean_fd(path):
    m = np.loadtxt(path)                       # (T,6): mm x3, rad x3
    if m.ndim != 2 or m.shape[1] < 6: return np.nan
    d = np.abs(np.diff(m, axis=0))             # frame-to-frame
    return float((d[:, :3].sum(1) + 50.0 * d[:, 3:6].sum(1)).mean())   # Power 2012, r=50mm


def residualize(R, p, tr):
    pc = p - p[tr].mean(); var = (pc[tr] ** 2).sum() + 1e-9
    beta = (R[tr] * pc[tr][:, None]).sum(0) / var
    return R - np.outer(pc, beta)


def main():
    # FD per subject id
    fd = {}
    for hm in glob.glob(f"{HMDIR}/*/Headmotion.txt"):
        sid = re.sub(r"^Scanner[0-9](_[0-9]+)?_", "", os.path.basename(os.path.dirname(hm)))
        v = mean_fd(hm)
        if not np.isnan(v): fd[sid] = v

    d = torch.load("outputs/activations/fused_HOLD_ALL.pt", weights_only=True)
    h = d["fused"].float(); sid = np.asarray(d["subject_id"]); site = np.asarray(d["site"])
    split = np.asarray(d["split"]); y = np.asarray(d["y_dx"])
    pk = np.isin(site, ["Scanner1", "Scanner2", "Scanner3"])
    fdv = np.array([fd.get(s, np.nan) for s in sid])
    have = pk & ~np.isnan(fdv)
    print(f"PK_MPRC fused={pk.sum()} | FD mapped={have.sum()}", flush=True)
    # FD by scanner (is motion confounded with the acquisition axis?)
    for sc in ["Scanner1", "Scanner2", "Scanner3"]:
        m = have & (site == sc)
        print(f"  {sc}: n={m.sum()} meanFD={fdv[m].mean():.3f}±{fdv[m].std():.3f}", flush=True)

    Rraw = h.numpy(); hW, hb = load_head(MHOLD)
    rows = []
    # (1) multiclass within-PK_MPRC site decodability, before vs after FD-residualization (held-out)
    idx = np.where(have)[0]; tr = idx[split[idx] == "train"]; te = idx[split[idx] == "test"]
    ytr, yte = site[tr], site[te]
    def site_macro_auc(Rep):
        lr = LogisticRegression(max_iter=4000, multi_class="ovr").fit(Rep[tr], ytr)
        P = lr.predict_proba(Rep[te])
        return roc_auc_score(pd.get_dummies(yte).values, P, average="macro")
    Rres = residualize(Rraw, fdv, have & (split == "train"))   # fit residualization on TRAIN rows only (no leakage)
    rows.append({"analysis": "site_macroAUC_raw", "value": round(site_macro_auc(Rraw), 3)})
    rows.append({"analysis": "site_macroAUC_FDresidualized", "value": round(site_macro_auc(Rres), 3)})
    # FD vs scanner: multiclass decodability of scanner FROM FD alone (how confounded)
    lrfd = LogisticRegression(max_iter=2000, multi_class="ovr").fit(fdv[tr].reshape(-1, 1), ytr)
    rows.append({"analysis": "scanner_from_FD_alone_macroAUC",
                 "value": round(roc_auc_score(pd.get_dummies(yte).values,
                                               lrfd.predict_proba(fdv[te].reshape(-1, 1)), average="macro"), 3)})

    # (2) binary-scanner DAS-IIA (Scanner2 vs Scanner3, the two largest), before vs after FD-residualization
    bmask = have & np.isin(site, ["Scanner2", "Scanner3"])
    lab = torch.tensor((site == "Scanner3").astype(int))
    bi = np.where(bmask)[0]; btr = bi[split[bi] == "train"]; bte = bi[split[bi] == "test"]
    for tag, Rep in [("DAS_raw", Rraw), ("DAS_FDresidualized", Rres)]:
        hh = torch.tensor(Rep, dtype=torch.float32)
        lr = LogisticRegression(max_iter=3000).fit(Rep[btr], lab.numpy()[btr])
        sw = torch.tensor(lr.coef_.ravel(), dtype=torch.float32); sb = torch.tensor(float(lr.intercept_[0]))
        bt, st = make_pairs(lab, torch.tensor(btr), 2000, 0); bv, sv = make_pairs(lab, torch.tensor(bte), 2000, 100)
        das = D.train_das(hh, lab, hW, hb, sw, sb, bt, st, k=1, steps=500, lr=5e-3, lam=1.0, seed=0, device="cpu")
        m = D.eval_iia(das, hh, lab, hW, hb, sw, sb, bv, sv, device="cpu")
        rows.append({"analysis": tag + "_scannerIIA", "value": round(m["scanner_iia"], 3)})
        rows.append({"analysis": tag + "_disease_pres", "value": round(m["disease_preserved"], 3)})
    for r in rows: print(r, flush=True)
    pd.DataFrame(rows).to_csv("outputs/sae_ckpts/fd_residualize.csv", index=False)
    print("\nINTERP: if FD differs little across scanners and site decodability + DAS-IIA survive FD-residualization,"
          " the acquisition axis is not a motion artifact -- now with a REAL FD control (all 306 PK_MPRC mapped).")


if __name__ == "__main__":
    main()
