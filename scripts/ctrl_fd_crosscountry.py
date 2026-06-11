#!/usr/bin/env python
"""Reviewer #1 (the binding threat): a CROSS-COUNTRY framewise-displacement control on the PRIMARY US-vs-China
population/scanner axis -- the place motion is most likely a confound and where we previously had only a proxy.
US FD from PK_MPRC Headmotion.txt; China FD from ChineseSZ SPM realignment (rp_abrat_4D.txt, id SZ-08-0010 ->
dir SZ_08_0010). We test (i) whether FD differs by country and is itself decodable from the population axis
(the confound), and (ii) whether the population DAS-IIA survives residualizing FD out of the decision rep.
If it survives even when FD genuinely differs across countries, the main axis is not a motion artifact.
Writes outputs/sae_ckpts/fd_crosscountry.csv.
"""
import os, sys, re, glob
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mib import das as D
from ctrl_das_null import load_head, make_pairs

PK = "/data/qneuromark/Data/PK_MPRC/ZN_Neuromark/ZN_Prep_fMRI"
CN = "/data/qneuromark/Data/ChineseSZ/Old_data/SZ_fMRI_only/Preprocessed"
MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"


def power_fd(path):
    m = np.loadtxt(path)
    if m.ndim != 2 or m.shape[1] < 6: return np.nan
    dd = np.abs(np.diff(m, axis=0))
    return float((dd[:, :3].sum(1) + 50.0 * dd[:, 3:6].sum(1)).mean())


def residualize(R, p, fitmask):
    pc = p - p[fitmask].mean(); var = (pc[fitmask] ** 2).sum() + 1e-9
    beta = (R[fitmask] * pc[fitmask][:, None]).sum(0) / var
    return R - np.outer(pc, beta)


def main():
    fd = {}
    for hm in glob.glob(f"{PK}/*/Headmotion.txt"):
        sid = re.sub(r"^Scanner[0-9](_[0-9]+)?_", "", os.path.basename(os.path.dirname(hm)))
        v = power_fd(hm)
        if not np.isnan(v): fd[sid] = v
    for rp in glob.glob(f"{CN}/*/rp_abrat*.txt"):
        sid = os.path.basename(os.path.dirname(rp)).replace("_", "-")   # SZ_08_0010 -> SZ-08-0010
        v = power_fd(rp)
        if not np.isnan(v): fd[sid] = v

    d = torch.load("outputs/activations/fused_HOLD_ALL.pt", weights_only=True)
    h = d["fused"].float().numpy(); sid = np.asarray(d["subject_id"]); pop = np.asarray(d["population"])
    split = np.asarray(d["split"])
    fdv = np.array([fd.get(s, np.nan) for s in sid])
    us = (pop == "US") & ~np.isnan(fdv); cn = (pop == "China") & ~np.isnan(fdv)
    have = us | cn
    print(f"FD coverage: US(PK_MPRC) {us.sum()}, China(ChineseSZ) {cn.sum()}, total {have.sum()}", flush=True)
    print(f"mean FD: US {fdv[us].mean():.3f}+-{fdv[us].std():.3f}  China {fdv[cn].mean():.3f}+-{fdv[cn].std():.3f}", flush=True)

    rows = []
    idx = np.where(have)[0]; tr = idx[split[idx] == "train"]; te = idx[split[idx] == "test"]
    ypop = (pop == "China").astype(int)
    # (i) is FD confounded with the country axis? decode population from FD alone
    a = roc_auc_score(ypop[te], LogisticRegression(max_iter=2000).fit(fdv[tr].reshape(-1, 1), ypop[tr])
                      .predict_proba(fdv[te].reshape(-1, 1))[:, 1])
    rows.append({"analysis": "population_from_FD_alone_AUC", "value": round(max(a, 1 - a), 3)})
    # (ii) population DAS-IIA before/after FD residualization (disease preserved)
    hW, hb = load_head(MHOLD)
    lab = torch.tensor(ypop)
    Rres = residualize(h, fdv, have & (split == "train"))
    for tag, Rep in [("DAS_raw", h), ("DAS_FDresidualized", Rres)]:
        hh = torch.tensor(Rep, dtype=torch.float32)
        lr = LogisticRegression(max_iter=3000).fit(Rep[tr], ypop[tr])
        sw = torch.tensor(lr.coef_.ravel(), dtype=torch.float32); sb = torch.tensor(float(lr.intercept_[0]))
        # decodability of population before/after (sanity)
        rows.append({"analysis": tag + "_population_decode_AUC",
                     "value": round(roc_auc_score(ypop[te], lr.predict_proba(Rep[te])[:, 1]), 3)})
        bt, st = make_pairs(lab, torch.tensor(tr), 3000, 0); bv, sv = make_pairs(lab, torch.tensor(te), 3000, 100)
        dd = D.train_das(hh, lab, hW, hb, sw, sb, bt, st, k=1, steps=500, lr=5e-3, lam=1.0, seed=0, device="cpu")
        m = D.eval_iia(dd, hh, lab, hW, hb, sw, sb, bv, sv, device="cpu")
        rows.append({"analysis": tag + "_scannerIIA", "value": round(m["scanner_iia"], 3)})
        rows.append({"analysis": tag + "_disease_pres", "value": round(m["disease_preserved"], 3)})
    for r in rows: print(r, flush=True)
    pd.DataFrame(rows).to_csv("outputs/sae_ckpts/fd_crosscountry.csv", index=False)
    print("\nINTERP: cross-country FD control on the PRIMARY axis. If FD differs by country yet the population "
          "DAS-IIA survives FD-residualization (disease preserved), the main scanner axis is not a motion artifact "
          "even where motion correlates with population -- closing the proxy-only gap.")


if __name__ == "__main__":
    main()
