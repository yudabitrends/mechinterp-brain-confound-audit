#!/usr/bin/env python
"""Group-mean FNC contrasts (Vince request): the *interpretable biology* a clinician recognizes, as a
SIGNED hot-cold 53x53 difference -- mean(SZ)-mean(HC) and mean(US)-mean(China) -- to sit beside (and
explain the difference from) the multivariate |L2-coefficient| classifier-handle map currently in Fig 4.

Two data sources, by purpose:
  (1) CANONICAL display: COBRE+FBIRN raw NeuroMark Pearson sFNC (US, diagnosis embedded in analysis_SCORE
      row 2: 1=SZ, 2=HC). Within-cohort mean(SZ)-mean(HC) + Cohen d, averaged. This is the clean,
      reference-recognizable SZ-HC pattern (SM/VI hypoconnectivity, cerebellar-SM hyperconnectivity).
  (2) RECONCILIATION on the model's OWN training FNC (the same 4-cohort 1378-edge vectors the L2 classifier
      was fit on): mean(SZ)-mean(HC) and mean(US)-mean(China), then the spatial correlation of |mean diff|
      with the |L2 coef| map. Low |r| is the quantitative answer to "why doesn't the disease map look like
      the group difference?": the classifier handle (multivariate, sign-free, variance-standardized) is a
      different object from the univariate group effect.

Writes outputs/fnc_groupdiff/{disease_meandiff,disease_cohend,scanner_meandiff,
  disease_meandiff_model,scanner_meandiff_model}.npy + group_diff.json.
"""
import os, sys, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
from pathlib import Path
import numpy as np, h5py

sys.path.insert(0, os.path.dirname(__file__))
from cross_arch import build_fnc

N = 53
IU = np.triu_indices(N, 1)
SF = "/data/qneuromark/Results/SFNC"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "fnc_groupdiff"
OUT.mkdir(parents=True, exist_ok=True)
IMP = Path(__file__).resolve().parent.parent / "outputs" / "fnc_importance"


def load_raw(path, dx_row=2):
    with h5py.File(path, "r") as f:
        s = np.array(f["sFNC"])
        s = np.moveaxis(s, 2, 0) if s.shape[:2] == (N, N) else s
        dx = np.array(f["analysis_SCORE"])[dx_row]
    return s.astype(float), dx.astype(float)


def mean_diff(s, pos, neg):
    return s[pos].mean(0) - s[neg].mean(0)


def cohen_d(s, pos, neg):
    a, b = s[pos], s[neg]
    return (a.mean(0) - b.mean(0)) / (np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-9)


def vec(mat):
    return mat[IU]


def main():
    # ---- (1) canonical SZ-HC from raw COBRE+FBIRN (1=SZ, 2=HC) ----
    cob_s, cob_dx = load_raw(f"{SF}/COBRE/COBRE.mat")
    fb_s, fb_dx = load_raw(f"{SF}/FBIRN/FBIRN.mat")
    print(f"[COBRE] n={len(cob_s)} SZ={int((cob_dx==1).sum())} HC={int((cob_dx==2).sum())}")
    print(f"[FBIRN] n={len(fb_s)} SZ={int((fb_dx==1).sum())} HC={int((fb_dx==2).sum())}")
    diff = (mean_diff(cob_s, cob_dx == 1, cob_dx == 2) + mean_diff(fb_s, fb_dx == 1, fb_dx == 2)) / 2
    d = (cohen_d(cob_s, cob_dx == 1, cob_dx == 2) + cohen_d(fb_s, fb_dx == 1, fb_dx == 2)) / 2
    np.fill_diagonal(diff, 0.0); np.fill_diagonal(d, 0.0)
    print(f"[canonical SZ-HC] max|Δr|={np.abs(diff).max():.3f}  max|d|={np.abs(d).max():.3f}  "
          f"min d={d[IU].min():.3f} (hypo)  max d={d[IU].max():.3f} (hyper)")

    # ---- (2) model's own training data: group means on the SAME edges as the L2 classifier ----
    mats, X, ydx, split, pop, site = build_fnc()
    # determine SZ label by aligning with canonical: SZ cohorts (US SZ subjects) should mean-diff match sign
    vals = np.unique(ydx)
    # heuristic: SZ is the label whose group mean-diff correlates positively with canonical diff
    cand = {}
    for sz_lab in vals:
        md = mean_diff(mats, ydx == sz_lab, ydx != sz_lab)
        cand[sz_lab] = float(np.corrcoef(vec(md), vec(diff))[0, 1])
    sz_label = max(cand, key=cand.get)
    print(f"[model dx] labels={vals.tolist()} corr-with-canonical={ {int(k):round(v,3) for k,v in cand.items()} } "
          f"-> SZ={int(sz_label)}")
    dmd = mean_diff(mats, ydx == sz_label, ydx != sz_label)        # SZ - HC on model data
    np.fill_diagonal(dmd, 0.0)
    keep = np.isin(pop, ["US", "China"])
    smd_full = mean_diff(mats[keep], (pop[keep] == "China"), (pop[keep] == "US"))  # China - US
    np.fill_diagonal(smd_full, 0.0)
    # scanner canonical-ish from raw (FBIRN sites pooled vs COBRE) is not US/China; keep model US-China
    np.fill_diagonal(smd_full, 0.0)

    # ---- reconciliation: |group mean diff| vs |L2 classifier coef| (the current Fig-4 disease map) ----
    dcoef = np.load(IMP / "disease_coef_abs.npy")            # (1378,) |L2 coef|, standardized edges
    scoef = np.load(IMP / "scanner_coef_abs.npy")
    r_disease = float(np.corrcoef(np.abs(vec(dmd)), dcoef)[0, 1])
    r_disease_signed = float(np.corrcoef(vec(dmd), dcoef)[0, 1])
    r_scanner = float(np.corrcoef(np.abs(vec(smd_full)), scoef)[0, 1])
    # also: canonical (raw) vs model mean-diff agreement (sanity that model data carries the same biology)
    r_canon_model = float(np.corrcoef(vec(diff), vec(dmd))[0, 1])
    print(f"[reconcile] corr(|SZ-HC mean diff|, |L2 disease coef|) = {r_disease:.3f}  (signed {r_disease_signed:.3f})")
    print(f"[reconcile] corr(|US-China mean diff|, |L2 scanner coef|) = {r_scanner:.3f}")
    print(f"[sanity]    corr(canonical raw SZ-HC, model SZ-HC) = {r_canon_model:.3f}")
    print(f"[scale]     disease max|Δr|={np.abs(dmd).max():.3f}  scanner max|Δr|={np.abs(smd_full).max():.3f}  "
          f"-> scanner/disease mean-diff magnitude ratio = "
          f"{np.abs(vec(smd_full)).mean()/ (np.abs(vec(dmd)).mean()+1e-12):.2f}x")

    np.save(OUT / "disease_meandiff.npy", diff.astype(np.float32))        # canonical, for display
    np.save(OUT / "disease_cohend.npy", d.astype(np.float32))
    np.save(OUT / "disease_meandiff_model.npy", dmd.astype(np.float32))   # model-data, for reconciliation
    np.save(OUT / "scanner_meandiff_model.npy", smd_full.astype(np.float32))
    (OUT / "group_diff.json").write_text(json.dumps({
        "canonical_disease_source": "COBRE+FBIRN raw NeuroMark Pearson sFNC, within-cohort mean(SZ)-mean(HC) averaged",
        "n_canonical": int(len(cob_s) + len(fb_s)),
        "sz_hc_max_abs_meandiff": float(np.abs(diff).max()),
        "sz_hc_max_abs_cohend": float(np.abs(d).max()),
        "model_data_source": "szdataset_modified.h5 4-cohort, same 1378 edges as the L2 classifier",
        "sz_label_in_model": int(sz_label),
        "corr_absMeanDiff_vs_absL2coef_disease": r_disease,
        "corr_signedMeanDiff_vs_L2coef_disease": r_disease_signed,
        "corr_absMeanDiff_vs_absL2coef_scanner": r_scanner,
        "corr_canonical_vs_modelMeanDiff_disease": r_canon_model,
        "scanner_over_disease_meandiff_magnitude_ratio":
            float(np.abs(vec(smd_full)).mean() / (np.abs(vec(dmd)).mean() + 1e-12)),
    }, indent=2))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
