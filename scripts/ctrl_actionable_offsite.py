#!/usr/bin/env python
"""Major 1 (actionable significance): the audit changes WHEN to stop harmonizing, with a MEASURABLE downstream
benefit. Two acceptance prescriptions for a successively-applied linear eraser (iterated rank-1 LEACE of site):
  - decodability-accepted: STOP when residual site DECODABILITY <= chance+0.03 (the standard practice);
  - causal-accepted: continue until residual behavioral exposure |bias| <= 0.03 (the audit's criterion).
Endpoint (practitioner-relevant, no clinical-outcome claim): the cross-site disease-output disparity among
MATCHED TRUE CONTROLS -- the model giving systematically different disease probabilities to same-diagnosis
subjects by site (a reliability/equity harm). We report, per erasure rank, residual decodability, the
matched-control disparity, and disease AUC, then read off where each prescription stops and the disparity it
leaves. Run on ABIDE (real under-alarm). Writes outputs/sae_ckpts/actionable_offsite.csv.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
import sys
import numpy as np, pandas as pd, scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mib.abide_data import build_abide_manifest, A1, A2
from cross_arch import MLP
from ctrl_crossarch_das import train_with_head
from ctrl_entangled_full import psz, leace1, bias_HC
from ctrl_natural_abide import HI, LO


def main():
    df = build_abide_manifest()
    fnc = {"ABIDE_I": np.asarray(sio.loadmat(A1["mat"])["sFNC"]),
           "ABIDE_II": np.asarray(sio.loadmat(A2["mat"])["sFNC"])}
    iu = np.triu_indices(53, 1)
    X = np.stack([fnc[r.cohort][r.sfnc_idx][iu] for _, r in df.iterrows()]).astype(np.float32)
    site = df.site.to_numpy(); dx = df.Diagnosis.to_numpy().astype(int)
    grp = np.where(np.isin(site, HI), 1, np.where(np.isin(site, LO), 0, -1)); m = grp >= 0
    Xs = StandardScaler().fit(X[m]).transform(X).astype(np.float32)
    rng = np.random.RandomState(0); tr = m & (rng.rand(len(df)) < 0.7); te = m & ~tr
    tri, tei = np.where(tr)[0], np.where(te)[0]
    R, hW, hb = train_with_head(MLP(X.shape[1]), Xs, dx, tri, seed=0)        # natural-prevalence model
    zg = grp
    hiC = tei[(dx[tei] == 0) & (zg[tei] == 1)]; loC = tei[(dx[tei] == 0) & (zg[tei] == 0)]

    def decode(Rk):
        a = roc_auc_score(zg[tei], LogisticRegression(max_iter=2000).fit(Rk[tri], zg[tri]).predict_proba(Rk[tei])[:, 1])
        return max(a, 1 - a)

    rows = []
    Rk = R.copy()
    for k in range(0, 7):
        if k > 0:
            Rk = leace1(Rk, zg, tri)                            # remove one more site direction (fit on train)
        dec = decode(Rk); disp = abs(bias_HC(Rk, hW, hb, hiC, loC))
        dxauc = roc_auc_score(dx[tei], psz(Rk[tei], hW, hb)); dxauc = max(dxauc, 1 - dxauc)
        rows.append({"erase_rank": k, "site_decode_auc": round(dec, 3),
                     "matched_control_disparity": round(disp, 3), "disease_auc": round(dxauc, 3)})
        print(rows[-1], flush=True)
    D = pd.DataFrame(rows)
    dec_stop = int(D[D.site_decode_auc <= 0.55].erase_rank.min()) if (D.site_decode_auc <= 0.55).any() else None
    cau_stop = int(D[D.matched_control_disparity <= 0.03].erase_rank.min()) if (D.matched_control_disparity <= 0.03).any() else None
    D.to_csv("outputs/sae_ckpts/actionable_offsite.csv", index=False)
    print(f"\nDecodability-accepted STOP at rank {dec_stop}: disparity "
          f"{D.loc[D.erase_rank==dec_stop,'matched_control_disparity'].values if dec_stop is not None else 'NA'}, "
          f"disease AUC {D.loc[D.erase_rank==dec_stop,'disease_auc'].values if dec_stop is not None else 'NA'}.")
    print(f"Causal-exposure-accepted STOP at rank {cau_stop}: disparity "
          f"{D.loc[D.erase_rank==cau_stop,'matched_control_disparity'].values if cau_stop is not None else 'NA'}, "
          f"disease AUC {D.loc[D.erase_rank==cau_stop,'disease_auc'].values if cau_stop is not None else 'NA'}.")
    print("MEASURABLE BENEFIT = matched-control cross-site disparity the decodability prescription LEAVES but the "
          "causal prescription REMOVES, at preserved disease AUC. If cau_stop==dec_stop or disease collapses, report honestly.")


if __name__ == "__main__":
    main()
