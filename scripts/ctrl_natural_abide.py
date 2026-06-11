#!/usr/bin/env python
"""Round-4 point 2, the UNDER-alarm direction on REAL data. ABIDE (autism) is the right testbed: the true signal
is weak (autism-from-FNC ~0.67), so a model is tempted to lean on site, and site x diagnosis is naturally
imbalanced by recruitment (high-autism sites USM 63%, SDSU_1 59%, NYU_1 57%, GU 52% vs low ETH 27%, KKI 35%,
UM_2 36%, SDSU 36%). We train on the NATURAL (un-rebalanced) data and test whether a harmonizer that drives
residual site DECODABILITY to chance (LEACE) still leaves the model BEHAVIORALLY site-exposed (over-calling
high-autism-site controls as autistic) -- the real-data 'LEACE certifies harmonized but behavior is not' case.
Writes outputs/sae_ckpts/natural_abide.csv.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
import sys
import numpy as np, pandas as pd, torch, scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mib.abide_data import build_abide_manifest, A1, A2
from cross_arch import MLP
from ctrl_crossarch_das import train_with_head
from ctrl_entangled_full import psz, leace1, bias_HC

HI = ["A2_ABIDEII-USM_1", "A2_ABIDEII-SDSU_1", "A2_ABIDEII-NYU_1", "A2_ABIDEII-GU_1", "A1_USM"]   # high autism %
LO = ["A2_ABIDEII-ETH_1", "A1_KKI", "A1_UM_2", "A1_SDSU", "A2_ABIDEII-OHSU_1"]                     # low autism %


def main():
    df = build_abide_manifest()
    fnc = {"ABIDE_I": np.asarray(sio.loadmat(A1["mat"])["sFNC"]),
           "ABIDE_II": np.asarray(sio.loadmat(A2["mat"])["sFNC"])}
    iu = np.triu_indices(53, 1)
    X = np.stack([fnc[r.cohort][r.sfnc_idx][iu] for _, r in df.iterrows()]).astype(np.float32)
    site = df.site.to_numpy(); dx = df.Diagnosis.to_numpy().astype(int)
    grp = np.where(np.isin(site, HI), 1, np.where(np.isin(site, LO), 0, -1))
    m = grp >= 0
    nat_auc = roc_auc_score(dx[m], grp[m]); nat_auc = max(nat_auc, 1 - nat_auc)
    sig_auc = "weak (~0.6-0.7, see paper)"
    print(f"ABIDE natural contrast: HI n={int((grp==1).sum())} ({100*dx[grp==1].mean():.0f}% aut), "
          f"LO n={int((grp==0).sum())} ({100*dx[grp==0].mean():.0f}% aut); site->dx AUC {nat_auc:.2f}", flush=True)

    Xs = StandardScaler().fit(X[m]).transform(X).astype(np.float32)
    rng = np.random.RandomState(0); tr = m & (rng.rand(len(df)) < 0.7); te = m & ~tr
    tri, tei = np.where(tr)[0], np.where(te)[0]
    R, hW, hb = train_with_head(MLP(X.shape[1]), Xs, dx, tri, seed=0)        # trained on NATURAL prevalences
    dx_auc = roc_auc_score(dx[tei], psz(R[tei], hW, hb)); dx_auc = max(dx_auc, 1 - dx_auc)
    zg = grp
    dec0 = roc_auc_score(zg[tei], LogisticRegression(max_iter=2000).fit(R[tri], zg[tri]).predict_proba(R[tei])[:, 1]); dec0 = max(dec0, 1 - dec0)
    Rl = leace1(R, zg, tri)
    decL = roc_auc_score(zg[tei], LogisticRegression(max_iter=2000).fit(Rl[tri], zg[tri]).predict_proba(Rl[tei])[:, 1]); decL = max(decL, 1 - decL)
    hiC = tei[(dx[tei] == 0) & (zg[tei] == 1)]; loC = tei[(dx[tei] == 0) & (zg[tei] == 0)]
    b_raw = bias_HC(R, hW, hb, hiC, loC); b_leace = bias_HC(Rl, hW, hb, hiC, loC)
    row = {"dataset": "ABIDE", "contrast": "high-autism vs low-autism sites", "disease_auc": round(dx_auc, 3),
           "natural_site_to_dx_auc": round(nat_auc, 3), "n_train": int(tr.sum()),
           "site_decode_raw": round(dec0, 3), "site_decode_after_LEACE": round(decL, 3),
           "behav_bias_raw": round(b_raw, 3), "behav_bias_after_LEACE": round(b_leace, 3)}
    print(row, flush=True)
    print(f"  -> REAL data: site decodable {dec0:.2f}->{decL:.2f} after LEACE (~chance), behavioral exposure "
          f"{b_raw:+.3f}->{b_leace:+.3f}. If behavior stays exposed after decodability->chance, this is the "
          f"real-data UNDER-alarm: LEACE certifies 'harmonized' but the model still uses site.", flush=True)
    pd.DataFrame([row]).to_csv("outputs/sae_ckpts/natural_abide.csv", index=False)


if __name__ == "__main__":
    main()
