#!/usr/bin/env python
"""R6 residual (reviewer): the audit so far is on an in-distribution, site-shared 70/30 split. Does the
low-dimensional causal compression survive when the MODEL never saw the scanner? We run a genuine
leave-one-US-cohort-out test on the FNC classifier (cheap to retrain): for each held-out US cohort H,
the MLP is trained on ALL OTHER subjects (H excluded entirely), the scanner probe + DAS are fit only on
the seen cohorts, and the k=1 interchange-IIA is then evaluated on pairs whose US side comes from the
NEVER-TRAINED cohort H (paired against seen China subjects). If IIA on the unseen cohort stays high (and
the random-rotation floor low), the causal axis transfers to scanners the model never trained on.

This is a model-level LOSO at the FNC-network level (the full multimodal ViT LOSO would need a GPU retrain).
Reuses ctrl_crossarch_das.train_with_head + mib.das. Run in the `project` env. Writes outputs/sae_ckpts/loso_iia.csv.
"""
import os, sys, json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mib import das as D
from cross_arch import MLP, H5, SPLIT, fnc_to_matrix
from geomultivit.data.multicohort import load_geometric_cohort
from ctrl_crossarch_das import train_with_head

US_COHORTS = ["COBRE", "FBIRN", "PK_MPRC"]


def build_fnc_cohort():
    si = json.load(open(SPLIT))
    sm = {**{s: "train" for s in si["train_ids"]}, **{s: "test" for s in si["test_ids"]}}
    g = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False).df
    g = g[g.cohort.isin(["COBRE", "FBIRN", "ChineseSZ", "PK_MPRC"])].drop_duplicates("SubjectID")
    g = g[g.SubjectID.isin(set(sm))].reset_index(drop=True)
    iu = np.triu_indices(53, 1)
    mats = np.stack([fnc_to_matrix(r["sFNC"], n_icns=53) for _, r in g.iterrows()]).astype(np.float32)
    X = mats[:, iu[0], iu[1]]
    return (X, g.Diagnosis.to_numpy().astype(int), np.array([sm[s] for s in g.SubjectID]),
            g.population.to_numpy(), g.cohort.to_numpy())


def directed_pairs(base_pool, src_pool, n, seed):
    g = torch.Generator().manual_seed(seed)
    base = base_pool[torch.randint(len(base_pool), (n,), generator=g)]
    src = src_pool[torch.randint(len(src_pool), (n,), generator=g)]
    return base, src


def main():
    X, ydx, split, pop, cohort = build_fnc_cohort()
    yp = (pop == "China").astype(int)                       # scanner axis: US(0) vs China(1)
    from sklearn.preprocessing import StandardScaler
    rows = []
    for H in US_COHORTS:
        seen = cohort != H                                  # H excluded from EVERYTHING upstream
        tr_seen = np.where(seen & (split == "train"))[0]    # train the MLP on seen-train only
        Xs = StandardScaler().fit(X[tr_seen]).transform(X).astype(np.float32)
        net = MLP(X.shape[1])
        R, hW, hb = train_with_head(net, Xs, ydx, tr_seen, seed=0)
        h = torch.tensor(R, dtype=torch.float32)
        # frozen scanner probe fit on SEEN train reps
        lr = LogisticRegression(max_iter=3000).fit(R[tr_seen], yp[tr_seen])
        sw = torch.tensor(lr.coef_.ravel(), dtype=torch.float32); sb = torch.tensor(float(lr.intercept_[0]))
        scanner = torch.tensor(yp)
        # index pools
        china = np.where(seen & (pop == "China"))[0]
        seen_us_te = np.where(seen & (pop == "US") & (split == "test"))[0]
        unseen_us = np.where(cohort == H)[0]                # ALL of held-out cohort (US, never trained)
        chinaT = torch.tensor(china); seenT = torch.tensor(seen_us_te); unseenT = torch.tensor(unseen_us)
        # scanner decodability on the unseen cohort (vs seen China), AUC of the frozen probe
        idx_dec = np.concatenate([unseen_us, china])
        dec_auc = roc_auc_score(yp[idx_dec], (h[torch.tensor(idx_dec)] @ sw + sb).numpy())
        # train DAS on SEEN cross-population pairs (both directions)
        seen_us_tr = np.where(seen & (pop == "US") & (split == "train"))[0]
        bt = torch.cat([torch.tensor(seen_us_tr)[torch.randint(len(seen_us_tr), (3000,))],
                        chinaT[torch.randint(len(china), (3000,))]])
        st = torch.cat([chinaT[torch.randint(len(china), (3000,))],
                        torch.tensor(seen_us_tr)[torch.randint(len(seen_us_tr), (3000,))]])
        das = D.train_das(h, scanner, hW, hb, sw, sb, bt, st, k=1, steps=600, lr=5e-3, lam=1.0, seed=0, device="cpu")
        rnd = D.DASRotation(h.shape[1], k=1, seed=7)
        # eval: in-distribution (seen-US base -> China) vs UNSEEN (held-out-cohort base -> China)
        bI, sI = directed_pairs(seenT, chinaT, 3000, 1)
        bU, sU = directed_pairs(unseenT, chinaT, 3000, 2)
        mI = D.eval_iia(das, h, scanner, hW, hb, sw, sb, bI, sI, device="cpu")
        mU = D.eval_iia(das, h, scanner, hW, hb, sw, sb, bU, sU, device="cpu")
        rU = D.eval_iia(rnd, h, scanner, hW, hb, sw, sb, bU, sU, device="cpu")
        rows.append({"held_out_cohort": H, "n_unseen": len(unseen_us), "dim": R.shape[1],
                     "scanner_dec_auc_unseen": round(dec_auc, 3),
                     "iia_indist": round(mI["scanner_iia"], 3),
                     "iia_unseen": round(mU["scanner_iia"], 3),
                     "iia_unseen_random": round(rU["scanner_iia"], 3),
                     "disease_pres_unseen": round(mU["disease_preserved"], 3)})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv("outputs/sae_ckpts/loso_iia.csv", index=False)
    df = pd.DataFrame(rows)
    print(f"\nmean unseen IIA={df.iia_unseen.mean():.3f}  vs in-dist={df.iia_indist.mean():.3f}  "
          f"vs random floor={df.iia_unseen_random.mean():.3f}")
    print("INTERPRETATION: unseen-cohort IIA >> random floor (with disease preserved) means the 1-D causal axis"
          " transfers to scanners the model never trained on -- the in-distribution audit predicts unseen-site"
          " behaviour, strengthening the trustworthiness claim.")


if __name__ == "__main__":
    main()
