#!/usr/bin/env python
"""A3 head-motion proxy control (reviewer #3). True FD is unavailable (the data carry only precomputed sFNC +
sMRIPath -- no realignment parameters), so a genuine FD-residualized DAS cannot be run. As the best available
substitute we use the canonical FC motion signature: head motion inflates global/short-range connectivity, so
mean |edge| over the static FNC is a per-subject motion proxy. We (1) report its per-site / per-population
distribution and (2) residualize it out of the FNC decision representation and re-run the scanner DAS-IIA. If
the causal axis survives proxy-residualization, the scanner axis is not merely a motion artifact.

FNC-network level (where the raw-connectivity proxy is defined). Reuses build_fnc + train_with_head + mib.das.
Run in `project`. Writes outputs/sae_ckpts/motion_proxy.csv.
"""
import os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mib import das as D
from cross_arch import build_fnc, MLP
from ctrl_crossarch_das import train_with_head, make_pairs


def residualize(R, p, tr):
    """Remove the proxy p (N,) from each column of R (N,d); fit slope/intercept on train rows only."""
    pc = p - p[tr].mean(); var = (pc[tr] ** 2).sum() + 1e-9
    beta = (R[tr] * pc[tr][:, None]).sum(0) / var                 # (d,) per-dim slope on train
    return R - np.outer(pc, beta)


def main():
    mats, X, ydx, split, pop, site = build_fnc()
    tr = split == "train"; keep = np.isin(pop, ["US", "China"]); yp = (pop == "China").astype(int)
    proxy = np.abs(mats).mean(axis=(1, 2)).astype(np.float64)     # mean |FNC edge| = global-FC motion proxy

    # (1) proxy distribution by population + correlation with the scanner label
    rcorr = np.corrcoef(proxy, yp)[0, 1]
    print(f"motion proxy (mean|FNC|): US mean={proxy[pop=='US'].mean():.3f}, "
          f"China mean={proxy[pop=='China'].mean():.3f}, corr(proxy, China)={rcorr:+.3f}", flush=True)
    persite = pd.DataFrame({"site": site, "proxy": proxy}).groupby("site").proxy.agg(["mean", "std", "count"])

    # train the MLP, get penultimate rep
    Xs = StandardScaler().fit(X[tr]).transform(X).astype(np.float32)
    R, hW, hb = train_with_head(MLP(X.shape[1]), Xs, ydx, np.where(tr)[0], seed=0)
    scanner = torch.tensor(yp); trk = np.where(tr & keep)[0]; tek = np.where((~tr) & keep)[0]

    rows = []
    for tag, Rep in [("raw_rep", R), ("proxy_residualized_rep", residualize(R, proxy, tr & keep))]:
        h = torch.tensor(Rep, dtype=torch.float32)
        lr = LogisticRegression(max_iter=3000).fit(Rep[trk], yp[trk])
        sw = torch.tensor(lr.coef_.ravel(), dtype=torch.float32); sb = torch.tensor(float(lr.intercept_[0]))
        sauc = roc_auc_score(yp[tek], (h[torch.tensor(tek)] @ sw + sb).numpy())
        bt, st = make_pairs(scanner, torch.tensor(trk), 3000, 0)
        bv, sv = make_pairs(scanner, torch.tensor(tek), 3000, 100)
        das = D.train_das(h, scanner, hW, hb, sw, sb, bt, st, k=1, steps=600, lr=5e-3, lam=1.0, seed=0, device="cpu")
        m = D.eval_iia(das, h, scanner, hW, hb, sw, sb, bv, sv, device="cpu")
        rnd = D.eval_iia(D.DASRotation(h.shape[1], k=1, seed=7), h, scanner, hW, hb, sw, sb, bv, sv, device="cpu")
        rows.append({"representation": tag, "scanner_probe_auc": round(sauc, 3),
                     "scanner_iia_k1": round(m["scanner_iia"], 3), "disease_preserved": round(m["disease_preserved"], 3),
                     "iia_random_floor": round(rnd["scanner_iia"], 3)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("outputs/sae_ckpts/motion_proxy.csv", index=False)
    print("\nper-site proxy (head):\n", persite.round(3).head(20))
    print("\nINTERPRETATION: if scanner IIA is essentially unchanged after residualizing the motion proxy, the"
          " causal scanner axis is not explained by the global-FC motion signature (best available control; true"
          " FD is unavailable in these data).")


if __name__ == "__main__":
    main()
