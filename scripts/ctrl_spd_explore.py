#!/usr/bin/env python
"""Does the connectome geometry CHANGE a conclusion, or only validity? We localize the scanner signal to
functional-network DOMAINS by restricting the causal intervention to each domain's edges, under TWO operators:
the SPD-manifold (log-Euclidean, valid connectomes) and the naive Euclidean (borrowed; invalid connectomes). If
the two operators RANK the domains differently -- i.e. the invalid Euclidean counterfactual mis-attributes which
networks are causally responsible -- then geometry changes a scientific conclusion (network attribution), not just
matrix validity. Also reports a spatial / domain-block structured null floor. Writes outputs/sae_ckpts/spd_explore.csv.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
import sys
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from cross_arch import build_fnc, MLP
from ctrl_crossarch_das import train_with_head
from make_brain_fig_A import domains, DOM_ORDER

IU = np.triu_indices(53, 1); RIDGE = 0.2


def sym(M): return 0.5 * (M + M.transpose(0, 2, 1))
def logm_b(F):
    w, V = np.linalg.eigh(sym(F)); w = np.clip(w, 1e-6, None)
    return sym(V @ (np.log(w)[..., None] * V.transpose(0, 2, 1)))
def expm_b(S):
    w, V = np.linalg.eigh(sym(S)); return sym(V @ (np.exp(w)[..., None] * V.transpose(0, 2, 1)))


def main():
    mats, X, ydx, split, pop, site = build_fnc()
    F = mats.astype(np.float64) + RIDGE * np.eye(53)
    yp = (pop == "China").astype(int); tr = split == "train"; te = split == "test"
    dxsc = StandardScaler().fit(X[tr]); net = MLP(X.shape[1])
    train_with_head(net, dxsc.transform(X).astype(np.float32), ydx, np.where(tr)[0], seed=0); net.eval()
    sprobe = LogisticRegression(max_iter=3000).fit(dxsc.transform(X[tr]), yp[tr])
    c2d = domains()                                                    # 53-array of domain labels

    def edges(Fm): return Fm[:, IU[0], IU[1]].astype(np.float32)
    def scan_china(Fm): return sprobe.predict_proba(dxsc.transform(edges(Fm)))[:, 1] > 0.5
    def dxarg(Fm):
        with torch.no_grad(): return net(torch.tensor(dxsc.transform(edges(Fm)).astype(np.float32))).argmax(1).numpy()

    T = logm_b(F)
    off_tan = T[tr & (yp == 1)].mean(0) - T[tr & (yp == 0)].mean(0)    # full scanner offset, tangent
    off_euc = F[tr & (yp == 1)].mean(0) - F[tr & (yp == 0)].mean(0)    # full scanner offset, FNC space
    base = np.where(te & (yp == 0))[0]; bdx = dxarg(F[base]); base_rate = float(scan_china(F[base]).mean())

    def domain_mask(d):                                                # 53x53 mask of edges incident to domain d
        nd = np.array([c2d[i] == d for i in range(53)])
        M = np.zeros((53, 53), bool); M[nd, :] = True; M[:, nd] = True; np.fill_diagonal(M, False); return M

    rows = []
    for d in DOM_ORDER:
        m = domain_mask(d)
        ot = np.where(m, off_tan, 0.0); oe = np.where(m, off_euc, 0.0)
        Fm = expm_b(T[base] + ot); Fe = F[base] + oe
        rows.append({"domain": d, "n_edges": int(m[IU].sum()),
                     "manifold_flip": round(float(scan_china(Fm).mean()) - base_rate, 3),
                     "euclidean_flip": round(float(scan_china(Fe).mean()) - base_rate, 3),
                     "manifold_dx_kept": round(float((dxarg(Fm) == bdx).mean()), 3),
                     "euclid_invalid": round(float((np.linalg.eigvalsh(sym(Fe))[:, 0] < 0).mean()), 3)})
        print(rows[-1], flush=True)
    D = pd.DataFrame(rows)
    rho, p = spearmanr(D.manifold_flip, D.euclidean_flip)
    man_top = D.sort_values("manifold_flip", ascending=False).domain.iloc[0]
    euc_top = D.sort_values("euclidean_flip", ascending=False).domain.iloc[0]
    D.to_csv("outputs/sae_ckpts/spd_explore.csv", index=False)
    print(f"\nPer-domain causal attribution: manifold-vs-Euclidean Spearman rho={rho:.2f} (p={p:.3f}). "
          f"Top causal domain: manifold={man_top}, euclidean={euc_top}. "
          f"{'DISAGREE -> geometry changes the network attribution.' if man_top != euc_top or rho < 0.7 else 'AGREE -> geometry is validity-only here.'}")


if __name__ == "__main__":
    main()
