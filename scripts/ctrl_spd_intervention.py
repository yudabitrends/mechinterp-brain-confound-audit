#!/usr/bin/env python
"""BRAIN-NATIVE METHOD (not a borrowed Euclidean tool): a causal scanner intervention that respects the SPD
(Riemannian) geometry of the functional connectome. A connectome is symmetric positive-definite on a curved
manifold; the standard Euclidean interchange/activation-patching operator (DAS) produces COUNTERFACTUALS THAT ARE
NOT VALID CONNECTOMES (indefinite matrices). We realise the scanner counterfactual as a log-Euclidean tangent-
space shift toward the opposite-scanner mean, so every counterfactual connectome is guaranteed SPD, then read the
causal effect off the classifier (scanner readout flips to source, disease decision preserved) = a manifold-
native interchange-intervention accuracy.

Reports: (i) manifold scanner-flip + disease preservation, 0% invalid; (ii) the Euclidean operator's PSD-validity
failure rate (% counterfactual connectomes with a negative eigenvalue) + its flip/disease; (iii) a BRAIN-
STRUCTURED null (scanner offset with the connectome's domain-block structure destroyed) as the manifold floor,
replacing the paper's isotropic random-rotation floor. Writes outputs/sae_ckpts/spd_intervention.csv.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
import sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from cross_arch import build_fnc, MLP
from ctrl_crossarch_das import train_with_head

IU = np.triu_indices(53, 1)
RIDGE = 0.2


def sym(M):
    return 0.5 * (M + M.transpose(0, 2, 1))


def logm_b(F):
    w, V = np.linalg.eigh(sym(F)); w = np.clip(w, 1e-6, None)
    return sym(V @ (np.log(w)[..., None] * V.transpose(0, 2, 1)))


def expm_b(S):
    w, V = np.linalg.eigh(sym(S))
    return sym(V @ (np.exp(w)[..., None] * V.transpose(0, 2, 1)))


def main():
    mats, X, ydx, split, pop, site = build_fnc()
    F = mats.astype(np.float64) + RIDGE * np.eye(53)
    yp = (pop == "China").astype(int); tr = split == "train"; te = split == "test"
    dxsc = StandardScaler().fit(X[tr])
    net = MLP(X.shape[1])
    train_with_head(net, dxsc.transform(X).astype(np.float32), ydx, np.where(tr)[0], seed=0)
    net.eval()
    sprobe = LogisticRegression(max_iter=3000).fit(dxsc.transform(X[tr]), yp[tr])

    def edges(Fm):
        return Fm[:, IU[0], IU[1]].astype(np.float32)

    def scan_china(Fm):
        return sprobe.predict_proba(dxsc.transform(edges(Fm)))[:, 1] > 0.5

    def dx_argmax(Fm):
        with torch.no_grad():
            return net(torch.tensor(dxsc.transform(edges(Fm)).astype(np.float32))).argmax(1).numpy()

    T = logm_b(F)
    off = T[tr & (yp == 1)].mean(0) - T[tr & (yp == 0)].mean(0)           # scanner offset, tangent (log) space
    base = np.where(te & (yp == 0))[0]                                    # US base subjects -> move to China
    F_man = expm_b(T[base] + off)                                         # guaranteed SPD
    F_euc = F[base] + (F[tr & (yp == 1)].mean(0) - F[tr & (yp == 0)].mean(0))

    def invalid(Fm):
        return float((np.linalg.eigvalsh(sym(Fm))[:, 0] < 0).mean())

    base_dx = dx_argmax(F[base])
    res = {}
    for tag, Fm in [("manifold_logeuclidean", F_man), ("euclidean_naive", F_euc)]:
        res[tag] = dict(scanner_flip=round(float(scan_china(Fm).mean()), 3),
                        disease_preserved=round(float((dx_argmax(Fm) == base_dx).mean()), 3),
                        pct_invalid_connectome=round(invalid(Fm), 3))
    base_rate = round(float(scan_china(F[base]).mean()), 3)

    # brain-structured null: destroy the offset's domain-block structure (permute upper-tri entries), keep SPD path
    rng = np.random.RandomState(0); floors = []
    for _ in range(20):
        o = off.copy(); p = rng.permutation(len(IU[0]))
        o[IU[0], IU[1]] = off[IU[0], IU[1]][p]; o[IU[1], IU[0]] = o[IU[0], IU[1]]
        floors.append(float(scan_china(expm_b(T[base] + o)).mean()))
    floor = round(float(np.mean(floors)), 3)

    rows = [{"operator": "manifold_logeuclidean", **res["manifold_logeuclidean"]},
            {"operator": "euclidean_naive", **res["euclidean_naive"]},
            {"operator": "base_rate_no_intervention", "scanner_flip": base_rate, "disease_preserved": 1.0, "pct_invalid_connectome": 0.0},
            {"operator": "brain_structured_null_blockpermuted", "scanner_flip": floor, "disease_preserved": np.nan, "pct_invalid_connectome": 0.0}]
    for r in rows: print(r, flush=True)
    pd.DataFrame(rows).to_csv("outputs/sae_ckpts/spd_intervention.csv", index=False)
    print(f"\nMANIFOLD intervention flips scanner to source on {res['manifold_logeuclidean']['scanner_flip']:.0%} of "
          f"held-out subjects, disease preserved {res['manifold_logeuclidean']['disease_preserved']:.0%}, with "
          f"{res['manifold_logeuclidean']['pct_invalid_connectome']:.0%} invalid connectomes. The EUCLIDEAN "
          f"(borrowed-DAS) operator produces {res['euclidean_naive']['pct_invalid_connectome']:.0%} geometrically "
          f"INVALID (non-PSD) counterfactuals. Brain-structured null floor {floor} vs base rate {base_rate}.")


if __name__ == "__main__":
    main()
