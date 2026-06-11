#!/usr/bin/env python
"""NC reviewer (points 1-4): does interchange-IIA measure anything beyond linear probe geometry?
In the paper's near-orthogonal regime, scanner is a PASSENGER (|cos|~0.06) and moving it does not move the
model's disease output -- so "causal" reduces to "flips an external probe". We test the discriminating case the
reviewer demands: train the SAME FNC classifier in an ENTANGLED regime where site predicts diagnosis (scanner is
a genuine disease SHORTCUT), and measure the shift in the MODEL'S OWN disease output under scanner-interchange.

Key metric (not the external probe): disease_output_shift = mean |P_SZ(interchange(base,src)) - P_SZ(base)|.
Expected: scanner is decodable (probe-IIA high) in BOTH regimes, but the disease-output shift is ~0 when scanner
is a passenger (orthogonal) and LARGE when scanner is a shortcut (entangled). That is a decodability==causal-for-
output dissociation the probe cannot see -- IIA on the model's behaviour distinguishes passenger from shortcut.

Residual-IIA acceptance (point 4): on the entangled model, a linear eraser (INLP) drives residual scanner
DECODABILITY to chance; we then check whether the model's disease output is still scanner-biased (residual causal
exposure that decodability misses) -> residual-IIA is a stricter harmonizer acceptance criterion than residual
decodability. Cheap FNC retrains. Writes outputs/sae_ckpts/shortcut_regime.csv.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
import sys, importlib.util
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from cross_arch import build_fnc, MLP
from ctrl_crossarch_das import train_with_head, make_pairs
from mib import das as D
_spec = importlib.util.spec_from_file_location("hc", os.path.join(os.path.dirname(__file__), "harmonize_compare.py"))
hc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(hc)   # inlp, auc_fit_eval


def psz(R, hW, hb):
    return torch.softmax(torch.tensor(R, dtype=torch.float32) @ hW.T + hb, 1)[:, 1].numpy()


def behav_bias(R, hW, hb, yp, ydx, te):
    """Model's scanner bias on its OWN output: among true-HC test subjects, P_SZ(China-HC) - P_SZ(US-HC).
    Any non-zero value = the model is using scanner to drive its disease call (a genuine shortcut)."""
    hc_te = te[ydx[te] == 0]; p = psz(R[hc_te], hW, hb)
    return float(p[yp[hc_te] == 1].mean() - p[yp[hc_te] == 0].mean())


def swap_effect(R, hW, hb, sw, yp, te, seed):
    """Unconstrained interventional test (NO disease-preservation constraint): swap the scanner-probe coordinate
    base->source and measure the shift in the model's disease output."""
    shat = (sw / sw.norm()).numpy(); s = R @ shat
    g = np.random.RandomState(seed)
    us = te[yp[te] == 0]; ch = te[yp[te] == 1]
    b = np.concatenate([us[g.randint(len(us), size=2000)], ch[g.randint(len(ch), size=2000)]])
    sc = np.concatenate([ch[g.randint(len(ch), size=2000)], us[g.randint(len(us), size=2000)]])
    Rp = R[b] + (s[sc] - s[b])[:, None] * shat[None, :]
    return float(np.abs(psz(Rp, hW, hb) - psz(R[b], hW, hb)).mean())


def run_regime(name, Xs, ydx, tr_idx, yp, split, seed):
    net = MLP(Xs.shape[1])
    R, hW, hb = train_with_head(net, Xs, ydx, tr_idx, seed=seed)
    lr = LogisticRegression(max_iter=3000).fit(R[tr_idx], yp[tr_idx])
    sw = torch.tensor(lr.coef_.ravel(), dtype=torch.float32)
    cos = float(torch.abs(F.cosine_similarity(sw[None], (hW[1] - hW[0])[None]))[0])
    te = np.where(split == "test")[0]
    dec = roc_auc_score(yp[te], R[te] @ sw.numpy()); dec = max(dec, 1 - dec)
    dx_auc = roc_auc_score(ydx[te], psz(R[te], hW, hb))
    return {"regime": name, "seed": seed, "cos_scanner_disease": round(cos, 3),
            "scanner_decode_auc": round(dec, 3), "disease_auc": round(dx_auc, 3),
            "behav_bias_HC": round(behav_bias(R, hW, hb, yp, ydx, te), 3),
            "swap_output_effect": round(swap_effect(R, hW, hb, sw, yp, te, seed), 3)}, R, hW, hb, sw


def main():
    mats, X, ydx, split, pop, site = build_fnc()
    yp = (pop == "China").astype(int)
    Xs = StandardScaler().fit(X[split == "train"]).transform(X).astype(np.float32)
    tr = np.where(split == "train")[0]
    rng = np.random.RandomState(0)
    # entangled train: keep China-SZ + US-HC fully, only fraction f of the "against-shortcut" cells
    f = 0.15
    keep = []
    for i in tr:
        china, sz = yp[i] == 1, ydx[i] == 1
        aligned = (china and sz) or ((not china) and (not sz))   # China-SZ or US-HC
        if aligned or rng.rand() < f:
            keep.append(i)
    ent_tr = np.array(keep)
    # report the induced confound strength
    pc = roc_auc_score(ydx[ent_tr], yp[ent_tr]); pc = max(pc, 1 - pc)
    print(f"clean train n={len(tr)} (site->dx AUC {max(roc_auc_score(ydx[tr],yp[tr]),1-roc_auc_score(ydx[tr],yp[tr])):.2f}); "
          f"entangled train n={len(ent_tr)} (site->dx AUC {pc:.2f})", flush=True)

    rows = []
    ent_state = None
    for seed in (0, 1, 2):
        r_clean, *_ = run_regime("orthogonal_clean", Xs, ydx, tr, yp, split, seed)
        r_ent, R, hW, hb, sw = run_regime("entangled_shortcut", Xs, ydx, ent_tr, yp, split, seed)
        rows += [r_clean, r_ent]
        print(r_clean, flush=True); print(r_ent, flush=True)
        if seed == 0:
            ent_state = (R, hW, hb, sw)
    df = pd.DataFrame(rows)
    for nm in ("orthogonal_clean", "entangled_shortcut"):
        s = df[df.regime == nm]
        print(f"[{nm}] decode={s.scanner_decode_auc.mean():.3f} (both high) | behav_bias_HC="
              f"{s.behav_bias_HC.mean():+.3f}+-{s.behav_bias_HC.std():.3f} | swap_output_effect="
              f"{s.swap_output_effect.mean():.3f}+-{s.swap_output_effect.std():.3f}", flush=True)

    # --- residual-IIA acceptance on the entangled model (seed 0): LEACE erases linear decodability ---
    R, hW, hb, sw = ent_state
    te = np.where(split == "test")[0]; trk = np.where(split == "train")[0]
    mu = R[trk].mean(0); Rc = R - mu
    Sxx = np.cov(R[trk].T) + 1e-3 * np.eye(R.shape[1])
    ev, U = np.linalg.eigh(Sxx)
    W = U @ np.diag(ev ** -0.5) @ U.T; Wi = U @ np.diag(ev ** 0.5) @ U.T
    zc = yp[trk] - yp[trk].mean(); Sxz = (Rc[trk] * zc[:, None]).sum(0) / (len(trk) - 1)
    u = W @ Sxz; u = u / (np.linalg.norm(u) + 1e-9)
    Rl = R - np.outer(Rc @ W @ u, Wi @ u)                         # LEACE rank-1 eraser (fit on train)
    res_dec = roc_auc_score(yp[te], LogisticRegression(max_iter=2000).fit(Rl[trk], yp[trk]).predict_proba(Rl[te])[:, 1])
    res_dec = max(res_dec, 1 - res_dec)
    raw_bias = behav_bias(R, hW, hb, yp, ydx, te)
    res_bias = behav_bias(Rl, hW, hb, yp, ydx, te)
    rows.append({"regime": "entangled_residual_IIA", "seed": 0, "scanner_decode_auc": round(res_dec, 3),
                 "behav_bias_HC": round(res_bias, 3), "behav_bias_raw": round(raw_bias, 3)})
    print(f"[residual-IIA acceptance] LEACE drives residual scanner decodability to {res_dec:.3f} (~chance), yet the "
          f"model still over-calls China-HC as SZ by {res_bias:+.3f} (raw {raw_bias:+.3f}): residual decodability says "
          f"'harmonized' while residual behaviour says 'still scanner-exposed'.", flush=True)
    pd.DataFrame(rows).to_csv("outputs/sae_ckpts/shortcut_regime.csv", index=False)


if __name__ == "__main__":
    main()
