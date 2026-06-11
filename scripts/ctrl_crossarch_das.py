#!/usr/bin/env python
"""Closes the single-model causal P0 (reviewer P0-2): does the low-dimensional causal compression
replicate on INDEPENDENTLY-trained models, not just the one multimodal classifier?

The cross-architecture nets (MLP, BrainNetCNN, Transformer) are separately trained on the same FNC
data and each has a penultimate decision rep + a linear head -- exactly the structure DAS needs. We
run the identical k=1 DAS interchange test on each model's penultimate rep: does a single dimension
transfer scanner (frozen readout flips to source) while the model's own disease decision (its head) is
preserved? Plus the random-rotation floor and |cos(DAS, linear scanner probe)|.

If k=1 scanner-IIA is high with disease preserved on these independent models too, the causal
compression is a property of trained FNC decision representations, not of the one multimodal network.
Reuses cross_arch.build_fnc + the model classes. Run in the `project` conda env. CPU.
Writes outputs/sae_ckpts/crossarch_das.csv.
"""
import os, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mib import das as D
from cross_arch import build_fnc, MLP, BrainNetCNN, TransformerBaseline


def train_with_head(model, X, y, tr, epochs=80, lr=1e-3, bs=64, seed=0):
    """Train a net; return (penultimate rep for all rows, head weight, head bias)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr])
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            j = perm[i:i + bs]; opt.zero_grad()
            nn.functional.cross_entropy(model(Xt[j]), yt[j]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        R = model(torch.tensor(X), rep=True)[1].numpy()
    return R, model.head.weight.detach().float(), model.head.bias.detach().float()


def make_pairs(label, idx, n_pairs, seed):
    g = torch.Generator().manual_seed(seed)
    a = idx[label[idx] == 0]; b = idx[label[idx] == 1]
    base = torch.cat([a[torch.randint(len(a), (n_pairs,), generator=g)],
                      b[torch.randint(len(b), (n_pairs,), generator=g)]])
    src = torch.cat([b[torch.randint(len(b), (n_pairs,), generator=g)],
                     a[torch.randint(len(a), (n_pairs,), generator=g)]])
    return base, src


def main():
    mats, X, ydx, split, pop, site = build_fnc()
    Xs = StandardScaler().fit(X[split == "train"]).transform(X)
    Ms = ((mats - mats[split == "train"].mean(0)) / (mats[split == "train"].std(0) + 1e-6)).astype(np.float32)[:, None]
    tr = split == "train"; keep = np.isin(pop, ["US", "China"]); yp = (pop == "China").astype(int)
    models = {"MLP": (MLP(X.shape[1]), Xs), "BrainNetCNN": (BrainNetCNN(), Ms),
              "Transformer": (TransformerBaseline(), Ms[:, 0])}
    rows = []
    for name, (net, feat) in models.items():
        R, hW, hb = train_with_head(net, feat, ydx, tr, seed=0)
        h = torch.tensor(R, dtype=torch.float32)
        trk = np.where(tr & keep)[0]; tek = np.where((~tr) & keep)[0]
        tri, tei = torch.tensor(trk), torch.tensor(tek)
        scanner = torch.tensor(yp)
        # frozen scanner probe on train penultimate rep
        lr = LogisticRegression(max_iter=3000).fit(R[trk], yp[trk])
        sw = torch.tensor(lr.coef_.ravel(), dtype=torch.float32); sb = torch.tensor(float(lr.intercept_[0]))
        probe_auc = roc_auc_score(yp[tek], (h[tei] @ sw + sb).numpy())
        bt, st = make_pairs(scanner, tri, 3000, 0); bv, sv = make_pairs(scanner, tei, 3000, 100)
        das = D.train_das(h, scanner, hW, hb, sw, sb, bt, st, k=1, steps=600, lr=5e-3, lam=1.0, seed=0, device="cpu")
        m = D.eval_iia(das, h, scanner, hW, hb, sw, sb, bv, sv, device="cpu")
        rnd = D.DASRotation(h.shape[1], k=1, seed=7)
        rm = D.eval_iia(rnd, h, scanner, hW, hb, sw, sb, bv, sv, device="cpu")
        u = das._W()[0].detach().numpy(); w = lr.coef_.ravel()
        cos = abs(float(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-12)))
        rows.append({"model": name, "dim": R.shape[1], "scanner_probe_auc": round(probe_auc, 3),
                     "scanner_iia_k1": round(m["scanner_iia"], 3), "disease_preserved": round(m["disease_preserved"], 3),
                     "scanner_iia_random": round(rm["scanner_iia"], 3), "cos_das_probe": round(cos, 3)})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv("outputs/sae_ckpts/crossarch_das.csv", index=False)
    print("\nwrote outputs/sae_ckpts/crossarch_das.csv")
    print("INTERPRETATION: high k=1 scanner-IIA with disease preserved on these independently-trained models"
          " = the low-dimensional causal compression is not specific to the one multimodal classifier.")


if __name__ == "__main__":
    main()
