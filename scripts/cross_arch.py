#!/usr/bin/env python
"""Cross-architecture robustness: is "distributed scanner encoding + low-dim removable subspace +
unit-ablation null" a MultiViT2 artifact, or a property of the FNC representation itself?

Same 4-cohort subjects + same M_hold 70/30 split. Representations: (a) raw FNC edge vector (NO
model), (b) MLP penultimate, (c) BrainNetCNN penultimate. For each, held-out: scanner & disease
AUC; scanner/disease after INLP subspace removal; scanner after top-k unit ablation (refit probe);
participation ratio of the scanner readout. If the pattern holds across all three (esp. raw FNC),
it is not MultiViT2-specific. Writes outputs/sae_ckpts/cross_arch.csv.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
import importlib.util, json, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from geomultivit.data.multicohort import load_geometric_cohort
from geomultivit.data.preprocess import fnc_to_matrix
_spec = importlib.util.spec_from_file_location("hc", os.path.join(os.path.dirname(__file__), "harmonize_compare.py"))
hc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(hc)   # inlp, auc_fit_eval

H5 = "/home/users/ybi3/data/szdataset_modified.h5"
SPLIT = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/split_info.json"


def participation_ratio(w):
    w = np.abs(w); return float(w.sum() ** 2 / (w ** 2).sum() + 1e-12)


def build_fnc():
    si = json.load(open(SPLIT))
    sm = {**{s: "train" for s in si["train_ids"]}, **{s: "test" for s in si["test_ids"]}}
    g = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False).df
    g = g[g.cohort.isin(["COBRE", "FBIRN", "ChineseSZ", "PK_MPRC"])].drop_duplicates("SubjectID")
    g = g[g.SubjectID.isin(set(sm))].reset_index(drop=True)
    iu = np.triu_indices(53, 1)
    mats = np.stack([fnc_to_matrix(r["sFNC"], n_icns=53) for _, r in g.iterrows()]).astype(np.float32)
    X = mats[:, iu[0], iu[1]]                              # (N,1378) edge vectors
    return (mats, X, g.Diagnosis.to_numpy().astype(int),
            np.array([sm[s] for s in g.SubjectID]), g.population.to_numpy(), g.site.to_numpy())


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU())
        self.head = nn.Linear(64, 2)
    def forward(self, x, rep=False):
        r = self.net(x); return (self.head(r), r) if rep else self.head(r)


class E2E(nn.Module):
    def __init__(self, ic, oc, d=53):
        super().__init__(); self.row = nn.Conv2d(ic, oc, (1, d)); self.col = nn.Conv2d(ic, oc, (d, 1))
    def forward(self, x):
        return self.row(x) + self.col(x)                  # (B,oc,1,d)+(B,oc,d,1) -> (B,oc,d,d)


class BrainNetCNN(nn.Module):
    def __init__(self, d=53):
        super().__init__(); self.a = nn.LeakyReLU(0.1)
        self.e2e = E2E(1, 16, d); self.e2n = nn.Conv2d(16, 32, (1, d)); self.fc1 = nn.Linear(32 * d, 64); self.head = nn.Linear(64, 2)
    def forward(self, x, rep=False):
        h = self.a(self.e2e(x)); h = self.a(self.e2n(h)).flatten(1); r = self.a(self.fc1(h))
        return (self.head(r), r) if rep else self.head(r)


class TransformerBaseline(nn.Module):
    """Plain Transformer encoder over the FNC graph: each of the 53 ICN rows (its 53-d connectivity
    profile) is a token; self-attention over the 53 tokens, mean-pool -> 64-d penultimate. A genuinely
    distinct architecture from the MLP (flat vector) and BrainNetCNN (edge-to-edge conv)."""
    def __init__(self, n=53, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(n, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 2)
    def forward(self, x, rep=False):                      # x: (B,53,53) row-tokens
        h = self.enc(self.proj(x)).mean(1)                # (B,d_model)
        return (self.head(h), h) if rep else self.head(h)


def train_net(model, X, y, tr, epochs=80, lr=1e-3, bs=64, seed=0):
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
        return model(torch.tensor(X), rep=True)[1].numpy()


def analyze(R, ydx, yp, tr, te, rounds=20, kfrac=0.10):
    R = StandardScaler().fit(R[tr]).transform(R)          # standardize (fit train) for stable INLP
    out = {"scanner": hc.auc_fit_eval(R[tr], yp[tr], R[te], yp[te]),
           "disease": hc.auc_fit_eval(R[tr], ydx[tr], R[te], ydx[te])}
    P = hc.inlp(R[tr], yp[tr], rounds)
    out["scanner_subspace_removed"] = hc.auc_fit_eval(R[tr] @ P, yp[tr], R[te] @ P, yp[te])
    out["disease_subspace_removed"] = hc.auc_fit_eval(R[tr] @ P, ydx[tr], R[te] @ P, ydx[te])
    w = LogisticRegression(solver="liblinear", max_iter=2000).fit(R[tr], yp[tr]).coef_.ravel()
    k = max(1, int(kfrac * R.shape[1])); top = np.argsort(-np.abs(w))[:k]
    Rab = R.copy(); Rab[:, top] = 0.0
    out["scanner_unit_ablated"] = hc.auc_fit_eval(Rab[tr], yp[tr], Rab[te], yp[te])
    out["participation_ratio"] = round(participation_ratio(w), 1)
    out["dim"] = R.shape[1]; out["n_ablated"] = k
    return out


def main():
    mats, X, ydx, split, pop, site = build_fnc()
    tr, te = split == "train", split == "test"
    keep = np.isin(pop, ["US", "China"]); yp = (pop == "China").astype(int)
    trk, tek = tr & keep, te & keep
    print(f"N={len(X)} train={tr.sum()} test={te.sum()} | dx {np.bincount(ydx)}", flush=True)

    Xs = StandardScaler().fit(X[tr]).transform(X)         # standardized FNC for the nets
    Ms = ((mats - mats[tr].mean(0)) / (mats[tr].std(0) + 1e-6)).astype(np.float32)[:, None]  # (N,1,53,53)

    reps = {"raw_FNC": X}
    print("training MLP...", flush=True);          reps["MLP"] = train_net(MLP(X.shape[1]), Xs, ydx, tr)
    print("training BrainNetCNN...", flush=True);  reps["BrainNetCNN"] = train_net(BrainNetCNN(), Ms, ydx, tr)
    print("training Transformer...", flush=True);  reps["Transformer"] = train_net(TransformerBaseline(), Ms[:, 0], ydx, tr)

    rows = []
    for name, R in reps.items():
        m = {"arch": name}
        # scanner/disease use US/China mask; INLP/ablation fit on scanner-labeled rows
        sc = analyze(R[keep], ydx[keep], yp[keep], trk[keep], tek[keep])
        m.update(sc); rows.append(m)
        print(f"{name:12s} scanner {sc['scanner']:.3f} disease {sc['disease']:.3f} | "
              f"subspace→scanner {sc['scanner_subspace_removed']:.3f} disease {sc['disease_subspace_removed']:.3f} | "
              f"unit-ablate→scanner {sc['scanner_unit_ablated']:.3f} | PR {sc['participation_ratio']}", flush=True)
    pd.DataFrame(rows).to_csv("outputs/sae_ckpts/cross_arch.csv", index=False)
    print("\nwrote cross_arch.csv")


if __name__ == "__main__":
    main()
