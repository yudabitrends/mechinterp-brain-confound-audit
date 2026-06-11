#!/usr/bin/env python
"""C1 (threat T1) — is "scanner is distributed and not removable by feature ablation" a real
property of the representation, or just an artifact of the SUBTRACTIVE operator?

The manuscript contrasts two facts: (i) zeroing the top scanner-tuned SAE features (a
disease-blind, ADDITIVE subtraction  A' = A - Σ_{i∈S} f_i(A) W_dec[i]) barely dents the scanner
readout, yet (ii) INLP (a learned PROJECTION) collapses it. The hostile reviewer's T1: this could
be an operator confound (subtraction != projection) rather than evidence that scanner is
distributed/redundant.

We disentangle OPERATOR from BASIS by applying, to the SAME scanner-tuned SAE feature set S,
all three removals at a representative hook (sFNC_encoder.norm, where scanner is already ~0.92
pre-fusion) on cached CLS activations — NO model forward:
  raw                         : scanner / disease readout on A.
  subtract_scanner_features   : A - Σ_{i∈S} f_i(A) W_dec[i]      (the manuscript's ablation op).
  project_out_feature_span    : A (I - B Bᵀ), B = QR(W_dec[S]ᵀ)  (SAME features, projection op).
  inlp_in_feature_span        : INLP confined to span(B) (project A onto B, erase, reconstruct).
  random_span_projection      : project out r random directions  (rank-matched specificity ctrl).
  full_space_inlp             : unconstrained INLP (the manuscript's success, reference).
Each readout is FIT on transformed TRAIN, evaluated on transformed TEST.

Decision rule:
  * subtraction keeps scanner high AND projection-in-span ALSO keeps it high  -> genuinely
    distributed; the dissociation is NOT an operator artifact (keep the claim, operator-matched).
  * subtraction keeps it high but projection-in-span COLLAPSES it             -> the wrong-operator
    story; reframe to "subtraction is the wrong operator", soften distributedness.
Writes outputs/sae_ckpts/ctrl_feature_subspace.csv.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mib.sae import SAEConfig, SparseAutoencoder
from mib.probe import encode_features, per_feature_auc

HOOK_DEFAULT = "sFNC_encoder.norm"


def load_sae(hook, tag="HOLDtrain", seed=0):
    d = torch.load(f"outputs/sae_ckpts/sae_{tag}_{hook}_seed{seed}.pt",
                   weights_only=True, map_location="cpu")
    sae = SparseAutoencoder(SAEConfig(**d["cfg"])); sae.load_state_dict(d["state_dict"])
    return sae.eval()


def auc(Atr, ytr, Ate, yte):
    sc = StandardScaler().fit(Atr)
    c = LogisticRegression(max_iter=2000, solver="liblinear").fit(sc.transform(Atr), ytr)
    return float(roc_auc_score(yte, c.predict_proba(sc.transform(Ate))[:, 1]))


def inlp_P(X, y, rounds):
    d = X.shape[1]; P = np.eye(d); Xc = X.copy()
    for _ in range(rounds):
        w = LogisticRegression(solver="liblinear", max_iter=200).fit(Xc, y).coef_.ravel()
        n = np.linalg.norm(w)
        if n < 1e-9:
            break
        w /= n; Pw = np.eye(d) - np.outer(w, w); Xc = Xc @ Pw; P = P @ Pw
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hook", default=HOOK_DEFAULT)
    ap.add_argument("--acts", default="outputs/activations/acts_HOLD_ALL_cls.pt")
    ap.add_argument("--labels", default="outputs/activations/labels_HOLD_ALL.csv")
    ap.add_argument("--fracs", default="0.01,0.02,0.05")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/sae_ckpts/ctrl_feature_subspace.csv")
    args = ap.parse_args()

    acts = torch.load(args.acts, weights_only=True)
    A = acts[args.hook].float()                                   # (N, d)
    lab = pd.read_csv(args.labels)
    pop = lab["population"].to_numpy(); split = lab["split"].to_numpy()
    ydx = lab["label"].to_numpy().astype(int)
    keep = np.isin(pop, ["US", "China"])
    tr = (split == "train") & keep; te = (split == "test") & keep
    yscan = (pop == "China").astype(int)
    print(f"hook={args.hook} d={A.shape[1]}  N_train={tr.sum()} N_test={te.sum()}", flush=True)

    sae = load_sae(args.hook, seed=args.seed)
    Wdec = sae.W_dec.detach().numpy()                            # (d_sae, d)
    Ftr = encode_features(sae, A[tr].numpy())                    # (n_tr, d_sae)
    # label scanner-tuned features by univariate AUC on TRAIN
    fa = per_feature_auc(Ftr, yscan[tr])
    order = np.argsort(-np.abs(fa - 0.5))                        # most scanner-informative first
    An = A.numpy(); rng = np.random.default_rng(args.seed)

    rows = []

    def record(cond, frac, r, Atr_t, Ate_t):
        # degeneracy guard: if the transform annihilates the representation (e.g. the
        # feature-decoder span fills the whole space), an AUC on ~constant features is
        # meaningless -> report NaN rather than a spurious number.
        if np.linalg.norm(Ate_t[te]) < 1e-6 * np.linalg.norm(An[te]):
            s = dz = float("nan")
            note = "degenerate (span fills space -> rep annihilated)"
        else:
            s = auc(Atr_t[tr], yscan[tr], Ate_t[te], yscan[te])
            dz = auc(Atr_t[tr], ydx[tr], Ate_t[te], ydx[te])
            note = ""
        rows.append({"condition": cond, "frac": frac, "rank": r,
                     "scanner_auc": round(s, 4) if s == s else s,
                     "disease_auc": round(dz, 4) if dz == dz else dz})
        print(f"{cond:28s} frac={frac:<5} r={r:<4} scanner={s:.3f} disease={dz:.3f} {note}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    record("raw", "-", 0, An, An)
    P_full = inlp_P(An[tr], yscan[tr], rounds=40)
    record("full_space_inlp", "-", 40, An @ P_full, An @ P_full)

    F_all = encode_features(sae, An)                            # (N, d_sae) for subtraction op
    for frac in [float(x) for x in args.fracs.split(",")]:
        m = max(1, int(len(fa) * frac))
        S = order[:m]
        # operator 1: subtractive ablation (disease-blind additive removal of S features)
        contrib = F_all[:, S] @ Wdec[S]                         # (N, d)
        A_sub = An - contrib
        record("subtract_scanner_features", frac, len(S), A_sub, A_sub)
        # operator 2: project out the span of those features' decoder directions
        B, _ = np.linalg.qr(Wdec[S].T)                          # (d, r)
        r = B.shape[1]
        Pproj = np.eye(An.shape[1]) - B @ B.T
        record("project_out_feature_span", frac, r, An @ Pproj, An @ Pproj)
        # operator 3: INLP confined to span(B): erase scanner WITHIN the r-dim feature subspace
        c_tr = An[tr] @ B                                        # coords in span
        Pc = inlp_P(c_tr, yscan[tr], rounds=min(r, 40))
        A_inlp_span = An - (An @ B) @ (np.eye(r) - Pc) @ B.T     # remove erased component, keep rest
        record("inlp_in_feature_span", frac, r, A_inlp_span, A_inlp_span)
        # rank-matched random-direction projection (specificity control)
        Br, _ = np.linalg.qr(rng.standard_normal((An.shape[1], r)))
        Pr = np.eye(An.shape[1]) - Br @ Br.T
        record("random_span_projection", frac, r, An @ Pr, An @ Pr)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
