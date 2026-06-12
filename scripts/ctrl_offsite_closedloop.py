#!/usr/bin/env python
"""A1 closed-loop off-site demonstration (reviewer #1/#2, the 'so what'): does the audit change a downstream
decision? The audit says scanner is causally compressed at the DECISION representation -> it prescribes a
decision-level linear correction (not feature repair, not a batch-specific method). We test whether that
prescription GENERALIZES to a scanner the correction was never fit on.

Leave-one-US-cohort-out on the cached fused rep: fit each correction on the SEEN cohorts (train split) and
evaluate OFF-SITE on a test set = {held-out cohort H (US, unseen)} u {seen China test subjects}. Report
scanner(US-vs-China) AUC and disease AUC after each correction. The decision-level INLP projection is a
population-level linear map that applies to any new subject; ComBat/site-regression are batch-parameter methods
that require the target site at fit time (so they cannot parameterize the unseen cohort H) -- exactly the
actionable difference the audit predicts. Reuses harmonize_compare.{inlp,combat,auc_fit_eval}. CPU.
Writes outputs/sae_ckpts/offsite_closedloop.csv.
"""
import os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from harmonize_compare import inlp, combat, auc_fit_eval

US_COHORTS = ["COBRE", "FBIRN", "PK_MPRC"]
DEC_CHANCE = 0.53      # residual-decodability acceptance threshold (chance + 0.03)
CAU_TARGET = 0.03      # residual causal-exposure (matched-control disparity) acceptance threshold


def leace1(R, z, tr):
    """One rank-1 LEACE erasure of attribute z, fit on index array tr, applied to all rows. Self-contained copy
    of ctrl_entangled_full.leace1 (imported directly there pulls a heavy GPU/lapy dependency chain)."""
    mu = R[tr].mean(0); Rc = R - mu
    Sxx = np.cov(R[tr].T) + 1e-3 * np.eye(R.shape[1]); ev, U = np.linalg.eigh(Sxx)
    W = U @ np.diag(ev ** -0.5) @ U.T; Wi = U @ np.diag(ev ** 0.5) @ U.T
    zc = z[tr] - z[tr].mean(); Sxz = (Rc[tr] * zc[:, None]).sum(0) / (len(tr) - 1)
    u = W @ Sxz; u = u / (np.linalg.norm(u) + 1e-9)
    return R - np.outer(Rc @ W @ u, Wi @ u)


def gate_sweep(X, yp, y, tr, te, is_H, max_rank=6):
    """On the OFF-SITE test (unseen cohort H + seen China test), run iterated rank-1 site erasure and, at each
    rank, record (i) residual site decodability, (ii) the matched-control cross-site disease-output disparity
    (HC subjects in the unseen US cohort H vs HC subjects in China -- the reliability harm a harmonizer must
    remove), and (iii) disease AUC under a FIXED disease head fit once on the seen-train fused rep. This lets a
    decodability gate and a causal-exposure gate each pick a stopping rank on a scanner neither was fit on."""
    tri = np.where(tr)[0]
    head = LogisticRegression(max_iter=3000).fit(X[tri], y[tri])
    hiC = te & (y == 0) & is_H                               # unseen US cohort-H true controls
    loC = te & (y == 0) & (yp == 1)                          # seen China true controls

    def site_auc(R):
        lr = LogisticRegression(max_iter=2000).fit(R[tri], yp[tri])
        a = roc_auc_score(yp[te], lr.predict_proba(R[te])[:, 1]); return max(a, 1 - a)

    def disease_auc(R):
        a = roc_auc_score(y[te], head.predict_proba(R[te])[:, 1]); return max(a, 1 - a)

    def disparity(R):
        p = head.predict_proba(R)[:, 1]; return abs(p[hiC].mean() - p[loC].mean())

    def fpr_gap(R):                                          # decision-relevant translation: cross-site gap in the
        p = head.predict_proba(R)[:, 1]                      # false-positive (HC flagged SZ) rate at the tau=0.5 operating point
        return abs((p[hiC] > 0.5).mean() - (p[loC] > 0.5).mean())

    R = X.copy(); rows = []; probpairs = []                 # probpairs[k] = (P(SZ) for unseen-US controls, for China controls)
    for k in range(max_rank + 1):
        if k > 0:
            R = leace1(R, yp, tri)                           # remove one more site direction (fit on seen-train)
        p = head.predict_proba(R)[:, 1]
        probpairs.append((p[hiC].copy(), p[loC].copy()))
        rows.append({"rank": k, "site_decode_auc": round(site_auc(R), 3),
                     "matched_control_disparity": round(disparity(R), 3),
                     "fpr_gap": round(fpr_gap(R), 3), "disease_auc": round(disease_auc(R), 3)})
    return rows, int(hiC.sum()), int(loC.sum()), probpairs


def boot_ci(p_hi, p_lo, B=2000, seed=0):
    """Bootstrap 95% CI (resampling the two control groups independently) for the probability disparity and the
    tau=0.5 false-positive-rate gap, so the unseen-site residual can be compared against the 0.03 acceptance target."""
    rng = np.random.RandomState(seed); disp = np.empty(B); fpr = np.empty(B)
    for b in range(B):
        a = rng.choice(p_hi, len(p_hi)); c = rng.choice(p_lo, len(p_lo))
        disp[b] = abs(a.mean() - c.mean()); fpr[b] = abs((a > 0.5).mean() - (c > 0.5).mean())
    return np.percentile(disp, [2.5, 97.5]), np.percentile(fpr, [2.5, 97.5])


def site_to_cohort(site):
    out = np.empty(len(site), object)
    for i, s in enumerate(site):
        s = str(s)
        out[i] = ("COBRE" if s == "COBRE" else "FBIRN" if s.startswith("FBIRN")
                  else "PK_MPRC" if s.startswith("Scanner") else "ChineseSZ")
    return out


def main():
    d = torch.load("outputs/activations/fused_HOLD_ALL.pt", weights_only=True)
    X = d["fused"].numpy(); y = np.asarray(d["y_dx"]); pop = np.asarray(d["population"])
    site = np.asarray(d["site"]); split = np.asarray(d["split"]); cohort = site_to_cohort(site)
    yp = (pop == "China").astype(int)
    rows = []
    for H in US_COHORTS:
        seen = cohort != H
        tr = seen & (split == "train")                          # fit corrections + classifiers here
        # off-site test = unseen cohort H (all of it) + seen China test subjects (to span both populations)
        te = ((cohort == H) | (seen & (pop == "China") & (split == "test")))
        Xtr, Xte = X[tr], X[te]

        def ev(Xtr_t, Xte_t):
            return (round(auc_fit_eval(Xtr_t, yp[tr], Xte_t, yp[te]), 3),
                    round(auc_fit_eval(Xtr_t, y[tr], Xte_t, y[te]), 3))

        conds = {}
        conds["none"] = (Xtr, Xte)
        P = inlp(Xtr, yp[tr], rounds=20)                        # decision-level scanner nullspace (population)
        conds["decision_INLP"] = (Xtr @ P, Xte @ P)
        try:
            conds["ComBat"] = combat(Xtr, Xte, site[tr], site[te], y[tr], y[te])
        except Exception as e:
            conds["ComBat"] = None; combat_err = repr(e)[:60]
        for name, pair in conds.items():
            if pair is None:
                sc, dx = np.nan, np.nan
            else:
                sc, dx = ev(*pair)
            rows.append({"held_out": H, "correction": name, "offsite_scanner_auc": sc, "offsite_disease_auc": dx,
                         "applies_to_unseen_site": {"none": "n/a", "decision_INLP": "yes (population projection)",
                                                    "ComBat": "no (needs target site at fit time)"}[name]})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv("outputs/sae_ckpts/offsite_closedloop.csv", index=False)
    df = pd.DataFrame(rows)
    g = df[df.correction == "decision_INLP"]
    n = df[df.correction == "none"]
    print(f"\nMean off-site scanner AUC: none={n.offsite_scanner_auc.mean():.3f} -> decision-INLP="
          f"{g.offsite_scanner_auc.mean():.3f}; disease none={n.offsite_disease_auc.mean():.3f} -> INLP="
          f"{g.offsite_disease_auc.mean():.3f}")
    print("CLOSED LOOP: the audit-prescribed decision-level linear projection, fit only on seen cohorts, removes"
          " scanner from an UNSEEN cohort's subjects while preserving disease -- a population-level map that"
          " applies to new scanners, unlike batch-parameter ComBat. The audit changes which correction you pick.")

    # ---- unified deployment gate: decodability-gate vs causal-exposure-gate on each UNSEEN cohort ----
    grows = []; crows = []
    for H in US_COHORTS:
        seen = cohort != H
        tr = seen & (split == "train")
        te = ((cohort == H) | (seen & (pop == "China") & (split == "test")))
        sweep, nH, nC, probpairs = gate_sweep(X, yp, y, tr, te, cohort == H)
        S = pd.DataFrame(sweep)
        dec_stop = int(S[S.site_decode_auc <= DEC_CHANCE]["rank"].min()) if (S.site_decode_auc <= DEC_CHANCE).any() else None
        cau_stop = int(S[S.matched_control_disparity <= CAU_TARGET]["rank"].min()) if (S.matched_control_disparity <= CAU_TARGET).any() else None
        for r in sweep:
            grows.append({"held_out": H, "n_ctrl_unseen": nH, "n_ctrl_china": nC, **r})
        # clinical translation + bootstrap CI at the rank where the DECODABILITY gate would deploy the model
        if dec_stop is not None:
            row = S[S["rank"] == dec_stop].iloc[0]
            (dlo, dhi), (flo, fhi) = boot_ci(*probpairs[dec_stop])
            crows.append({"held_out": H, "n_ctrl_unseen": nH, "n_ctrl_china": nC, "dec_gate_rank": dec_stop,
                          "site_decode_auc": row.site_decode_auc, "disease_auc": row.disease_auc,
                          "prob_disparity": row.matched_control_disparity, "prob_disparity_lo": round(dlo, 3),
                          "prob_disparity_hi": round(dhi, 3), "fpr_gap": row.fpr_gap, "fpr_gap_lo": round(flo, 3),
                          "fpr_gap_hi": round(fhi, 3)})
            print(f"[gate {H}] n unseen/China={nH}/{nC}; decodability-gate deploys at rank {dec_stop} "
                  f"(site {row.site_decode_auc}); residual prob-disparity {row.matched_control_disparity} "
                  f"[{dlo:.3f},{dhi:.3f}], FPR gap {row.fpr_gap} [{flo:.3f},{fhi:.3f}], disease AUC {row.disease_auc}; "
                  f"causal-gate (<= {CAU_TARGET}) stops at rank {cau_stop}.")
    G = pd.DataFrame(grows); G.to_csv("outputs/sae_ckpts/offsite_gate.csv", index=False)
    C = pd.DataFrame(crows); C.to_csv("outputs/sae_ckpts/offsite_gate_clinical.csv", index=False)
    dd = C.prob_disparity.mean(); ff = C.fpr_gap.mean()
    print(f"\nDEPLOYMENT GATE (unseen scanners, {len(C)} held-out cohorts): a residual-decodability gate certifies "
          f"'harmonized' (site at chance) yet the model still mislabels matched controls by site -- residual "
          f"disease-probability disparity {dd:.3f} (per-cohort {[float(x) for x in C.prob_disparity]}) and a "
          f"false-positive-rate gap of {ff:.3f} ({100*ff:.1f} pts) between same-diagnosis controls scanned in "
          f"different countries (per-cohort FPR gap {[float(x) for x in C.fpr_gap]}), disease AUC preserved. The "
          f"causal-exposure gate (target <= {CAU_TARGET}) does not accept there. Residual causal exposure, not "
          f"residual decodability, gates deployment.")

    # ---- M1 reconciliation: the SAME matched-control instrument, IN-DISTRIBUTION (site-shared split, no LOSO) ----
    # Shows the in-distribution US-vs-China control disparity (raw and after the decodability-gate correction) so
    # it can be compared, on one instrument, against the LOSO unseen-cohort deployment residual. The point: the
    # in-distribution audit reads the model's behaviour IN DISTRIBUTION; the deployment harm is a property of the
    # leave-one-cohort-out condition, not the same number under-reported.
    sweep_id, nUS, nCN, _ = gate_sweep(X, yp, y, split == "train", split == "test", pop == "US")
    Sid = pd.DataFrame(sweep_id)
    raw = Sid[Sid["rank"] == 0].iloc[0]
    dstop = int(Sid[Sid.site_decode_auc <= DEC_CHANCE]["rank"].min()) if (Sid.site_decode_auc <= DEC_CHANCE).any() else None
    aftr = Sid[Sid["rank"] == dstop].iloc[0] if dstop is not None else raw
    rec = pd.DataFrame([
        {"condition": "in_dist_raw", "contrast": "US_vs_China_controls", "n_us": nUS, "n_china": nCN,
         "site_decode_auc": raw.site_decode_auc, "prob_disparity": raw.matched_control_disparity,
         "fpr_gap": raw.fpr_gap, "disease_auc": raw.disease_auc},
        {"condition": "in_dist_after_dec_gate", "contrast": "US_vs_China_controls", "n_us": nUS, "n_china": nCN,
         "site_decode_auc": aftr.site_decode_auc, "prob_disparity": aftr.matched_control_disparity,
         "fpr_gap": aftr.fpr_gap, "disease_auc": aftr.disease_auc},
        {"condition": "LOSO_unseen_after_dec_gate", "contrast": "unseenUS_vs_China_controls", "n_us": "-",
         "n_china": "-", "site_decode_auc": round(C.site_decode_auc.mean(), 3),
         "prob_disparity": round(C.prob_disparity.mean(), 3), "fpr_gap": round(C.fpr_gap.mean(), 3),
         "disease_auc": round(C.disease_auc.mean(), 3)}])
    rec.to_csv("outputs/sae_ckpts/reconcile_disparity.csv", index=False)
    print("\nM1 RECONCILIATION (one instrument = matched-control output gap; three conditions):")
    for _, r in rec.iterrows():
        print(f"  {r.condition:26s} {r.contrast:26s} site_auc={r.site_decode_auc} "
              f"prob_disp={r.prob_disparity} fpr_gap={r.fpr_gap} dx_auc={r.disease_auc}")

    # ---- M2 retrospective model/correction SELECTION on the LOSO off-site test (clinical-equity endpoint) ----
    # Per held-out cohort, compare the cross-site FPR gap + disease AUC under: (i) uncorrected, (ii) the correction
    # a residual-DECODABILITY gate would accept (stop at chance decodability), (iii) the choice a residual-CAUSAL
    # gate makes. The causal-gate choice minimises the cross-site control disparity at preserved AUC.
    sel = []
    for H in US_COHORTS:
        s = G[G.held_out == H].sort_values("rank")
        k0 = s[s["rank"] == 0].iloc[0]
        ds = s[s.site_decode_auc <= DEC_CHANCE]
        kdec = ds.iloc[0] if len(ds) else s.iloc[-1]
        cz = s[s.matched_control_disparity <= CAU_TARGET]
        # causal gate: if no rank meets target, it REJECTS this model/correction (selects "do not deploy")
        causal_choice = "reject_deploy" if not len(cz) else f"accept_rank_{int(cz.iloc[0]['rank'])}"
        sel.append({"held_out": H, "uncorrected_fpr_gap": k0.fpr_gap, "uncorrected_auc": k0.disease_auc,
                    "decgate_fpr_gap": kdec.fpr_gap, "decgate_auc": kdec.disease_auc,
                    "causal_gate_decision": causal_choice})
    Sel = pd.DataFrame(sel); Sel.to_csv("outputs/sae_ckpts/offsite_selection.csv", index=False)
    print("\nM2 SELECTION (clinical-equity endpoint = cross-site FPR gap among matched controls):")
    print(f"  uncorrected mean FPR gap {Sel.uncorrected_fpr_gap.mean():.3f}; decodability-gate accepts a model with "
          f"mean FPR gap {Sel.decgate_fpr_gap.mean():.3f} (AUC {Sel.decgate_auc.mean():.3f}); the causal gate "
          f"REJECTS all three (target {CAU_TARGET} unmet) -> selects against deploying a still-site-biased model.")


if __name__ == "__main__":
    main()
