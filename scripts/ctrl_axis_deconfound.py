#!/usr/bin/env python
"""C3 (threat T3) — the WITHIN-COUNTRY site axis as the clean headline, and an age/sex deconfound.

The hostile review's strongest objection: our "scanner confound" axis was US-vs-China population,
which conflates acquisition (scanner/site) with population/sampling. The fix the user chose is to
make the *within-country site* axis the headline — same country, same population, two scanners:
  US:    COBRE  vs  Scanner2 (PK_MPRC)
  China: GZ     vs  ZMD
On each contrast we measure, on the fused decision rep (held-out): site readout AUC, INLP
site-subspace removal (site down, disease preserved?), DAS k=1 site-IIA + disease-preservation,
and then REPEAT after linearly regressing Age+Sex out of the rep. If site survives age/sex
removal, the axis is acquisition, not a demographic proxy. Population (US-vs-China) is reported
alongside as the confounded-but-illustrative case.

Demographics (Age, Gender) are read from the source h5 (written with numpy>=2, so we shim
numpy._core for the <2 base env). Cached fused rep + frozen head; no model forward.
Writes outputs/sae_ckpts/ctrl_axis_deconfound.csv.
"""
import argparse, os, sys, types
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mib import das as D
from ctrl_das_null import load_head

H5 = "/home/users/ybi3/data/szdataset_modified.h5"
MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"


def read_demographics():
    """Read SubjectID/Age/Gender from the pandas-HDFStore h5 (numpy>=2 pickled block)."""
    import numpy, numpy.core
    sys.modules.setdefault("numpy._core", numpy.core)
    for sub in (".multiarray", ".numeric", ".umath", "._multiarray_umath"):
        try:
            sys.modules.setdefault("numpy._core" + sub, __import__("numpy.core" + sub, fromlist=["x"]))
        except Exception:
            pass
    parts = [pd.read_hdf(H5, k) for k in ("train", "test")]
    demo = pd.concat(parts)[["SubjectID", "Age", "Gender"]].copy()
    demo["SubjectID"] = demo["SubjectID"].astype(str)
    demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")
    demo["Gender"] = pd.to_numeric(demo["Gender"], errors="coerce")
    return demo.drop_duplicates("SubjectID").set_index("SubjectID")


def auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    c = LogisticRegression(max_iter=2000, solver="liblinear").fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, c.predict_proba(sc.transform(Xte))[:, 1]))


def inlp_P(X, y, rounds=20):
    d = X.shape[1]; P = np.eye(d); Xc = X.copy()
    for _ in range(rounds):
        w = LogisticRegression(solver="liblinear", max_iter=200).fit(Xc, y).coef_.ravel()
        n = np.linalg.norm(w)
        if n < 1e-9:
            break
        w /= n; Pw = np.eye(d) - np.outer(w, w); Xc = Xc @ Pw; P = P @ Pw
    return P


def residualize_age_sex(X, age, sex, tr):
    """Linearly regress [Age, Sex, 1] out of X; fit on train rows, apply to all. NaN->train mean."""
    a = age.copy(); s = sex.copy()
    a[np.isnan(a)] = np.nanmean(age[tr]); s[np.isnan(s)] = np.nanmean(sex[tr])
    C = np.column_stack([a, s])
    lr = LinearRegression().fit(C[tr], X[tr])
    return X - lr.predict(C)


def das_iia_axis(h, axis, hW, hb, tr_idx, te_idx, pairs, seed, dev, steps=400):
    """DAS k=1 IIA on a binary axis (axis: torch long {0,1}), disease-preservation via head."""
    from ctrl_das_null import make_pairs
    lr = LogisticRegression(max_iter=3000).fit(h[tr_idx].numpy(), axis[tr_idx].numpy())
    aw = torch.tensor(lr.coef_.ravel(), dtype=torch.float32)
    ab = torch.tensor(float(lr.intercept_[0]), dtype=torch.float32)
    bt, st = make_pairs(axis, tr_idx, pairs, seed)
    bv, sv = make_pairs(axis, te_idx, pairs, seed + 100)
    das = D.train_das(h, axis, hW, hb, aw, ab, bt, st, k=1, steps=steps, lr=5e-3, lam=1.0,
                      seed=seed, device=dev)
    return D.eval_iia(das, h, axis, hW, hb, aw, ab, bv, sv, device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="outputs/activations/fused_HOLD_ALL.pt")
    ap.add_argument("--pairs", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/sae_ckpts/ctrl_axis_deconfound.csv")
    ap.add_argument("--device", default=None)
    ap.add_argument("--das", action="store_true",
                    help="also run the (slow) per-axis DAS-IIA; default skips it and reports the "
                         "fast probe/INLP/age-sex deconfound evidence only")
    ap.add_argument("--das-steps", type=int, default=400)
    ap.add_argument("--das-resid", action="store_true", help="also DAS the age/sex-residualized axis")
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    d = torch.load(args.fused, weights_only=True)
    h = d["fused"].float(); X = d["fused"].numpy()
    pop = np.asarray(d["population"]); site = np.asarray(d["site"]); split = np.asarray(d["split"])
    ydx = np.asarray(d["y_dx"]).astype(int); sid = np.asarray(d["subject_id"]).astype(str)
    hW, hb = load_head(MHOLD)

    demo = read_demographics()
    age = np.array([demo.loc[s, "Age"] if s in demo.index else np.nan for s in sid], float)
    sex = np.array([demo.loc[s, "Gender"] if s in demo.index else np.nan for s in sid], float)
    print(f"demographics matched: Age {np.isfinite(age).sum()}/{len(sid)}  "
          f"Sex {np.isfinite(sex).sum()}/{len(sid)}", flush=True)

    contrasts = [
        ("US_COBRE_vs_Scanner2", "US", ["COBRE", "Scanner2"], "Scanner2"),
        ("China_GZ_vs_ZMD",      "China", ["GZ", "ZMD"],      "ZMD"),
        ("population_US_vs_China", None, None, None),   # confounded reference
    ]
    rows = []

    def record(**kw):
        rows.append(kw); pd.DataFrame(rows).to_csv(args.out, index=False)
        print("  ".join(f"{k}={v}" for k, v in kw.items()), flush=True)

    for name, country, sites, pos in contrasts:
        if country is None:                              # population axis
            keep = np.isin(pop, ["US", "China"]); axis_np = (pop == "China").astype(int)
        else:
            keep = (pop == country) & np.isin(site, sites); axis_np = (site == pos).astype(int)
        tr = (split == "train") & keep; te = (split == "test") & keep
        if tr.sum() < 30 or te.sum() < 15:
            print(f"[skip {name}] insufficient N (train {tr.sum()}, test {te.sum()})"); continue
        axis = torch.tensor(axis_np)
        tri = torch.tensor(np.where(tr)[0]); tei = torch.tensor(np.where(te)[0])

        # raw site/disease + age/sex decodability (how confounded is the contrast?)
        site_raw = auc(X[tr], axis_np[tr], X[te], axis_np[te])
        dis_raw = auc(X[tr], ydx[tr], X[te], ydx[te])
        m = np.isfinite(age)
        age_auc = (auc(X[tr & m], (age[tr & m] > np.nanmedian(age[keep])).astype(int),
                       X[te & m], (age[te & m] > np.nanmedian(age[keep])).astype(int))
                   if (tr & m).sum() > 20 and len(np.unique((age[te & m] > np.nanmedian(age[keep])))) > 1 else np.nan)
        # INLP site removal
        P = inlp_P(X[tr], axis_np[tr], args.rounds)
        site_inlp = auc(X[tr] @ P, axis_np[tr], X[te] @ P, axis_np[te])
        dis_inlp = auc(X[tr] @ P, ydx[tr], X[te] @ P, ydx[te])
        # age/sex deconfound: residualize then re-measure (FAST: probe + INLP)
        Xr = residualize_age_sex(X, age, sex, tr)
        site_resid = auc(Xr[tr], axis_np[tr], Xr[te], axis_np[te])
        Pr2 = inlp_P(Xr[tr], axis_np[tr], args.rounds)
        site_resid_inlp = auc(Xr[tr] @ Pr2, axis_np[tr], Xr[te] @ Pr2, axis_np[te])
        dis_resid = auc(Xr[tr], ydx[tr], Xr[te], ydx[te])
        # optional slow per-axis DAS-IIA — only for the within-country contrasts (the new headline;
        # the population axis IIA is already reported via ctrl_das_seeds). Residualized-axis DAS is
        # skipped unless --das-resid (the fast residualized probe/INLP above already deconfounds).
        if args.das and country is not None:
            di = das_iia_axis(h, axis, hW, hb, tri, tei, args.pairs, args.seed, dev, steps=args.das_steps)
            iia, iia_pres = round(di["scanner_iia"], 3), round(di["disease_preserved"], 3)
            if args.das_resid:
                hr = torch.tensor(Xr, dtype=torch.float32)
                iia_r = round(das_iia_axis(hr, axis, hW, hb, tri, tei, args.pairs, args.seed, dev,
                                           steps=args.das_steps)["scanner_iia"], 3)
            else:
                iia_r = "na"
        else:
            iia = iia_pres = iia_r = "na"

        record(contrast=name, n_train=int(tr.sum()), n_test=int(te.sum()),
               site_auc_raw=round(site_raw, 3), site_auc_inlp=round(site_inlp, 3),
               disease_auc_raw=round(dis_raw, 3), disease_auc_inlp=round(dis_inlp, 3),
               age_auc=round(age_auc, 3) if age_auc == age_auc else "na",
               site_auc_resid_agesex=round(site_resid, 3),
               site_auc_resid_then_inlp=round(site_resid_inlp, 3),
               disease_auc_resid=round(dis_resid, 3),
               site_iia_das=iia, disease_pres_das=iia_pres, site_iia_das_resid=iia_r)

    print(f"\nwrote {args.out}")
    print("\nINTERPRETATION: within-country site_auc_raw high + INLP collapses it + DAS site-IIA high"
          "\nwith disease preserved = a clean acquisition axis with NO population confound. If"
          "\nsite_auc_resid_agesex and site_iia_das_resid stay high, the axis is acquisition, not an"
          "\nage/sex proxy. This becomes the manuscript headline; population US-vs-China is the"
          "\nconfounded illustrative case.")


if __name__ == "__main__":
    main()
