#!/usr/bin/env python
"""Stage 1 — cross-attention CRIME SCENE (raw-activation patching on M_hold).

Hypothesis: scanner identity (a structural-modality property) LEAKS INTO the disease branch
through the FNC<-sMRI fusion edge `ca_b` (output = the cross-modal contribution residually added
to the FNC stream). We mean-ablate that contribution (replace each subject's ca_b output by the
grand mean = remove between-subject structural injection) and ask whether scanner becomes
UN-decodable from the FUNCTIONAL half of the fused rep (fused[:,256:]) at preserved disease --
all on MODEL-UNSEEN held-out subjects (M_hold split=='test', non-ceiling disease ~0.81-0.84).

Headline metric = scanner(pop) AUC from the FNC half. The sMRI half (the scanner machine) is the
built-in negative-control axis (should stay high regardless); `ca_a` (sMRI<-FNC) is the
direction control op. Necessity = ca_b ablation drops FNC-half scanner; ca_a flat. We also report
a noising SUFFICIENCY shift: inject the opposite-population ca_b mean and watch the FNC-half
scanner probability move toward the injected population.

Writes outputs/sae_ckpts/phase4_patch_fusion.csv (+ _noise.csv). All causal numbers held-out.
"""
import argparse, os, sys, json
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from geomultivit.data.dataset import MultiModalH5Dataset
from geomultivit.data.multicohort import load_geometric_cohort
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from mib.extract import load_model
from mib import patch

H5 = "/home/users/ybi3/data/szdataset_modified.h5"
MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"
SPLIT = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/split_info.json"
CA_HOOKS = ["cross_blocks.0.ca_a", "cross_blocks.0.ca_b",
            "cross_blocks.1.ca_a", "cross_blocks.1.ca_b"]


def build_loader():
    full = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False)
    sub = full.df[full.df["cohort"].isin(["COBRE", "FBIRN", "ChineseSZ", "PK_MPRC"])].copy()
    ds = MultiModalH5Dataset(sub, volume_shape=(96, 112, 96), num_icns=53,
                             require_sMRI=True, cache_dir="outputs/cache", augment=False)
    return DataLoader(ds, batch_size=16, shuffle=False, num_workers=4), sub


@torch.no_grad()
def capture_many(model, loader, device, hooks):
    """One forward pass, returning {hook: (N,T,d)} + subject_id order."""
    bufs = {h: [] for h in hooks}
    handles = [patch._resolve(model, h).register_forward_hook(
        (lambda hh: (lambda _m, _i, out: bufs[hh].append(
            (out[0] if isinstance(out, tuple) else out).detach().float().cpu())))(h))
        for h in hooks]
    sids = []
    for batch in loader:
        model(batch["sMRI"].to(device), batch["sFNC"].to(device))
        sids += list(batch["subject_id"])
    for hd in handles:
        hd.remove()
    return {h: torch.cat(v, 0) for h, v in bufs.items()}, sids


def auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))


def panel(fused, logits, ypop, ydx, tr, te, tag):
    """Held-out probe panel on a patched fused rep. sMRI half = [:256], FNC half = [256:]."""
    F = fused.numpy()
    return {
        "condition": tag,
        "scanner_fnc": auc(F[tr, 256:], ypop[tr], F[te, 256:], ypop[te]),   # HEADLINE
        "scanner_smri": auc(F[tr, :256], ypop[tr], F[te, :256], ypop[te]),  # control axis (machine)
        "scanner_full": auc(F[tr], ypop[tr], F[te], ypop[te]),
        "disease_fnc": auc(F[tr, 256:], ydx[tr], F[te, 256:], ydx[te]),
        "disease_full": auc(F[tr], ydx[tr], F[te], ydx[te]),
        "disease_logits": float(roc_auc_score(ydx[te], (logits[:, 1] - logits[:, 0]).numpy()[te])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/sae_ckpts")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(MHOLD, device=dev)
    si = json.load(open(SPLIT))
    split_map = {**{s: "train" for s in si["train_ids"]}, **{s: "test" for s in si["test_ids"]}}
    loader, sub = build_loader()

    # ---- one capture pass -> grand + per-population means of the 4 cross-modal contributions ----
    acts, sids = capture_many(model, loader, dev, CA_HOOKS)
    dmeta = sub.drop_duplicates("SubjectID").set_index("SubjectID")
    pop = np.array([dmeta.loc[s, "population"] if s in dmeta.index else "NA" for s in sids])
    split = np.array([split_map.get(s, "NA") for s in sids])
    keep = np.isin(pop, ["US", "China"])
    grand = {h: acts[h].mean(0) for h in CA_HOOKS}                      # (T,d)
    popmean = {h: {p: acts[h][pop == p].mean(0) for p in ["US", "China"]} for h in CA_HOOKS}

    # ---- labels aligned to model order (run_model preserves loader order = sids) ----
    base = patch.run_model(model, loader, dev)
    assert base["subject_id"] == sids, "order mismatch"
    ydx = base["y"].astype(int)
    ypop = (pop == "China").astype(int)
    tr = (split == "train") & keep
    te = (split == "test") & keep
    print(f"N_train={tr.sum()} N_test={te.sum()} (model-unseen)")

    rows = [panel(base["fused"], base["logits"], ypop, ydx, tr, te, "baseline")]

    def mean_ablate(hooks):
        return [(h, patch.make_write_hook(grand[h])) for h in hooks]

    conds = {
        "ablate_ca_b@0": ["cross_blocks.0.ca_b"],
        "ablate_ca_b@1": ["cross_blocks.1.ca_b"],
        "ablate_ca_b@both": ["cross_blocks.0.ca_b", "cross_blocks.1.ca_b"],
        "ablate_ca_a@both(ctrl)": ["cross_blocks.0.ca_a", "cross_blocks.1.ca_a"],
    }
    for tag, hooks in conds.items():
        r = patch.run_model(model, loader, dev, extra_hooks=mean_ablate(hooks))
        rows.append(panel(r["fused"], r["logits"], ypop, ydx, tr, te, tag))

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    out = f"{args.out}/phase4_patch_fusion.csv"
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    b, a = df.iloc[0], df[df.condition == "ablate_ca_b@both"].iloc[0]
    c = df[df.condition == "ablate_ca_a@both(ctrl)"].iloc[0]
    print(f"\nNECESSITY  FNC-half scanner: baseline {b.scanner_fnc:.3f} "
          f"-> ca_b-sever {a.scanner_fnc:.3f} (Δ {a.scanner_fnc-b.scanner_fnc:+.3f}) | "
          f"ca_a-ctrl {c.scanner_fnc:.3f} | disease(logits) {b.disease_logits:.3f}->{a.disease_logits:.3f}")
    print(f"sMRI-half scanner (machine, should stay high): {b.scanner_smri:.3f}->{a.scanner_smri:.3f}")
    print(f"wrote {out}")

    # ---- noising SUFFICIENCY: inject opposite-population ca_b mean, measure FNC-half China-prob shift ----
    Ffnc = StandardScaler().fit(base["fused"].numpy()[tr, 256:])
    clf = LogisticRegression(max_iter=2000).fit(Ffnc.transform(base["fused"].numpy()[tr, 256:]), ypop[tr])
    def china_prob(fused):
        return clf.predict_proba(Ffnc.transform(fused.numpy()[:, 256:]))[:, 1]
    p0 = china_prob(base["fused"])
    noise_rows = []
    for target, src in [("US->China", "China"), ("China->US", "US")]:
        hooks = [(h, patch.make_write_hook(popmean[h][src]))
                 for h in ["cross_blocks.0.ca_b", "cross_blocks.1.ca_b"]]
        r = patch.run_model(model, loader, dev, extra_hooks=hooks)
        p1 = china_prob(r["fused"])
        grp = te & (pop == target.split("->")[0])
        noise_rows.append({"shift": target, "n": int(grp.sum()),
                           "china_prob_base": float(p0[grp].mean()),
                           "china_prob_noised": float(p1[grp].mean()),
                           "delta": float((p1 - p0)[grp].mean())})
    ndf = pd.DataFrame(noise_rows)
    ndf.to_csv(f"{args.out}/phase4_patch_fusion_noise.csv", index=False)
    print("\nSUFFICIENCY (noising, held-out, FNC-half China-prob):")
    print(ndf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
