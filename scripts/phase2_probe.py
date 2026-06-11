#!/usr/bin/env python
"""Phase 2: scanner-vs-disease probing of SAE features across all hooks (depth profile).

For each hook: apply the trained SAE to CLS activations, then linear-probe the features
for diagnosis, population (US vs China), and site identity; repeat on raw activations as
baseline; and measure the disjointness (top-feature Jaccard) of disease- vs scanner-tuned
features. Writes outputs/sae_ckpts/phase2_probe_<tag>_seed<S>.csv.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mib.sae import SAEConfig, SparseAutoencoder
from mib import probe


def load_sae(path):
    d = torch.load(path, weights_only=True, map_location="cpu")
    sae = SparseAutoencoder(SAEConfig(**d["cfg"]))
    sae.load_state_dict(d["state_dict"])
    return sae.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls-acts", default="outputs/activations/acts_ALL_cls.pt")
    ap.add_argument("--labels", default="outputs/activations/labels_ALL.csv")
    ap.add_argument("--sae-dir", default="outputs/sae_ckpts")
    ap.add_argument("--tag", default="ALLtok")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-site-n", type=int, default=30)
    ap.add_argument("--hooks", nargs="*", default=None)
    ap.add_argument("--split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    lab = pd.read_csv(args.labels)
    acts = torch.load(args.cls_acts, weights_only=True)
    hooks = args.hooks or sorted(acts.keys())

    # alignment invariant: row i of activations == subject i of labels
    n0 = next(iter(acts.values())).shape[0]
    assert n0 == len(lab), f"row mismatch acts({n0}) vs labels({len(lab)}) -- concat order broken"

    if args.split != "all" and "split" in lab.columns:
        keep = (lab["split"] == args.split).to_numpy()
        lab = lab[keep].reset_index(drop=True)
        acts = {k: v[keep] for k, v in acts.items()}
        print(f"[split={args.split}] kept {keep.sum()} subjects", flush=True)

    y_dx = lab["label"].to_numpy()
    pop_mask = lab["population"].isin(["US", "China"]).to_numpy()
    y_pop = (lab["population"] == "China").astype(int).to_numpy()
    vc = lab["site"].value_counts()
    keep_sites = vc[vc >= args.min_site_n].index
    site_mask = lab["site"].isin(keep_sites).to_numpy()
    y_site = lab["site"].to_numpy()
    print(f"N={len(lab)} | dx {np.bincount(y_dx)} | pop(US/CN) {pop_mask.sum()} "
          f"| sites>= {args.min_site_n}: {list(keep_sites)}", flush=True)

    rows = []
    for hook in hooks:
        sae_path = f"{args.sae_dir}/sae_{args.tag}_{hook}_seed{args.seed}.pt"
        if not os.path.exists(sae_path):
            continue
        Xr = acts[hook].numpy()
        F = probe.encode_features(load_sae(sae_path), Xr)
        live = F.std(0) > 1e-9
        Fl = F[:, live]
        r = {"hook": hook, "n_live": int(live.sum()), "d_sae": F.shape[1]}
        try:
            r["dx_auc_sae"] = probe.linear_probe_auc(Fl, y_dx)
            r["dx_auc_raw"] = probe.linear_probe_auc(Xr, y_dx)
            r["pop_auc_sae"] = probe.linear_probe_auc(Fl[pop_mask], y_pop[pop_mask])
            r["pop_auc_raw"] = probe.linear_probe_auc(Xr[pop_mask], y_pop[pop_mask])
            r["site_auc_sae"] = probe.linear_probe_auc(Fl[site_mask], y_site[site_mask], multiclass=True)
            # disjointness of disease- vs scanner(population)-tuned features
            a_dx = probe.per_feature_auc(Fl, y_dx)
            a_pop = probe.per_feature_auc(Fl[pop_mask], y_pop[pop_mask])
            r["jaccard_dx_pop_top5pct"] = probe.top_feature_jaccard(a_dx, a_pop, 0.05)["jaccard"]
        except Exception as e:
            r["error"] = repr(e)
        rows.append(r)
        print(f"{hook:34s} live={r['n_live']:4d} "
              f"dx_sae={r.get('dx_auc_sae',float('nan')):.3f} "
              f"pop_sae={r.get('pop_auc_sae',float('nan')):.3f} "
              f"site_sae={r.get('site_auc_sae',float('nan')):.3f} "
              f"J(dx,pop)={r.get('jaccard_dx_pop_top5pct',float('nan')):.3f}", flush=True)

    out = f"{args.sae_dir}/phase2_probe_{args.tag}_seed{args.seed}{args.out_suffix}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
