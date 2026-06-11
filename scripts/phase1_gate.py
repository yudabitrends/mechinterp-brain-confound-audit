#!/usr/bin/env python
"""Phase 1 acceptance gate: how many SAE features replicate across seeds, per hook.

For each hook, load its per-seed SAEs, run feature_stability on every seed pair
(Hungarian + mutual-NN), and report n_stable. Features that survive are the only ones
we carry into Phase-2/3. Writes outputs/sae_ckpts/phase1_gate.csv.
"""
import argparse, collections, glob, os, sys
from itertools import combinations
import pandas as pd, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mib import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", default="outputs/sae_ckpts")
    ap.add_argument("--tag", default="ALLtok")
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()

    files = glob.glob(f"{args.sae_dir}/sae_{args.tag}_*_seed*.pt")
    byhook = collections.defaultdict(dict)
    for f in files:
        base = os.path.basename(f)[len(f"sae_{args.tag}_"):-3]   # strip prefix + ".pt"
        hook, seed = base.rsplit("_seed", 1)
        byhook[hook][int(seed)] = f

    rows = []
    for hook, seedmap in sorted(byhook.items()):
        seeds = sorted(seedmap)
        Wd = {s: torch.load(seedmap[s], weights_only=True, map_location="cpu")["state_dict"]["W_dec"]
              for s in seeds}
        d_sae = Wd[seeds[0]].shape[0]
        ns = []
        for a, b in combinations(seeds, 2):
            s = metrics.feature_stability(Wd[a], Wd[b], threshold=args.threshold)
            stable_cos = s["matched_cosine"][s["is_stable"]]
            rows.append({"hook": hook, "pair": f"{a}-{b}", "d_sae": d_sae,
                         "n_stable": s["n_stable"], "frac_stable": s["frac_stable"],
                         "median_stable_cos": float(stable_cos.median()) if s["n_stable"] else float("nan")})
            ns.append(s["n_stable"])
        mean_ns = sum(ns) / len(ns)
        print(f"{hook:36s} n_stable(pairs)={ns} mean={mean_ns:5.0f} / {d_sae}", flush=True)

    df = pd.DataFrame(rows)
    out = f"{args.sae_dir}/phase1_gate_{args.tag}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(df)} pair-rows)")
    summ = df.groupby("hook")["n_stable"].mean().sort_values(ascending=False)
    print("\n=== mean stable features per hook (top/bottom) ===")
    print(summ.head(6).to_string())
    print("...")
    print(summ.tail(4).to_string())


if __name__ == "__main__":
    main()
