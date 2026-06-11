#!/usr/bin/env python
"""C2 (threat T2) — does the DAS/IIA result survive an honest null, and is the DAS
direction anything more than the linear scanner probe?

The hostile review flagged three things about the headline causal claim (k=1 scanner-IIA
=0.86 on the fused decision rep):
  (a) WRONG NULL. We compared trained-DAS-IIA only to a *random rotation* (~0.13). But the
      mechanistically meaningful reference points are the NO-OP floor (do nothing: hp=h_base
      -> the base subject's scanner, which by construction != source -> IIA~0) and the
      FULL-SWAP ceiling (hp=h_src -> source scanner trivially transfers, but the model's
      disease decision is DESTROYED). 0.86 only means something between those poles.
  (b) CIRCULARITY. Maybe the k=1 DAS "scanner direction" is just the linear scanner-probe
      weight re-discovered, so IIA adds nothing over correlational probing. We measure
      |cos(DAS k=1 direction, probe weight)|. If ~1, DAS recovers the readout; its added
      value is then the *interchange + disease-preservation constraint*, not direction-finding.
  (c) PLACEBO. Train DAS to transfer a balanced RANDOM pseudo-label instead of scanner. A
      genuine causal subspace should NOT exist for noise -> placebo IIA ~ no-op floor.

Everything runs on cached fused_HOLD_ALL.pt + the frozen model head (the cached `fused` IS the
head input, so head(h')=h'@W.T+b is the model's real disease logit). No model forward.
Writes outputs/sae_ckpts/ctrl_das_null.csv.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mib import das as D

MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"


def load_head(ckpt_path):
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    return sd["head.weight"].float(), sd["head.bias"].float()


def make_pairs(label, idx, n_pairs, seed):
    """Cross-label pairs: base has label 0, source label 1 (and the mirror)."""
    g = torch.Generator().manual_seed(seed)
    a = idx[label[idx] == 0]; b = idx[label[idx] == 1]
    if len(a) == 0 or len(b) == 0:
        raise SystemExit("need both label groups present")
    base = torch.cat([a[torch.randint(len(a), (n_pairs,), generator=g)],
                      b[torch.randint(len(b), (n_pairs,), generator=g)]])
    src = torch.cat([b[torch.randint(len(b), (n_pairs,), generator=g)],
                     a[torch.randint(len(a), (n_pairs,), generator=g)]])
    return base, src


@torch.no_grad()
def iia_from_hp(hp, h_base, scanner_src, base_dx, scan_w, scan_b, head_W, head_b):
    """Generic IIA given an already-computed counterfactual hp (B,d)."""
    scan_pred = (hp @ scan_w + scan_b) > 0
    dx_pred = (hp @ head_W.T + head_b).argmax(1)
    iia = (scan_pred.long() == scanner_src).float().mean().item()
    pres = (dx_pred == base_dx).float().mean().item()
    return iia, pres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="outputs/activations/fused_HOLD_ALL.pt")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/sae_ckpts/ctrl_das_null.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    d = torch.load(args.fused, weights_only=True)
    h = d["fused"].float()
    pop = np.asarray(d["population"]); split = np.asarray(d["split"])
    keep = np.isin(pop, ["US", "China"])
    hW, hb = load_head(MHOLD)
    rec_err = ((h @ hW.T + hb) - d["logits"].float()).abs().max().item()
    print(f"head(fused) vs cached logits max|Δ| = {rec_err:.2e} (should be ~0)", flush=True)

    scanner = torch.tensor((pop == "China").astype(int))
    tr = torch.tensor(np.where((split == "train") & keep)[0])
    te = torch.tensor(np.where((split == "test") & keep)[0])
    print(f"N_train={len(tr)} N_test={len(te)}", flush=True)

    # frozen scanner readout (LR on TRAIN fused), as a torch linear
    lr = LogisticRegression(max_iter=3000).fit(h[tr].numpy(), scanner[tr].numpy())
    scan_w = torch.tensor(lr.coef_.ravel(), dtype=torch.float32)
    scan_b = torch.tensor(float(lr.intercept_[0]), dtype=torch.float32)
    probe_auc = roc_auc_score(scanner[te].numpy(), (h[te] @ scan_w + scan_b).numpy())
    print(f"frozen scanner probe held-out AUC = {probe_auc:.3f}", flush=True)

    h = h.to(dev); hW, hb = hW.to(dev), hb.to(dev)
    scan_w, scan_b = scan_w.to(dev), scan_b.to(dev); scanner = scanner.to(dev)

    bt, st = make_pairs(scanner.cpu(), tr, args.pairs, args.seed)
    bv, sv = make_pairs(scanner.cpu(), te, args.pairs, args.seed + 100)
    bt, st, bv, sv = [x.to(dev) for x in (bt, st, bv, sv)]
    base_dx_te = (h[bv] @ hW.T + hb).argmax(1)
    scan_src_te = scanner[sv]

    rows = []

    def record(name, iia, pres, extra=""):
        rows.append({"condition": name, "scanner_iia": round(iia, 4),
                     "disease_preserved": round(pres, 4), "note": extra})
        print(f"{name:28s} IIA={iia:.3f}  disease_pres={pres:.3f}  {extra}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    # (a) reference points -------------------------------------------------
    # no-op floor: hp = h_base (do nothing)
    i, p = iia_from_hp(h[bv], h[bv], scan_src_te, base_dx_te, scan_w, scan_b, hW, hb)
    record("noop_floor", i, p, "hp=h_base (transfer nothing)")
    # full-swap ceiling: hp = h_src (k=d, disease must collapse)
    i, p = iia_from_hp(h[sv], h[bv], scan_src_te, base_dx_te, scan_w, scan_b, hW, hb)
    record("full_swap_ceiling", i, p, "hp=h_src (disease destroyed)")
    # random rotation k=1
    rnd = D.DASRotation(h.shape[1], k=1, seed=args.seed + 7).to(dev)
    hp = rnd.interchange(h[bv], h[sv])
    i, p = iia_from_hp(hp, h[bv], scan_src_te, base_dx_te, scan_w, scan_b, hW, hb)
    record("random_rotation_k1", i, p, "swap 1 random coord")

    # (b) trained DAS k=1 + cosine vs probe --------------------------------
    das = D.train_das(h, scanner, hW, hb, scan_w, scan_b, bt, st, k=1,
                      steps=args.steps, lr=5e-3, lam=1.0, seed=args.seed, device=dev)
    m = D.eval_iia(das, h, scanner, hW, hb, scan_w, scan_b, bv, sv, device=dev)
    # the swapped subspace for k=1 is span(R.weight[0]); compare to probe direction
    das_dir = das._W()[0].detach().cpu().numpy()
    pw = scan_w.detach().cpu().numpy()
    cos = abs(float(das_dir @ pw / (np.linalg.norm(das_dir) * np.linalg.norm(pw) + 1e-12)))
    record("trained_das_k1", m["scanner_iia"], m["disease_preserved"],
           f"|cos(DAS_dir, probe_w)|={cos:.3f}")

    # (c) placebo DAS: transfer a balanced RANDOM pseudo-label -------------
    g = torch.Generator().manual_seed(args.seed + 31)
    placebo = torch.zeros(len(scanner), dtype=torch.long)
    perm = torch.randperm(len(tr), generator=g)
    placebo[tr[perm[: len(tr) // 2]]] = 1                       # balanced on train
    perm2 = torch.randperm(len(te), generator=g)
    placebo[te[perm2[: len(te) // 2]]] = 1                       # balanced on test
    placebo = placebo.to(dev)
    pbt, pst = make_pairs(placebo.cpu(), tr, args.pairs, args.seed + 5)
    pbv, psv = make_pairs(placebo.cpu(), te, args.pairs, args.seed + 105)
    pbt, pst, pbv, psv = [x.to(dev) for x in (pbt, pst, pbv, psv)]
    # frozen "placebo readout" on train fused
    plr = LogisticRegression(max_iter=2000).fit(h[tr].cpu().numpy(), placebo[tr].cpu().numpy())
    pw_w = torch.tensor(plr.coef_.ravel(), dtype=torch.float32, device=dev)
    pw_b = torch.tensor(float(plr.intercept_[0]), dtype=torch.float32, device=dev)
    das_p = D.train_das(h, placebo, hW, hb, pw_w, pw_b, pbt, pst, k=1,
                        steps=args.steps, lr=5e-3, lam=1.0, seed=args.seed, device=dev)
    mp = D.eval_iia(das_p, h, placebo, hW, hb, pw_w, pw_b, pbv, psv, device=dev)
    record("placebo_das_k1", mp["scanner_iia"], mp["disease_preserved"],
           "transfer random pseudo-label (expect ~floor)")

    print(f"\nwrote {args.out}")
    print("\nINTERPRETATION: trained-DAS IIA should sit far above the no-op floor and the random"
          "\nrotation, while preserving disease (unlike the full-swap ceiling). High |cos| means DAS"
          "\nre-finds the linear scanner axis -> its novelty is the interchange+disease-preservation"
          "\ntest on the REAL decision, not direction discovery. Placebo near floor = no spurious"
          "\ntransfer of arbitrary labels.")


if __name__ == "__main__":
    main()
