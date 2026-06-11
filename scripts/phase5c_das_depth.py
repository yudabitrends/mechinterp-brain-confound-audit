#!/usr/bin/env python
"""#2 (reviewers P2/P7) — TRAINED forward-propagated DAS at each hook depth: makes the
'causal handle crystallizes at the decision representation' claim rigorous.

The untrained early-hook interchange (phase5b: 0.35 at sFNC_encoder.norm) could be weak simply
because it is untrained. Here we TRAIN a 1-D direction u at each hook by gradient THROUGH the frozen
model: interchange the component along u (source->base) at the hook, propagate forward, and optimize u
so the downstream scanner readout flips to source while the downstream disease decision (head) is
preserved. We then report downstream scanner-IIA-vs-depth for the trained u vs the untrained probe u.

If even a TRAINED early-hook direction gives lower downstream IIA than the decision-rep DAS (0.89), the
causal handle genuinely consolidates with depth; the trained-vs-untrained confound is removed.

Memory: model params are frozen and u is the only grad leaf (introduced at the hook), so autograd builds
the graph only from the hook onward. GPU. Writes outputs/sae_ckpts/phase5c_das_depth.csv.
"""
import json, os, sys, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from geomultivit.data.dataset import MultiModalH5Dataset
from geomultivit.data.multicohort import load_geometric_cohort
from mib.extract import load_model
from mib import patch

H5 = "/home/users/ybi3/data/szdataset_modified.h5"
MHOLD = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/best.pt"
SPLIT = "/home/users/ybi3/MultiViT2/outputs/p6f_custom_split/split_info.json"
# depth-ordered hooks (early -> decision)
HOOKS = ["sFNC_encoder.blocks.0", "sFNC_encoder.blocks.1", "sFNC_encoder.blocks.2",
         "sFNC_encoder.blocks.3", "sFNC_encoder.norm", "cross_blocks.0.ca_b", "cross_blocks.1.ca_b"]


def loader_for(gdf, ids, bs=8):
    sub = gdf[gdf.SubjectID.isin(set(ids))].drop_duplicates("SubjectID").reset_index(drop=True)
    ds = MultiModalH5Dataset(sub, (96, 112, 96), 53, require_sMRI=True, cache_dir="outputs/cache")
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4), sub


class InterchangeHook:
    """Forward hook replacing CLS token t=0 with a 1-D interchange along u, consumed in loader order.
    src_cls: (N,d) source CLS in loader order; u: trainable (d,). i advances per batch, reset per pass."""
    def __init__(self, src_cls, u):
        self.src_cls, self.u, self.i = src_cls, u, 0
    def reset(self):
        self.i = 0
    def __call__(self, _m, _i, out):
        x = out[0] if isinstance(out, tuple) else out
        b = x.shape[0]
        s = self.src_cls[self.i:self.i + b].to(x.device, x.dtype); self.i += b
        un = self.u / (self.u.norm() + 1e-8)
        proj_b = (x[:, 0] * un).sum(-1, keepdim=True)
        proj_s = (s * un).sum(-1, keepdim=True)
        x = x.clone()
        x[:, 0] = x[:, 0] + (proj_s - proj_b) * un
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120); ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--max-train", type=int, default=400)
    ap.add_argument("--out", default="outputs/sae_ckpts/phase5c_das_depth.csv")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)

    si = json.load(open(SPLIT))
    gdf = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False).df
    gdf = gdf[gdf.cohort.isin(["COBRE", "FBIRN", "ChineseSZ", "PK_MPRC"])]
    model = load_model(MHOLD, device=dev)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    # capture the head input (fused decision rep) each forward, WITH grad
    store = {}
    model.head.register_forward_pre_hook(lambda m, inp: store.__setitem__("f", inp[0]))

    tr_loader, tr_sub = loader_for(gdf, si["train_ids"])
    te_loader, te_sub = loader_for(gdf, si["test_ids"])

    def pop_of(sub, sids):
        m = sub.drop_duplicates("SubjectID").set_index("SubjectID")
        return np.array([m.loc[s, "population"] for s in sids])

    # baseline (no grad): fused + logits in loader order
    base_tr = patch.run_model(model, tr_loader, dev)
    base_te = patch.run_model(model, te_loader, dev)
    Ftr = base_tr["fused"].numpy(); pop_tr = pop_of(tr_sub, base_tr["subject_id"])
    pop_te = pop_of(te_sub, base_te["subject_id"]); base_dx_te = base_te["logits"].argmax(1).numpy()
    yscan_tr = (pop_tr == "China").astype(int); yscan_te = (pop_te == "China").astype(int)
    keep_tr = np.isin(pop_tr, ["US", "China"]); keep_te = np.isin(pop_te, ["US", "China"])
    # frozen scanner readout on train fused (de-standardized torch linear)
    sc = StandardScaler().fit(Ftr[keep_tr])
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(Ftr[keep_tr]), yscan_tr[keep_tr])
    w = torch.tensor(lr.coef_.ravel() / sc.scale_, dtype=torch.float32, device=dev)
    b = torch.tensor(float(lr.intercept_[0] - (lr.coef_.ravel() * sc.mean_ / sc.scale_).sum()),
                     dtype=torch.float32, device=dev)

    g = np.random.default_rng(0)

    def pair_src(yscan, keep, n):
        us = np.where((yscan == 0) & keep)[0]; ch = np.where((yscan == 1) & keep)[0]
        src = np.zeros(n, int)
        for i in range(n):
            if not keep[i]:
                src[i] = i
            else:
                src[i] = g.choice(ch) if yscan[i] == 0 else g.choice(us)
        return src

    src_te = pair_src(yscan_te, keep_te, len(yscan_te)); src_scan_te = 1 - yscan_te

    rows = []
    for di, H in enumerate(HOOKS):
        # per-subject CLS at H (baseline), loader order
        cls_tr = patch.capture_hook_tokens(model, tr_loader, dev, H)[0][:, 0]   # (Ntr,d) cpu
        cls_te = patch.capture_hook_tokens(model, te_loader, dev, H)[0][:, 0]
        dH = cls_tr.shape[1]
        # init u from linear scanner probe at this hook
        scs = StandardScaler().fit(cls_tr[keep_tr].numpy())
        lrh = LogisticRegression(max_iter=2000).fit(scs.transform(cls_tr[keep_tr].numpy()), yscan_tr[keep_tr])
        u0 = lrh.coef_.ravel() / scs.scale_; u0 = u0 / np.linalg.norm(u0)
        u = nn.Parameter(torch.tensor(u0, dtype=torch.float32, device=dev))
        opt = torch.optim.Adam([u], lr=args.lr)

        # training source pairing (train subjects)
        src_tr = pair_src(yscan_tr, keep_tr, len(yscan_tr))
        src_cls_tr = cls_tr[src_tr]                                  # (Ntr,d) loader order
        src_scan_tr = torch.tensor(1 - yscan_tr, dtype=torch.float32, device=dev)
        base_dx_tr = torch.tensor(base_tr["logits"].argmax(1).numpy(), device=dev)

        hook_obj = InterchangeHook(src_cls_tr, u)
        handle = patch._resolve(model, H).register_forward_hook(hook_obj)
        step = 0
        while step < args.steps:
            hook_obj.reset(); pos = 0
            for batch in tr_loader:
                if step >= args.steps or pos >= args.max_train:
                    break
                b_ = batch["sMRI"].shape[0]
                lo, hi = pos, pos + b_; pos += b_
                km = torch.tensor(keep_tr[lo:hi], device=dev, dtype=torch.bool)
                if km.sum() == 0:
                    continue
                opt.zero_grad()
                _ = model(batch["sMRI"].to(dev), batch["sFNC"].to(dev))
                fused = store["f"]                                  # (b,512) grad
                scan_logit = (fused @ w + b)[km]
                dx_logit = model.head(fused)[km]
                tgt_scan = src_scan_tr[lo:hi][km]
                tgt_dx = base_dx_tr[lo:hi][km]
                loss = F.binary_cross_entropy_with_logits(scan_logit, tgt_scan) \
                    + args.lam * F.cross_entropy(dx_logit, tgt_dx)
                loss.backward(); opt.step(); step += 1
        handle.remove()

        # eval downstream IIA on test (no grad): trained u, and untrained probe u0
        @torch.no_grad()
        def eval_iia(u_eval):
            hk = InterchangeHook(cls_te[src_te], torch.tensor(u_eval, dtype=torch.float32, device=dev)
                                 if not torch.is_tensor(u_eval) else u_eval.detach())
            hh = patch._resolve(model, H).register_forward_hook(hk)
            r = patch.run_model(model, te_loader, dev)
            hh.remove()
            F_ = r["fused"].to(dev); scan_pred = ((F_ @ w + b) > 0).cpu().numpy(); dx = r["logits"].argmax(1).numpy()
            iia = float((scan_pred[keep_te] == src_scan_te[keep_te]).mean())
            pres = float((dx[keep_te] == base_dx_te[keep_te]).mean())
            return iia, pres
        iia_tr, pres_tr = eval_iia(u)
        iia_un, pres_un = eval_iia(u0)
        rows.append({"depth_idx": di, "hook": H, "iia_trained": round(iia_tr, 3),
                     "disease_pres_trained": round(pres_tr, 3), "iia_untrained": round(iia_un, 3)})
        print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(args.out, index=False)

    print(f"\nwrote {args.out}")
    print("INTERPRETATION: if iia_trained at early hooks stays well below the decision-rep 0.89 (with"
          " disease preserved), the causal handle genuinely crystallizes with depth -- the trained-vs-"
          "untrained confound is removed.")


if __name__ == "__main__":
    main()
