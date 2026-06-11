"""Sparse-Feature-Circuit node attribution via attribution patching (Stage 3).

Goal under the redundant-confound thesis: quantify HOW DISTRIBUTED the scanner signal is across
SAE features and layers, and test whether a JOINT graph-ablation of the attributed scanner nodes
moves scanner where single-layer SAE-atom ablation (the prior NULL, RESULTS §6) did not.

Attribution patching (AtP), avoiding the SAE in the forward graph: a feature j at a hook
contributes ~ f_j * W_dec[:,j] to the residual-stream activation `act`. Its indirect effect on a
scalar scanner metric m (a frozen scanner readout on the fused rep) is, to first order,

    IE_j = sum_tokens ( f_j - f_j^ref ) * ( grad_act . W_dec[:,j] )

where grad_act = d m / d act is obtained by ONE backward pass of m w.r.t. each hook activation,
f = sae.encode(act), and f^ref = sae.encode(mean act) is the mean-ablation reference. This yields
every feature's signed importance at every hook from a single backward pass.
"""
from __future__ import annotations

import sys
import torch

_GEO_SRC = "/home/users/ybi3/MultiViT2/geometric_multivit/src"
if _GEO_SRC not in sys.path:
    sys.path.insert(0, _GEO_SRC)


def _resolve(model, dotted: str):
    mod = model
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def node_attribution(model, loader, device, hooks, saes, metric_fn, ref_acts=None,
                     max_batches=None):
    """Per-(hook, feature) signed attribution IE toward the scalar scanner metric.

    model      : frozen MultiViT (params need no grad; we differentiate the metric w.r.t. acts).
    hooks      : list of dotted hook names (each must have a trained SAE in ``saes``).
    saes       : {hook: SparseAutoencoder} (decoder W_dec used for the linear feature->act map).
    metric_fn  : callable(fused_batch) -> (B,) scalar scanner readout to differentiate.
    ref_acts   : {hook: (T,d) mean activation} for the mean-ablation reference f^ref; if None,
                 uses the per-batch mean (cheap, slightly noisier).
    Returns    : {hook: (d_sae,) summed signed IE over all samples/tokens}, and the per-hook
                 absolute totals for the distribution diagnostic.
    """
    saes = {h: s.to(device) for h, s in saes.items()}
    accum = {h: torch.zeros(saes[h].W_dec.shape[0], device=device) for h in hooks}

    # Re-leaf ONE hook at a time: the backward graph then spans only the part of the model
    # downstream of that hook (no giant 3D-input gradient), which keeps memory bounded on the GPU.
    # Correct per-hook (the upstream-detach issue only arises when re-leafing many hooks at once).
    for h in hooks:
        sae = saes[h]
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            box, fused_box = {}, {}

            def releaf(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                t2 = t.detach().requires_grad_(True)               # leaf -> grad reaches it
                box["t"] = t2
                return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2

            def grab(_m, inp):
                fused_box["x"] = inp[0]

            h1 = _resolve(model, h).register_forward_hook(releaf)
            h2 = _resolve(model, "head").register_forward_pre_hook(grab)
            model.zero_grad(set_to_none=True)
            model(batch["sMRI"].to(device), batch["sFNC"].to(device))
            h1.remove(); h2.remove()
            metric_fn(fused_box["x"]).sum().backward()
            act = box["t"].detach()                                # (B,T,d)
            grad = box["t"].grad.detach()                          # (B,T,d)
            f = sae.encode(act)
            if ref_acts is not None and h in ref_acts:
                fref = sae.encode(ref_acts[h].to(device)).unsqueeze(0)
            else:
                fref = sae.encode(act.mean(0, keepdim=True))
            gW = grad @ sae.W_dec.T                                # (B,T,d_sae)
            accum[h] += ((f - fref) * gW).sum(dim=(0, 1))
            del box, act, grad, f, fref, gW
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    totals = {h: accum[h].abs().sum().item() for h in hooks}
    return {h: accum[h].detach().cpu() for h in hooks}, totals


def participation_ratio(weights: torch.Tensor) -> float:
    """Effective number of contributing features: (sum|w|)^2 / sum(w^2). High = distributed,
    low (~1) = localized to a few features. The redundant-confound thesis predicts HIGH PR."""
    w = weights.abs().float()
    s1 = w.sum() ** 2
    s2 = (w ** 2).sum()
    return float((s1 / (s2 + 1e-12)).item())


def top_features(attr: dict, frac_per_hook=0.0, k_per_hook=None, by_abs=True):
    """Select scanner-relevant features per hook for the joint graph-ablation.
    Returns {hook: LongTensor(idx)} of the top features by |IE| (or signed)."""
    sel = {}
    for h, a in attr.items():
        score = a.abs() if by_abs else a
        n = a.numel()
        k = k_per_hook if k_per_hook is not None else max(1, int(frac_per_hook * n))
        sel[h] = torch.topk(score, min(k, n)).indices
    return sel
