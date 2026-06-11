"""Phase 3: causal feature ablation (activation patching) on MultiViT2.

Ablation semantics: at a target hook we subtract the SAE-decoded contribution of a chosen
feature set S from the activation that flows downstream:

    x_new = x - f_S @ W_dec[S]          (f = sae.encode(x); f_S keeps only columns in S)

This removes only the directions those features write, leaving the SAE residual (what the
dictionary cannot capture) untouched. We then read the model's own disease logit and a
downstream fused representation to test necessity/sufficiency causally.
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


def make_ablation_hook(sae, idx: torch.Tensor):
    """Forward hook that subtracts the decoded contribution of features ``idx``.
    Works on (B, T, d) or (B, d) module outputs (encode acts on the last dim)."""
    Wd_S = sae.W_dec[idx]                     # (|S|, d)

    def hook(_m, _i, out):
        x = out[0] if isinstance(out, tuple) else out
        f = sae.encode(x)                     # (..., d_sae)
        delta = f[..., idx] @ Wd_S            # (..., d)
        x_new = x - delta
        if isinstance(out, tuple):
            return (x_new,) + tuple(out[1:])
        return x_new

    return hook


def make_write_hook(value: torch.Tensor, token_idx=None):
    """Forward hook that OVERWRITES a module output with a constant ``value`` (raw-activation
    patching, the denoising/noising paradigm) -- unlike ``make_ablation_hook`` which subtracts a
    self-derived decode, this injects a precomputed activation (e.g. a per-group mean).

    token_idx=None  -> overwrite the whole token sequence; ``value`` is (T, d) or (d,), broadcast
                       over the batch (mean-ablate the module's between-subject variance).
    token_idx=int/list -> overwrite only those token rows; ``value`` is (k, d) or (d,).
    """
    def hook(_m, _i, out):
        x = out[0] if isinstance(out, tuple) else out      # (B, T, d) or (B, d)
        x = x.clone()
        v = value.to(device=x.device, dtype=x.dtype)
        if token_idx is None:
            x[:] = v                                       # broadcasts (T,d)/(d,) over batch
        else:
            x[:, token_idx] = v
        if isinstance(out, tuple):
            return (x,) + tuple(out[1:])
        return x

    return hook


@torch.no_grad()
def capture_hook_tokens(model, loader, device, hook: str, max_batches=None):
    """One forward pass, returning the FULL per-subject token activation at ``hook``
    (N, T, d) plus the subject_id order -- used to compute per-group (population/site) means
    that ``make_write_hook`` then injects. Keeps subject grouping (unlike the (N*T,d) extractor)."""
    buf = []
    def _grab(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        buf.append(t.detach().float().cpu())
    h = _resolve(model, hook).register_forward_hook(_grab)
    sids = []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        model(batch["sMRI"].to(device), batch["sFNC"].to(device))
        sids += list(batch["subject_id"])
    h.remove()
    return torch.cat(buf, 0), sids                          # (N, T, d), [sid...]


@torch.no_grad()
def run_model(model, loader, device, target_hook=None, sae=None, ablate_idx=None,
              capture="head", max_batches=None, ablations=None, extra_hooks=None):
    """Run the model (optionally with intervention at one or many layers) and capture
    the input to ``capture`` (default the classifier head = fused 512-d rep).

    Single-layer SAE ablation: pass target_hook/sae/ablate_idx. Multi-layer: ``ablations`` =
    list of (hook_name, sae, idx). Raw-activation patching (mean-patch / write): ``extra_hooks``
    = list of (hook_name, forward_hook_callable) built with ``make_write_hook`` -- these run in
    addition to any ablations, enabling the cross-attention path-patching crime-scene test.

    Returns dict: logits (N,2), fused (N, d_cap), subject_id (list), y (N,).
    """
    handles = []
    abls = ablations if ablations is not None else (
        [(target_hook, sae, ablate_idx)] if (ablate_idx is not None and len(ablate_idx) > 0) else [])
    for hook, s, idx in abls:
        if idx is None or len(idx) == 0:
            continue
        s = s.to(device)
        handles.append(_resolve(model, hook).register_forward_hook(
            make_ablation_hook(s, torch.as_tensor(idx, device=device))))
    for hook, fn in (extra_hooks or []):
        handles.append(_resolve(model, hook).register_forward_hook(fn))

    cap = {}
    def pre_hook(_m, inp):
        cap.setdefault("x", []).append(inp[0].detach().float().cpu())
    handles.append(_resolve(model, capture).register_forward_pre_hook(pre_hook))

    logits, sids, ys = [], [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        out = model(batch["sMRI"].to(device), batch["sFNC"].to(device))
        logits.append(out.detach().float().cpu())
        sids += list(batch["subject_id"])
        ys.append(batch["label"])
    for h in handles:
        h.remove()
    return {"logits": torch.cat(logits), "fused": torch.cat(cap["x"]),
            "subject_id": sids, "y": torch.cat(ys).numpy()}
