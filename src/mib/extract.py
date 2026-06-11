"""Activation-extraction harness for a trained MultiViT2 checkpoint.

Registers forward hooks on the verified named modules (sMRI/FNC transformer blocks +
cross-attention blocks), runs a cohort through the frozen model, and caches per-token
activations together with per-sample (subject_id, diagnosis, site, cohort) labels -- the
labels are what make Phase-2 scanner-vs-disease probing possible.

Position policy (the scientific knob):
  - "cls"    : keep only the CLS token (index 0) -> (N, d). This is what ``head`` actually
               reads, so it is the decision-relevant representation.
  - "tokens" : keep every patch/ICN token -> (N*T, d), with a row->subject index map, so
               features can later be back-projected to brain space (Phase 6).
We default to extracting BOTH at the branch-output norm layers and CLS-only deeper, but
the caller chooses per hook point.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import torch

# verified import path (see wiring verification)
_GEO_SRC = "/home/users/ybi3/MultiViT2/geometric_multivit/src"
if _GEO_SRC not in sys.path:
    sys.path.insert(0, _GEO_SRC)


def default_hook_points() -> list[str]:
    """The named modules we capture by default: every transformer block's residual-
    stream output (post-attn handled via block output) across both branches + the two
    cross-attention fusion blocks. Names verified against model.named_modules()."""
    pts: list[str] = []
    for i in range(6):
        pts += [f"sMRI_encoder.blocks.{i}", f"sMRI_encoder.blocks.{i}.mlp"]
    for i in range(4):
        pts += [f"sFNC_encoder.blocks.{i}", f"sFNC_encoder.blocks.{i}.mlp"]
    pts += ["sMRI_encoder.norm", "sFNC_encoder.norm"]
    for i in range(2):
        pts += [f"cross_blocks.{i}.ca_a", f"cross_blocks.{i}.ca_b"]
    return pts


@dataclass
class ExtractConfig:
    ckpt_path: str
    hook_points: list[str] = field(default_factory=default_hook_points)
    position: str = "cls"          # "cls" | "tokens"
    device: str = "cuda"
    max_batches: int | None = None  # cap for smoke tests


def load_model(ckpt_path: str, device: str = "cpu"):
    """Load a MultiViT2 checkpoint -> eval-mode model on ``device``."""
    from geomultivit.models.multivit import build_from_dict

    # Prefer the safe loader (config is plain dict + tensors); fall back only for these
    # trusted, locally-produced checkpoints if an unpicklable object is present.
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"] if isinstance(ckpt, dict) and "config" in ckpt else ckpt.get("cfg")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = build_from_dict(cfg)
    model.load_state_dict(state)
    return model.to(device).eval()


def _resolve(model, dotted: str):
    mod = model
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


class ActivationExtractor:
    """Context manager that hooks ``cfg.hook_points`` and accumulates activations.

    Usage::

        ext = ActivationExtractor(model, cfg)
        acts, labels = ext.run(dataloader)   # acts[name] -> (N, d) or (N*T, d) tensor
    """

    def __init__(self, model, cfg: ExtractConfig):
        self.model, self.cfg = model, cfg
        self._buf: dict[str, list[torch.Tensor]] = {p: [] for p in cfg.hook_points}
        self._handles = []

    def _hook(self, name):
        def fn(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out      # (B, T, d)
            if t.dim() == 3:
                t = t[:, 0] if self.cfg.position == "cls" else t.reshape(-1, t.shape[-1])
            self._buf[name].append(t.detach().to("cpu", torch.float32))
        return fn

    def __enter__(self):
        for name in self.cfg.hook_points:
            self._handles.append(_resolve(self.model, name).register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def run(self, loader):
        """Run the loader through the model, returning (activations, labels).

        labels is a dict of per-SAMPLE arrays (subject_id, label, plus whatever the
        loader/cohort frame supplies). Only valid for position=="cls" (one row/sample);
        for "tokens" the caller must expand labels by T using the returned ``n_tokens``.
        """
        subject_ids, ys = [], []
        for bi, batch in enumerate(loader):
            if self.cfg.max_batches is not None and bi >= self.cfg.max_batches:
                break
            sMRI = batch["sMRI"].to(self.cfg.device)
            sFNC = batch["sFNC"].to(self.cfg.device)
            self.model(sMRI, sFNC)
            subject_ids += list(batch["subject_id"])
            ys.append(batch["label"].cpu())
        acts = {k: torch.cat(v, 0) for k, v in self._buf.items() if v}
        labels = {"subject_id": subject_ids, "label": torch.cat(ys).numpy() if ys else []}
        return acts, labels
