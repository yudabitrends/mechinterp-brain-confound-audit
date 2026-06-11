"""Sparse autoencoders for MultiViT2 activation dictionaries.

Self-contained (no sae_lens dependency) because MultiViT2 is a custom 3D-ViT +
FNC-transformer, not a HookedTransformer language model. We keep the SAELens
*conventions* though:

- pre-encoder bias subtraction (x - b_dec), tied init, unit-norm decoder columns
  (so the L1 penalty on feature activations is scale-meaningful), as in Anthropic's
  "Towards Monosemanticity".
- two architectures: ``standard`` (ReLU + L1) and ``topk`` (exactly k active features,
  no L1 tuning -- usually the easier, more stable choice on small medical-N data).

Activations fed in are per-token vectors of dim ``d_in`` (256 for the ViT/FNC branch
blocks, 512 for the fused head input). See ``mib.extract`` for how they are produced.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_in: int
    expansion: int = 8          # d_sae = expansion * d_in  (Anthropic uses 4-16x)
    architecture: str = "topk"  # "standard" | "topk"
    k: int = 32                 # topk: active features per sample
    l1_coef: float = 1e-3       # standard: sparsity penalty
    seed: int = 0

    @property
    def d_sae(self) -> int:
        return self.expansion * self.d_in


class SparseAutoencoder(nn.Module):
    """Single-hidden-layer SAE with a tied-init, unit-norm decoder.

    forward(x) -> (x_hat, features). Use ``loss(...)`` for the training objective and
    ``mib.metrics`` for L0 / explained-variance / dead-feature reporting.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        d_in, d_sae = cfg.d_in, cfg.d_sae

        # Decoder columns initialised on the unit sphere; encoder tied to its transpose.
        W_dec = torch.randn(d_sae, d_in, generator=g)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    # -- core -------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        if self.cfg.architecture == "topk":
            return self._topk(F.relu(pre))
        return F.relu(pre)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        return self.decode(f), f

    def _topk(self, f: torch.Tensor) -> torch.Tensor:
        k = min(self.cfg.k, f.shape[-1])
        vals, idx = f.topk(k, dim=-1)
        out = torch.zeros_like(f)
        return out.scatter_(-1, idx, vals)

    # -- training objective ----------------------------------------------
    def loss(self, x: torch.Tensor):
        """Return (total_loss, parts_dict).

        MSE reconstruction + (standard only) L1 weighted by decoder-column norm so the
        penalty is invariant to feature rescaling. TopK needs no L1 -- sparsity is
        structural. ``l1_weight`` lets the caller warm the penalty up from 0.
        """
        x_hat, f = self.forward(x)
        mse = F.mse_loss(x_hat, x)
        if self.cfg.architecture == "standard":
            dec_norm = self.W_dec.norm(dim=1)                  # (d_sae,)
            l1 = (f.abs() * dec_norm).sum(-1).mean()
            total = mse + self.cfg.l1_coef * l1
            return total, {"mse": mse.detach(), "l1": l1.detach()}
        return mse, {"mse": mse.detach(), "l1": torch.zeros((), device=x.device)}

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Project decoder columns back to the unit sphere (call after each step for
        the standard architecture; harmless for topk)."""
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))
