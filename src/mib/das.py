"""Distributed Alignment Search (DAS) + Interchange Intervention Accuracy (IIA).

Causal-abstraction faithfulness for the scanner confound, replacing correlational probing
(readout |cos|) and the mechanism-agnostic INLP knob with a *measured* causal number.

Setup (operating on the fused decision rep h = head input, so the model's real disease output
is head(h)): we posit the high-level scanner variable S occupies a k-dim subspace of h, found by
a learned orthogonal rotation R. An INTERCHANGE INTERVENTION takes a base subject (scanner=US)
and a source subject (scanner=China), swaps the first k rotated coordinates from source into base,
rotates back -> h'. The causal-abstraction hypothesis holds iff, on h', a frozen scanner readout
flips to the SOURCE scanner while the model's own disease decision head(h') stays at the BASE
value. IIA = fraction of held-out interchange pairs where both hold. The minimal k with high
scanner-IIA at preserved disease = the minimal causal scanner dimension.

Controls: a random (untrained) orthogonal R should give chance scanner-IIA.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DASRotation(nn.Module):
    """Learned orthogonal change-of-basis R (d×d); the first ``k`` rotated dims are the
    hypothesized scanner subspace. R orthogonal => unrotate is R^T (rotate is R)."""

    def __init__(self, d: int, k: int, seed: int = 0):
        super().__init__()
        self.d, self.k = d, k
        g = torch.Generator().manual_seed(seed)
        lin = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.linalg.qr(torch.randn(d, d, generator=g))[0])
        self.R = nn.utils.parametrizations.orthogonal(lin)   # .weight stays orthogonal

    def _W(self):
        return self.R.weight                                  # (d,d), orthonormal rows

    def rotate(self, h):    # h (B,d) -> rotated coords
        return h @ self._W().T

    def unrotate(self, z):  # inverse (R orthogonal)
        return z @ self._W()

    def interchange(self, h_base, h_src):
        """Swap the first k rotated coords from source into base, then rotate back."""
        zb, zs = self.rotate(h_base), self.rotate(h_src)
        z = torch.cat([zs[:, :self.k], zb[:, self.k:]], dim=1)
        return self.unrotate(z)


def train_das(h, scanner, head_W, head_b, scan_w, scan_b, base_idx, src_idx,
              k, steps=600, lr=5e-3, lam=1.0, seed=0, device="cpu"):
    """Fit a DAS rotation so interchanging the k-dim subspace transfers SCANNER (source) while
    preserving the model's DISEASE decision (base).

    h        : (N,d) fused reps (head input).            scanner: (N,) {0,1} scanner labels.
    head_W,head_b : model classifier head -> real disease logits head(h)=h@W.T+b.
    scan_w,scan_b : frozen linear scanner readout (e.g. trained LogisticRegression on train h).
    base_idx,src_idx : 1-D index tensors of paired subjects with DIFFERENT scanner (train pairs).
    Returns the trained DASRotation.
    """
    h = h.to(device)
    head_W, head_b = head_W.to(device), head_b.to(device)
    scan_w, scan_b = scan_w.to(device), scan_b.to(device)
    scanner = scanner.to(device)
    base_idx, src_idx = base_idx.to(device), src_idx.to(device)
    das = DASRotation(h.shape[1], k, seed=seed).to(device)
    opt = torch.optim.Adam(das.parameters(), lr=lr)
    base_dx = (h[base_idx] @ head_W.T + head_b).argmax(1)     # model's base disease decision
    s_src = scanner[src_idx].float().to(device)
    for _ in range(steps):
        opt.zero_grad()
        hp = das.interchange(h[base_idx], h[src_idx])
        scan_logit = hp @ scan_w + scan_b                      # (B,)
        dx_logit = hp @ head_W.T + head_b                      # (B,2)
        loss = F.binary_cross_entropy_with_logits(scan_logit, s_src) \
            + lam * F.cross_entropy(dx_logit, base_dx)
        loss.backward()
        opt.step()
    return das


@torch.no_grad()
def eval_iia(das, h, scanner, head_W, head_b, scan_w, scan_b, base_idx, src_idx, device="cpu"):
    """Held-out IIA: scanner flips to source, model disease decision preserved at base."""
    h = h.to(device)
    hp = das.interchange(h[base_idx], h[src_idx])
    scan_pred = (hp @ scan_w + scan_b) > 0
    dx_pred = (hp @ head_W.T + head_b).argmax(1)
    base_dx = (h[base_idx] @ head_W.T + head_b).argmax(1)
    scanner_iia = (scan_pred.long() == scanner[src_idx].to(device)).float().mean().item()
    disease_preserved = (dx_pred == base_dx).float().mean().item()
    return {"scanner_iia": scanner_iia, "disease_preserved": disease_preserved}
