"""SAE quality + stability metrics.

Reporting metrics (L0, explained variance, dead-feature %) are implemented here and
are what we log every epoch. The *feature-stability* metric is the Phase-1 verification
gate -- it decides which discovered features we trust enough to carry into the Phase-2
scanner-vs-disease probing -- and is left for you to implement (see TODO below).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def l0(features: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean number of active (non-zero) features per sample."""
    return (features.abs() > eps).float().sum(-1).mean().item()


@torch.no_grad()
def explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """1 - FVU. >0.9 is a healthy reconstruction; report alongside L0."""
    resid_var = (x - x_hat).var(dim=0, unbiased=False).sum()
    total_var = x.var(dim=0, unbiased=False).sum().clamp_min(1e-12)
    return (1.0 - resid_var / total_var).item()


@torch.no_grad()
def dead_feature_fraction(features: torch.Tensor, eps: float = 1e-8) -> float:
    """Fraction of dictionary features that never fire across the whole batch.
    >5% dead is a warning sign on small medical-N (consider resampling / lower L1)."""
    ever_active = (features.abs() > eps).any(dim=0)
    return (1.0 - ever_active.float().mean()).item()


@torch.no_grad()
def feature_stability(
    W_dec_a: torch.Tensor,
    W_dec_b: torch.Tensor,
    threshold: float = 0.7,
) -> dict:
    """Match dictionary features across two independently-trained SAEs (different seeds)
    and report how many replicate -- our definition of a "real", trustworthy feature.

    Method: optimal 1-to-1 assignment (Hungarian / ``linear_sum_assignment``) on the
    sign-sensitive cosine-similarity matrix of L2-normalised decoder rows, then keep a
    pair only if it is ALSO a mutual nearest neighbour and its cosine >= ``threshold``.

    Rationale for this (stricter) gate: optimal assignment forbids two A-features
    claiming the same B-feature (greedy does not); the mutual-NN filter on top removes
    pairs that the global assignment matched only because something better was already
    taken. What survives both is a feature each SAE independently put in (nearly) the
    same direction -- exactly what we want to risk Phase-3 activation-patching GPU on.
    Cosine is NOT abs()'d: SAE features are not sign-symmetric (decoder direction is the
    feature's meaning), so an anti-aligned pair is a different feature, not a match.

    Args:
        W_dec_a, W_dec_b: decoder weights (d_sae, d_in); each ROW is one feature's
            direction. Need not share d_sae (rectangular assignment is handled).
        threshold: cosine above which a mutually-matched pair counts as stable.

    Returns:
        n_stable      : int, number of stable feature pairs.
        frac_stable   : float, n_stable / number of assigned pairs (= min(d_sae_a, d_sae_b)).
        matched_cosine: (P,) cosine of every assigned pair (P = min d_sae).
        is_stable     : (P,) bool mask (mutual-NN AND cosine>=threshold) over assigned pairs.
        row, col      : (P,) long indices of the assigned A-feature / B-feature per pair.
    """
    from scipy.optimize import linear_sum_assignment

    A = F.normalize(W_dec_a.float().cpu(), dim=1)
    B = F.normalize(W_dec_b.float().cpu(), dim=1)
    C = A @ B.t()                                   # (na, nb) sign-sensitive cosine

    row_np, col_np = linear_sum_assignment(-C.numpy())   # maximise total cosine
    row = torch.as_tensor(row_np, dtype=torch.long)
    col = torch.as_tensor(col_np, dtype=torch.long)
    matched_cos = C[row, col]

    a_best = C.argmax(dim=1)                         # each A-feature's best B-feature
    b_best = C.argmax(dim=0)                         # each B-feature's best A-feature
    mutual = (a_best[row] == col) & (b_best[col] == row)
    is_stable = mutual & (matched_cos >= threshold)

    n_stable = int(is_stable.sum())
    return {
        "n_stable": n_stable,
        "frac_stable": n_stable / len(row),
        "matched_cosine": matched_cos,
        "is_stable": is_stable,
        "row": row,
        "col": col,
    }
