"""CPU smoke tests for the SAE core: shapes, sparsity, and that it actually learns.

Run: cd mechinterp_brain && PYTHONPATH=src python -m pytest tests/test_sae.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from mib.sae import SAEConfig, SparseAutoencoder
from mib import metrics


def _synthetic(d_in=32, n=512, rank=8, seed=0):
    """Low-rank-ish data: a small dictionary of directions, sparse mixtures + noise.
    A good SAE should recover high explained variance with low L0."""
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(rank, d_in, generator=g)
    codes = (torch.rand(n, rank, generator=g) < 0.25).float() * torch.rand(n, rank, generator=g)
    return codes @ D + 0.01 * torch.randn(n, d_in, generator=g)


def test_shapes_and_topk_sparsity():
    cfg = SAEConfig(d_in=32, expansion=4, architecture="topk", k=5)
    sae = SparseAutoencoder(cfg)
    x = _synthetic()
    x_hat, f = sae(x)
    assert x_hat.shape == x.shape
    assert f.shape == (x.shape[0], cfg.d_sae)
    # topk: at most k active per row
    assert (f > 0).sum(-1).max().item() <= cfg.k


def test_standard_learns_and_is_sparse():
    cfg = SAEConfig(d_in=32, expansion=8, architecture="standard", l1_coef=1e-3, seed=1)
    sae = SparseAutoencoder(cfg)
    x = _synthetic(seed=1)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    ev0 = metrics.explained_variance(x, sae(x)[0])
    for _ in range(400):
        opt.zero_grad()
        total, _ = sae.loss(x)
        total.backward()
        opt.step()
        sae.normalize_decoder()
    x_hat, f = sae(x)
    ev1 = metrics.explained_variance(x, x_hat)
    assert ev1 > ev0 and ev1 > 0.8, f"explained var did not improve enough: {ev0:.3f}->{ev1:.3f}"
    assert metrics.l0(f) < cfg.d_sae * 0.5, "features not sparse"


def test_topk_learns():
    cfg = SAEConfig(d_in=32, expansion=8, architecture="topk", k=8, seed=2)
    sae = SparseAutoencoder(cfg)
    x = _synthetic(seed=2)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for _ in range(400):
        opt.zero_grad()
        total, _ = sae.loss(x)
        total.backward()
        opt.step()
    ev = metrics.explained_variance(x, sae(x)[0])
    assert ev > 0.8, f"topk SAE under-reconstructs: {ev:.3f}"
    assert metrics.l0(sae.encode(x)) <= cfg.k + 1e-6


def test_metrics_basic():
    f = torch.tensor([[1.0, 0, 0, 2.0], [0, 0, 0, 3.0]])
    assert abs(metrics.l0(f) - 1.5) < 1e-6
    # feature 1 and 2 never fire -> 50% dead
    assert abs(metrics.dead_feature_fraction(f) - 0.5) < 1e-6


def _rand_dec(d_sae=16, d_in=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(d_sae, d_in, generator=g)


def test_stability_identical_and_permuted():
    W = _rand_dec()
    # identical decoders -> every feature stable
    s = metrics.feature_stability(W, W.clone())
    assert s["n_stable"] == W.shape[0] and s["frac_stable"] == 1.0
    # row permutation -> assignment recovers it, still all stable
    perm = torch.randperm(W.shape[0], generator=torch.Generator().manual_seed(9))
    s2 = metrics.feature_stability(W, W[perm].clone())
    assert s2["n_stable"] == W.shape[0]


def test_stability_sign_flip_not_matched():
    # cosine is sign-sensitive: a directly anti-aligned feature must score -1 -> 0 stable
    A = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    B = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
    assert metrics.feature_stability(A, B)["n_stable"] == 0
    # flipping one row among otherwise-identical rows drops the stable count, and every
    # surviving stable pair is genuinely above threshold (no abs() leakage)
    W = _rand_dec(seed=3)
    Wb = W.clone()
    Wb[0] = -Wb[0]
    s = metrics.feature_stability(W, Wb)
    assert s["n_stable"] < W.shape[0]
    assert bool((s["matched_cosine"][s["is_stable"]] >= 0.7).all())


def test_stability_random_pairs_mostly_unstable():
    # two independent high-dim dictionaries share almost no directions
    s = metrics.feature_stability(_rand_dec(seed=1), _rand_dec(seed=2), threshold=0.7)
    assert s["frac_stable"] < 0.2
