"""CPU smoke tests for the raw-activation patching primitives (Stage 1 crime-scene engine).

Validates make_write_hook (constant overwrite of a module output, optionally a token slice),
capture_hook_tokens (per-subject (N,T,d) capture), and the extra_hooks channel of run_model --
all on a tiny synthetic two-input module so no GPU / MultiViT checkpoint is needed.

Run: cd mechinterp_brain && PYTHONPATH=src python -m pytest tests/test_patch.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from mib import patch


class _CA(nn.Module):
    """Stand-in for a CrossAttention submodule: returns a (B, T, d) contribution."""
    def __init__(self, T, d):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.T, self.d = T, d

    def forward(self, x):
        return self.lin(x)


class _Tiny(nn.Module):
    """Two-input model mirroring MultiViT's call signature model(sMRI, sFNC) -> logits,
    with a named submodule `ca_b` whose output we patch and a `head` we capture."""
    def __init__(self, T=5, d=8):
        super().__init__()
        self.embed = nn.Linear(4, d)
        self.ca_b = _CA(T, d)
        self.head = nn.Linear(d, 2)
        self.T, self.d = T, d

    def forward(self, sMRI, sFNC):
        b = self.embed(sFNC)            # (B, T, d)
        b = b + self.ca_b(b)            # residual add of the contribution (like b2 = b + ca_b(...))
        return self.head(b[:, 0])       # logits from the CLS token


def _loader(n=12, T=5):
    data = [{"sMRI": torch.randn(4), "sFNC": torch.randn(T, 4),
             "subject_id": f"S{i}", "label": torch.tensor(i % 2)} for i in range(n)]
    def collate(items):
        return {"sMRI": torch.stack([x["sMRI"] for x in items]),
                "sFNC": torch.stack([x["sFNC"] for x in items]),
                "subject_id": [x["subject_id"] for x in items],
                "label": torch.stack([x["label"] for x in items])}
    return torch.utils.data.DataLoader(data, batch_size=4, collate_fn=collate)


def test_capture_hook_tokens_shape_and_order():
    m = _Tiny().eval()
    acts, sids = patch.capture_hook_tokens(m, _loader(12), "cpu", "ca_b")
    assert acts.shape == (12, m.T, m.d)
    assert sids == [f"S{i}" for i in range(12)]


def test_write_hook_overwrites_full_sequence():
    m = _Tiny().eval()
    val = torch.full((m.T, m.d), 3.14)          # constant (T,d) injected for every subject
    hk = patch.make_write_hook(val, token_idx=None)
    acts, _ = patch.capture_hook_tokens(m, _loader(8), "cpu", "ca_b",)
    # with the write hook the captured ca_b output must equal the constant everywhere
    h = patch._resolve(m, "ca_b").register_forward_hook(hk)
    acts2, _ = patch.capture_hook_tokens(m, _loader(8), "cpu", "ca_b")
    h.remove()
    assert torch.allclose(acts2, val.expand_as(acts2)), "write hook did not overwrite output"
    assert not torch.allclose(acts, acts2), "baseline and patched should differ"


def test_write_hook_token_slice_only():
    m = _Tiny().eval()
    val = torch.zeros(m.d)                        # zero only the CLS row (token 0)
    hk = patch.make_write_hook(val, token_idx=0)
    h = patch._resolve(m, "ca_b").register_forward_hook(hk)
    acts, _ = patch.capture_hook_tokens(m, _loader(8), "cpu", "ca_b")
    h.remove()
    assert torch.allclose(acts[:, 0], torch.zeros(8, m.d)), "CLS row not zeroed"
    assert not torch.allclose(acts[:, 1:], torch.zeros(8, m.T - 1, m.d)), "non-CLS wrongly touched"


def test_run_model_extra_hooks_changes_logits():
    m = _Tiny().eval()
    base = patch.run_model(m, _loader(8), "cpu", capture="head")
    grand = patch.capture_hook_tokens(m, _loader(8), "cpu", "ca_b")[0].mean(0)   # (T,d) mean
    patched = patch.run_model(m, _loader(8), "cpu", capture="head",
                              extra_hooks=[("ca_b", patch.make_write_hook(grand))])
    assert base["logits"].shape == (8, 2) and patched["logits"].shape == (8, 2)
    assert base["fused"].shape == (8, m.d)        # head input captured
    # mean-patching the ca_b contribution must move the decision representation
    assert not torch.allclose(base["fused"], patched["fused"]), "extra_hooks had no effect"
