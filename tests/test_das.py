"""CPU smoke test for DAS/IIA on synthetic data with a KNOWN scanner subspace.

Construct h whose first 2 dims encode scanner and remaining dims encode disease (read by a fixed
head). A trained DAS rotation should recover high scanner-IIA while preserving the disease
decision; an untrained random rotation should not. Run:
  cd mechinterp_brain && PYTHONPATH=src python -m pytest tests/test_das.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from mib import das as D


def _synth(n=800, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    scanner = (torch.rand(n, generator=g) < 0.5).long()
    disease = (torch.rand(n, generator=g) < 0.5).long()
    h = 0.1 * torch.randn(n, d, generator=g)
    h[:, 0] += torch.where(scanner == 1, 2.0, -2.0)      # scanner lives in dim 0..1
    h[:, 1] += torch.where(scanner == 1, 1.5, -1.5)
    h[:, 5] += torch.where(disease == 1, 2.0, -2.0)      # disease lives in dim 5..6
    h[:, 6] += torch.where(disease == 1, 1.5, -1.5)
    head_W = torch.zeros(2, d); head_W[1, 5] = 2.0; head_W[1, 6] = 1.5    # head reads disease dims
    head_b = torch.zeros(2)
    scan_w = torch.zeros(d); scan_w[0] = 2.0; scan_w[1] = 1.5             # frozen scanner readout
    scan_b = torch.zeros(())
    return h, scanner, disease, head_W, head_b, scan_w, scan_b


def _pairs(scanner, idx, n_pairs=2000, seed=1):
    g = torch.Generator().manual_seed(seed)
    us = idx[scanner[idx] == 0]; ch = idx[scanner[idx] == 1]
    b = torch.cat([us[torch.randint(len(us), (n_pairs,), generator=g)],
                   ch[torch.randint(len(ch), (n_pairs,), generator=g)]])
    s = torch.cat([ch[torch.randint(len(ch), (n_pairs,), generator=g)],
                   us[torch.randint(len(us), (n_pairs,), generator=g)]])
    return b, s


def test_das_recovers_scanner_subspace_and_preserves_disease():
    h, scanner, disease, hW, hb, sw, sb = _synth()
    idx = torch.arange(len(h))
    tr, te = idx[:600], idx[600:]
    bt, st = _pairs(scanner, tr, seed=1)
    bv, sv = _pairs(scanner, te, seed=2)

    das = D.train_das(h, scanner, hW, hb, sw, sb, bt, st, k=2, steps=400, lr=5e-3)
    trained = D.eval_iia(das, h, scanner, hW, hb, sw, sb, bv, sv)

    rand = D.DASRotation(h.shape[1], k=2, seed=7)            # untrained control
    control = D.eval_iia(rand, h, scanner, hW, hb, sw, sb, bv, sv)

    assert trained["scanner_iia"] > 0.85, trained
    assert trained["disease_preserved"] > 0.9, trained
    assert trained["scanner_iia"] > control["scanner_iia"] + 0.15, (trained, control)
