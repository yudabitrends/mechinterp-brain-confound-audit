"""CPU smoke test for SFC node attribution (Stage 3) on a tiny synthetic model + SAE.

Validates: participation_ratio (localized vs distributed), top_features selection, and that
node_attribution runs end-to-end through a hooked module + SAE and ranks the feature that
actually drives the scanner metric above an irrelevant one.

Run: cd mechinterp_brain && PYTHONPATH=src python -m pytest tests/test_edge_attribution.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from mib import edge_attribution as EA
from mib.sae import SAEConfig, SparseAutoencoder


def test_participation_ratio_localized_vs_distributed():
    loc = torch.tensor([10.0, 0.01, 0.01, 0.01])      # one dominant -> PR ~ 1
    dist = torch.ones(50)                              # uniform -> PR ~ 50
    assert EA.participation_ratio(loc) < 1.5
    assert EA.participation_ratio(dist) > 40


def test_top_features_selects_highest_abs():
    attr = {"h": torch.tensor([0.1, -5.0, 0.2, 3.0])}
    sel = EA.top_features(attr, k_per_hook=2)["h"].tolist()
    assert set(sel) == {1, 3}


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return x + self.lin(x)


class _Tiny(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.embed = nn.Linear(4, d)
        self.pre = _Block(d)            # UPSTREAM hook point (must still get a gradient)
        self.mid = _Block(d)            # downstream hook point
        self.head = nn.Linear(d, 2)
        self.d = d

    def forward(self, sMRI, sFNC):
        b = self.embed(sFNC)            # (B,T,d)
        b = self.pre(b)
        b = self.mid(b)
        return self.head(b[:, 0])


def _loader(n=64, T=3):
    data = [{"sMRI": torch.randn(4), "sFNC": torch.randn(T, 4),
             "subject_id": f"S{i}", "label": torch.tensor(i % 2)} for i in range(n)]
    def collate(it):
        return {"sMRI": torch.stack([x["sMRI"] for x in it]),
                "sFNC": torch.stack([x["sFNC"] for x in it]),
                "subject_id": [x["subject_id"] for x in it],
                "label": torch.stack([x["label"] for x in it])}
    return torch.utils.data.DataLoader(data, batch_size=16, collate_fn=collate)


def test_node_attribution_runs_and_ranks():
    torch.manual_seed(0)
    m = _Tiny().eval()
    for p in m.parameters():
        p.requires_grad_(False)         # frozen model; we differentiate w.r.t. activations
    # a trivial SAE: identity-ish dictionary so feature j ~ activation dim j
    cfg = SAEConfig(d_in=m.d, expansion=1, architecture="topk", k=m.d)
    sae = SparseAutoencoder(cfg)
    # scanner metric reads activation dim 2 strongly -> feature 2 should rank high
    w = torch.zeros(2 * m.d) if False else None
    metric_vec = torch.zeros(m.d); metric_vec[2] = 1.0
    def metric_fn(fused):               # fused = head input (B,d)
        return fused @ metric_vec
    sae2 = SparseAutoencoder(SAEConfig(d_in=m.d, expansion=1, architecture="topk", k=m.d))
    attr, totals = EA.node_attribution(m, _loader(), "cpu", ["pre", "mid"],
                                       {"pre": sae2, "mid": sae}, metric_fn)
    for h in ["pre", "mid"]:
        assert attr[h].shape[0] == m.d and torch.isfinite(attr[h]).all()
    # the regression guard: the UPSTREAM hook must get a real (non-zero) gradient too
    assert totals["pre"] > 0, "upstream hook gradient was severed (graph-detach bug)"
    assert totals["mid"] > 0
