#!/usr/bin/env python
"""E5 (reviewer / Vince) — what does the scanner circuit look like on the connectome?

Two-panel publication figure on the 53-ICN Neuromark FNC:
  (A) domain-ordered 53x53 per-edge scanner-importance heatmap (|AUC-0.5| of each edge for the
      scanner axis) with functional-domain grid -> WHERE the model's nuisance signal lives.
  (B) per-domain mean edge importance, scanner vs disease -> which networks carry nuisance vs disease.
Per-edge importance via mib.probe.per_feature_auc on the raw 1378-edge FNC (this paper's data/labels,
not MultiViT2's precomputed coefs). Reuses cross_arch.build_fnc + TeX Gyre Heros style.
Run in the `project` conda env (load_geometric_cohort). Writes manuscript/figures/fig_circuit.pdf.
"""
import os, sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mib.probe import per_feature_auc
from cross_arch import build_fnc

_HEROS = Path("/home/users/ybi3/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre")
for _f in ("texgyreheros-regular.otf", "texgyreheros-bold.otf"):
    try: fm.fontManager.addfont(str(_HEROS / _f))
    except Exception: pass
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "mathtext.fontset": "stixsans",
    "axes.linewidth": 0.7, "legend.frameon": False,
})
MM = 1 / 25.4
OUT = Path("manuscript/figures"); OUT.mkdir(parents=True, exist_ok=True)
# Neuromark 1.0 domain partition (0-indexed IC ranges)
DOMAINS = [("SC", 0, 5), ("AUD", 5, 7), ("SM", 7, 16), ("VS", 16, 25),
           ("CC", 25, 42), ("DM", 42, 49), ("CB", 49, 53)]
C_SCAN, C_DIS = "#C0392B", "#1E8449"


def main():
    mats, X, ydx, split, pop, site = build_fnc()
    tr = split == "train"; keep = np.isin(pop, ["US", "China"])
    yscan = (pop == "China").astype(int)
    trk = tr & keep
    iu = np.triu_indices(53, 1)

    scan_auc = per_feature_auc(X[trk], yscan[trk])          # (1378,)
    dis_auc = per_feature_auc(X[trk], ydx[trk])
    scan_imp = np.abs(scan_auc - 0.5); dis_imp = np.abs(dis_auc - 0.5)

    def to_mat(vec):
        M = np.zeros((53, 53)); M[iu] = vec; M = M + M.T; return M
    Sm = to_mat(scan_imp)
    order = np.concatenate([np.arange(lo, hi) for _, lo, hi in DOMAINS])
    Sm_o = Sm[np.ix_(order, order)]

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 78 * MM),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- Panel A: domain-ordered scanner-importance heatmap ----
    ax = axes[0]
    im = ax.imshow(Sm_o, cmap="Reds", vmin=0, vmax=np.percentile(Sm_o[Sm_o > 0], 98))
    pos = 0
    for name, lo, hi in DOMAINS:
        w = hi - lo
        ax.add_patch(plt.Rectangle((pos - 0.5, pos - 0.5), w, w, fill=False,
                                    edgecolor="#222428", lw=0.9))
        ax.text(pos + w / 2 - 0.5, -1.6, name, ha="center", va="bottom", fontsize=6.2, color="#222428")
        ax.text(-1.6, pos + w / 2 - 0.5, name, ha="right", va="center", fontsize=6.2, color="#222428")
        pos += w
    ax.set(xticks=[], yticks=[], xlim=(-0.5, 52.5), ylim=(52.5, -0.5))
    ax.set_title("A  Scanner circuit on the FNC connectome", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("scanner edge importance |AUC$-$0.5|", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    # ---- Panel B: per-domain mean importance, scanner vs disease ----
    ax = axes[1]
    dom_of = np.zeros(53, int)
    for di, (_, lo, hi) in enumerate(DOMAINS):
        dom_of[lo:hi] = di
    # edge -> domain pair; assign each edge to the (min) domain of its two nodes for a per-domain mean
    e_dom = np.minimum(dom_of[iu[0]], dom_of[iu[1]])
    names = [n for n, _, _ in DOMAINS]
    sc_by = [scan_imp[e_dom == di].mean() for di in range(len(DOMAINS))]
    di_by = [dis_imp[e_dom == di].mean() for di in range(len(DOMAINS))]
    y = np.arange(len(names)); w = 0.38
    ax.barh(y - w/2, sc_by, w, color=C_SCAN, label="scanner")
    ax.barh(y + w/2, di_by, w, color=C_DIS, label="disease")
    ax.set(yticks=y, yticklabels=names, xlabel="mean edge importance |AUC$-$0.5|")
    ax.invert_yaxis(); ax.legend(loc="lower right")
    ax.set_title("B  Which networks carry nuisance vs disease", loc="left", fontweight="bold")

    fig.tight_layout(w_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_circuit.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    # report the top scanner domains for the text
    top = sorted(zip(names, sc_by), key=lambda t: -t[1])[:3]
    print("top scanner domains:", [(n, round(v, 3)) for n, v in top])
    print(f"wrote {OUT}/fig_circuit.pdf")


if __name__ == "__main__":
    main()
