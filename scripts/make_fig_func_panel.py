#!/usr/bin/env python
"""Functional connectome figure for the TMI paper -- deliberately a DIFFERENT visual idiom from the
sibling MultiViT2 atlas (which uses glass-brain projections + 53x53 matrices). Here:
  (a) circular CONNECTOGRAM (radial 53-ICN layout, domain arcs) with scanner (red) vs disease (blue) chords
  (b) coarse 7x7 DOMAIN-BLOCK scanner-importance matrix (network-pair concentration)
  (c) per-domain scanner-vs-disease recruitment DUMBBELL
Reads outputs/fnc_importance/ (this paper's US-vs-China scanner / SZ-vs-HC disease importance).
Writes manuscript/figures/fig_func_panel.pdf.
"""
from pathlib import Path as FPath
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Wedge

_HEROS = FPath("/home/users/ybi3/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre")
for _f in ("texgyreheros-regular.otf", "texgyreheros-bold.otf"):
    try: fm.fontManager.addfont(str(_HEROS / _f))
    except Exception: pass
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "mathtext.fontset": "stixsans",
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.6,
})
MM = 1 / 25.4
DAT = FPath("/home/users/ybi3/mechinterp_brain/outputs/fnc_importance")
OUT = FPath("/home/users/ybi3/mechinterp_brain/manuscript/figures"); OUT.mkdir(parents=True, exist_ok=True)
N = 53
DOM_ORDER = ["SC", "AU", "SM", "VI", "CC", "DM", "CB"]
DOM_COL = {"SC": "#7E4FA0", "AU": "#E0A526", "SM": "#3FA0C4", "VI": "#4DAF63",
           "CC": "#D8612C", "DM": "#C0476B", "CB": "#7A7D82"}
DOM_RANGE = {"SC": (0, 5), "AU": (5, 7), "SM": (7, 16), "VI": (16, 25), "CC": (25, 42),
             "DM": (42, 49), "CB": (49, 53)}
C_SCAN, C_DIS, INK = "#B64342", "#1F5FA6", "#222428"


def main():
    scanner = np.load(DAT / "scanner_coef_abs.npy").astype(float)
    disease = np.load(DAT / "disease_coef_abs.npy").astype(float)
    s_top = np.load(DAT / "scanner_top100.npy"); d_top = np.load(DAT / "disease_top100.npy")
    meta = json.loads((DAT / "meta.json").read_text())
    iu = np.triu_indices(N, 1)
    dom_of = np.empty(N, object)
    for d, (lo, hi) in DOM_RANGE.items():
        dom_of[lo:hi] = d

    # node angles (domain-ordered with inter-domain gaps)
    gap = 0.6
    total = N + gap * len(DOM_ORDER)
    ang = np.zeros(N); pos = gap / 2; rim = []
    for d in DOM_ORDER:
        lo, hi = DOM_RANGE[d]; a0 = pos
        for c in range(lo, hi):
            ang[c] = 2 * np.pi * (pos + 0.5) / total; pos += 1
        rim.append((2 * np.pi * a0 / total, 2 * np.pi * pos / total, d)); pos += gap
    P = np.c_[np.cos(ang), np.sin(ang)]

    fig = plt.figure(figsize=(183 * MM, 96 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 0.85],
                          left=0.02, right=0.97, top=0.9, bottom=0.08, wspace=0.32, hspace=0.45)
    fig.text(0.02, 0.955, "Functional connectome (53 ICNs): scanner vs. disease",
             fontsize=10.5, fontweight="bold")

    # ---- (a) circular connectogram ----
    ax = fig.add_subplot(gs[:, 0]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.42, 1.42); ax.set_ylim(-1.45, 1.42)
    for a0, a1, d in rim:
        ax.add_patch(Wedge((0, 0), 1.18, np.degrees(a0), np.degrees(a1), width=0.10,
                           facecolor=DOM_COL[d], edgecolor="white", lw=0.6))
        am = (a0 + a1) / 2
        ax.text(1.30 * np.cos(am), 1.30 * np.sin(am), d, ha="center", va="center",
                fontsize=6.2, fontweight="bold", color=DOM_COL[d])

    def draw_chords(top, coef, color, z):
        cs = coef[top]; w = (cs - cs.min()) / (np.ptp(cs) + 1e-9)
        for e, ww in zip(top, w):
            i, j = iu[0][e], iu[1][e]
            ctrl = 0.18 * (P[i] + P[j])
            ax.add_patch(PathPatch(Path([P[i], ctrl, P[j]], [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                                   fill=False, edgecolor=color, lw=0.4 + 1.4 * ww,
                                   alpha=0.20 + 0.55 * ww, zorder=z, capstyle="round"))
    draw_chords(d_top, disease, C_DIS, 2)
    draw_chords(s_top, scanner, C_SCAN, 3)
    hub = np.zeros(N)
    for e in np.concatenate([s_top, d_top]):
        hub[iu[0][e]] += scanner[e] + disease[e]; hub[iu[1][e]] += scanner[e] + disease[e]
    nd = 4 + 30 * (hub / (hub.max() + 1e-9))
    ax.scatter(P[:, 0], P[:, 1], s=nd, c=[DOM_COL[dom_of[c]] for c in range(N)],
               edgecolors="white", linewidths=0.3, zorder=5)
    ax.text(0, -1.42, "a   scanner (red) vs disease (blue) edges, by network",
            ha="center", fontsize=8, fontweight="bold")

    # ---- (b) 7x7 domain-block scanner matrix ----
    ax = fig.add_subplot(gs[0, 1])
    nd_ = len(DOM_ORDER); B = np.zeros((nd_, nd_))
    di = np.array([DOM_ORDER.index(dom_of[iu[0][e]]) for e in range(len(scanner))])
    dj = np.array([DOM_ORDER.index(dom_of[iu[1][e]]) for e in range(len(scanner))])
    for a in range(nd_):
        for b in range(nd_):
            m = ((di == a) & (dj == b)) | ((di == b) & (dj == a))
            B[a, b] = scanner[m].mean() if m.any() else 0
    im = ax.imshow(B, cmap="Reds", vmin=0)
    ax.set_xticks(range(nd_)); ax.set_yticks(range(nd_))
    ax.set_xticklabels(DOM_ORDER, fontsize=6); ax.set_yticklabels(DOM_ORDER, fontsize=6)
    ax.set_title("b   scanner by network pair", loc="left", fontweight="bold", fontsize=8.5, pad=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=5.5)
    cb.set_label("mean edge $|$coef$|$", fontsize=6)

    # ---- (c) per-domain recruitment dumbbell ----
    ax = fig.add_subplot(gs[1, 1])
    sc_by = np.array([scanner[(di == k) | (dj == k)].mean() for k in range(nd_)])
    di_by = np.array([disease[(di == k) | (dj == k)].mean() for k in range(nd_)])
    y = np.arange(nd_)
    for k in range(nd_):
        ax.plot([di_by[k], sc_by[k]], [y[k], y[k]], color="#bbb", lw=1.2, zorder=1)
    ax.scatter(di_by, y, s=26, color=C_DIS, zorder=2, label="disease")
    ax.scatter(sc_by, y, s=26, color=C_SCAN, zorder=2, label="scanner")
    ax.set_yticks(y); ax.set_yticklabels(DOM_ORDER, fontsize=6); ax.invert_yaxis()
    ax.set_xlabel("mean edge importance", fontsize=7)
    ax.legend(loc="lower right", fontsize=6, handletextpad=0.2, frameon=False)
    ax.set_title("c   network recruitment", loc="left", fontweight="bold", fontsize=8.5, pad=3)

    n_shared = len(set(s_top.tolist()) & set(d_top.tolist()))
    fig.text(0.02, 0.015, f"Scanner/disease top-100 edges near-disjoint "
             f"(Jaccard {meta.get('jaccard_top100_l2', 0):.2f}, {n_shared} shared).",
             fontsize=6, color="#5A5E63")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_func_panel.{ext}", dpi=380 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"connectogram saved; shared={n_shared}, jaccard={meta.get('jaccard_top100_l2',0):.3f}")
    print(f"wrote {OUT}/fig_func_panel.pdf")


if __name__ == "__main__":
    main()
