"""Figure 4 — Functional connectome: the GROUP-MEAN biology of disease and scanner, and why the
classifier handle is a different object (Vince viz round).

Redesigned per reviewer request to lead with the interpretable biology:
  a  disease chord connectogram      signed mean(SZ)-mean(HC) edges (red hyper / blue hypo), domain-grouped
  b  scanner chord connectogram      signed mean(China)-mean(US) edges
  c  disease 53x53 hot-cold matrix   canonical mean(SZ)-mean(HC) (COBRE+FBIRN raw NeuroMark sFNC)
  d  scanner 53x53 hot-cold matrix   mean(China)-mean(US) on the model's own 4-cohort FNC
  e  reconciliation scatter          |L2 classifier coef| vs |group mean diff|: the handle != the biology (r~0.06)
  f  NeuroMark 53-ICN atlas          the parcellation that every edge comes from (data provenance)
  g  disease 7x7 domain blocks       signed mean(SZ)-mean(HC) network-block means
  h  scanner 7x7 domain blocks       signed mean(China)-mean(US) network-block means

Data: outputs/fnc_groupdiff/ (signed group-mean diffs, from scripts/fnc_group_diff.py) and
outputs/fnc_importance/ (|L2 coef| classifier maps, for the reconciliation). 53-ICN NeuroMark atlas.
Run: python scripts/make_brain_fig_A.py
"""
from __future__ import annotations
import os, json, tempfile
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, ListedColormap
from matplotlib.path import Path as MPath
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

_HEROS = Path("/home/users/ybi3/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre")
for _f in ("texgyreheros-regular.otf", "texgyreheros-bold.otf", "texgyreheros-italic.otf"):
    try:
        fm.fontManager.addfont(str(_HEROS / _f))
    except Exception:
        pass
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "mathtext.fontset": "stixsans",
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.6,
})
MM = 1 / 25.4
REPO = Path(__file__).resolve().parent.parent
GD = REPO / "outputs" / "fnc_groupdiff"
IMP = REPO / "outputs" / "fnc_importance"
OUT = REPO / "manuscript" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
SRC = REPO / "manuscript" / "source_data"; SRC.mkdir(parents=True, exist_ok=True)

C_INK = "#222428"
C_DIS, C_SCAN, C_BOTH = "#1F5FA6", "#B64342", "#7B4EA3"
N_ICN = 53
DOM_ORDER = ["SC", "AU", "SM", "VI", "CC", "DM", "CB"]
DOM_NAME = {"SC": "Subcortical", "AU": "Auditory", "SM": "Sensorimotor", "VI": "Visual",
            "CC": "Cognitive control", "DM": "Default mode", "CB": "Cerebellar"}
DOM_COL = {"SC": "#7E4FA0", "AU": "#E0A526", "SM": "#3FA0C4", "VI": "#4DAF63",
           "CC": "#D8612C", "DM": "#C0476B", "CB": "#7A7D82"}
_DFILE = Path("/data/qneuromark/Network_templates/NeuroMark1/Neuromark_fMRI_1.0.txt")
_ATLAS = "/data/qneuromark/Network_templates/NeuroMark1/Neuromark_fMRI_1.0.nii"
IU = np.triu_indices(N_ICN, 1)


def domains():
    c2d = np.empty(N_ICN, dtype=object)
    if _DFILE.exists():
        for line in _DFILE.read_text().splitlines():
            p = [q.strip() for q in line.split(",") if q.strip()]
            if len(p) >= 2 and p[0] in DOM_ORDER:
                for x in p[1:]:
                    if x.lstrip("-").isdigit():
                        c2d[int(x) - 1] = p[0]
    if any(v is None for v in c2d):
        rng = {"SC": range(0, 5), "AU": range(5, 7), "SM": range(7, 16), "VI": range(16, 25),
               "CC": range(25, 42), "DM": range(42, 49), "CB": range(49, 53)}
        for d, r in rng.items():
            for cc in r:
                c2d[cc] = d
    return c2d


def order_and_bounds(c2d):
    order = []
    for dom in DOM_ORDER:
        order += [c for c in range(N_ICN) if c2d[c] == dom]
    bnds, s = [], 0
    for dom in DOM_ORDER:
        n = int((c2d == dom).sum()); bnds.append((s, s + n, dom)); s += n
    return np.array(order), bnds


def ring_positions(c2d, gap=0.06):
    order = [i for d in DOM_ORDER for i in range(N_ICN) if c2d[i] == d]
    span = (2 * np.pi - gap * len(DOM_ORDER))
    ang = np.zeros(N_ICN); sectors = {}; a = np.pi / 2
    for d in DOM_ORDER:
        idx = [i for i in order if c2d[i] == d]; a0 = a
        for i in idx:
            ang[i] = a - (span / N_ICN) / 2; a -= span / N_ICN
        sectors[d] = (a0, a); a -= gap
    return np.c_[np.cos(ang), np.sin(ang)], ang, sectors


# ---- signed diverging colormap (hot = positive/hyper, cold = negative/hypo) ----
HOTCOLD = "RdBu_r"


def chord_signed(ax, M, c2d, pos, sectors, top=80, title="", letter="", vlim=None, pos_lbl="", neg_lbl=""):
    """Chord connectogram of a SIGNED 53x53 matrix: top-|edge| arcs coloured red (positive) / blue
    (negative) by a diverging map; node dots coloured by domain, sized by incident |value|."""
    vals = M[IU]
    vlim = vlim or float(np.percentile(np.abs(vals), 99.5))
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    cmap = plt.get_cmap(HOTCOLD)
    order = np.argsort(np.abs(vals))[::-1][:top]
    inc = np.zeros(N_ICN)
    for e in range(len(vals)):
        inc[IU[0][e]] += abs(vals[e]); inc[IU[1][e]] += abs(vals[e])
    inc /= inc.max() + 1e-12
    R = 1.13
    for d, (a0, a1) in sectors.items():
        th = np.linspace(a0, a1, 40)
        ax.plot(R * np.cos(th), R * np.sin(th), color=DOM_COL[d], lw=3.2, solid_capstyle="butt")
        am = (a0 + a1) / 2
        ax.text(1.32 * np.cos(am), 1.32 * np.sin(am), DOM_NAME[d].replace(" ", "\n"),
                ha="center", va="center", fontsize=4.9, color=DOM_COL[d], fontweight="bold")
    for e in order:
        i, j = IU[0][e], IU[1][e]; v = vals[e]; w = abs(v) / vlim
        p0, p1 = pos[i], pos[j]; ctrl = (p0 + p1) / 2 * (0.18 + 0.32 * (1 - min(w, 1)))
        path = MPath([tuple(p0), tuple(ctrl), tuple(p1)], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])
        ax.add_patch(mpatches.PathPatch(path, fc="none", ec=cmap(norm(v)),
                                        lw=0.4 + 1.9 * min(w, 1), alpha=0.45 + 0.5 * min(w, 1),
                                        capstyle="round"))
    for i in range(N_ICN):
        ax.scatter(*pos[i], s=6 + 90 * inc[i] ** 1.4, c=DOM_COL[c2d[i]],
                   edgecolors="white", linewidths=0.3, zorder=5)
    ax.set_xlim(-1.42, 1.42); ax.set_ylim(-1.46, 1.42); ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8.2, fontweight="bold", pad=1)
    if letter:
        ax.text(-1.40, 1.40, letter, fontsize=11, fontweight="bold", va="top")
    if pos_lbl:
        ax.text(1.40, -1.30, pos_lbl, fontsize=5.2, color="#B2182B", ha="right", fontweight="bold")
        ax.text(1.40, -1.44, neg_lbl, fontsize=5.2, color="#2166AC", ha="right", fontweight="bold")
    return vlim


def matrix_signed(fig, ax, M, order, bnds, title, lab, vlim=None, unit=r"$\Delta$ r"):
    Mo = M[np.ix_(order, order)]
    vlim = vlim or float(np.percentile(np.abs(M[IU]), 99))
    im = ax.imshow(Mo, cmap=HOTCOLD, vmin=-vlim, vmax=vlim, origin="upper", interpolation="nearest")
    for (s, e, dom) in bnds:
        ax.add_patch(Rectangle((s - .5, s - .5), e - s, e - s, fill=False, ec=C_INK, lw=0.55))
        ax.text(-1.6, (s + e) / 2 - .5, dom, ha="right", va="center", fontsize=5.0, color=DOM_COL[dom])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-6, N_ICN - 0.5)
    ax.set_title(f"{lab}   {title}", loc="left", fontweight="bold", fontsize=8.2, pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=5.0, length=2); cb.set_label(unit, fontsize=5.6)
    cb.outline.set_linewidth(0.4)
    return vlim


def block_signed(fig, ax, M, c2d, title, lab):
    idx = {d: [k for k in range(N_ICN) if c2d[k] == d] for d in DOM_ORDER}
    B = np.zeros((7, 7))
    for i, di in enumerate(DOM_ORDER):
        for j, dj in enumerate(DOM_ORDER):
            B[i, j] = M[np.ix_(idx[di], idx[dj])].mean()
    vlim = float(np.abs(B).max()) + 1e-9
    im = ax.imshow(B, cmap=HOTCOLD, vmin=-vlim, vmax=vlim, origin="upper", interpolation="nearest")
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(DOM_ORDER, fontsize=5.6); ax.set_yticklabels(DOM_ORDER, fontsize=5.6)
    ax.tick_params(length=0)
    for i in range(7):
        for j in range(7):
            if abs(B[i, j]) >= 0.55 * vlim:
                ax.text(j, i, f"{B[i, j]:+.02f}", ha="center", va="center", fontsize=4.4,
                        color="white" if abs(B[i, j]) > 0.72 * vlim else C_INK, fontweight="bold")
    ax.set_title(f"{lab}   {title}", loc="left", fontweight="bold", fontsize=8.0, pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=4.6, length=2); cb.outline.set_linewidth(0.4)


def atlas_png():
    """Render the 53-ICN NeuroMark probabilistic atlas to a temp PNG (filled contours, glass brain),
    so we can imshow it into the gridspec without nilearn fighting the shared layout."""
    from nilearn import plotting, image
    fp = Path(tempfile.gettempdir()) / "_neuromark_atlas_fig4.png"
    try:
        disp = plotting.plot_prob_atlas(_ATLAS, display_mode="z", cut_coords=[-12, 4, 18, 36],
                                        view_type="filled_contours", threshold=0.5,
                                        draw_cross=False, annotate=False, alpha=0.7)
        disp.savefig(str(fp), dpi=300); disp.close()
    except Exception as e:
        print(f"[atlas] prob_atlas failed ({repr(e)[:60]}); falling back to centroid markers")
        coords = np.load(IMP / "icn_mni_coords.npy"); c2d = domains()
        cols = [DOM_COL[c2d[i]] for i in range(N_ICN)]
        disp = plotting.plot_markers(np.ones(N_ICN), coords, node_size=22, node_cmap=None,
                                     display_mode="z", node_values=None) if False else None
        fig, ax = plt.subplots(figsize=(4, 1.2))
        plotting.plot_connectome(np.zeros((N_ICN, N_ICN)), coords, node_color=cols, node_size=14,
                                 display_mode="z", axes=ax, annotate=False, colorbar=False)
        fig.savefig(str(fp), dpi=300); plt.close(fig)
    return fp


def main():
    c2d = domains()
    order, bnds = order_and_bounds(c2d)
    pos, ang, sectors = ring_positions(c2d)

    dis = np.load(GD / "disease_meandiff.npy").astype(float)         # canonical SZ-HC (COBRE+FBIRN raw)
    dis_model = np.load(GD / "disease_meandiff_model.npy").astype(float)
    scan = np.load(GD / "scanner_meandiff_model.npy").astype(float)  # China-US (model 4-cohort)
    gj = json.loads((GD / "group_diff.json").read_text())
    r_dis = gj["corr_absMeanDiff_vs_absL2coef_disease"]
    r_scan = gj["corr_absMeanDiff_vs_absL2coef_scanner"]
    r_canon = gj["corr_canonical_vs_modelMeanDiff_disease"]
    dcoef = np.load(IMP / "disease_coef_abs.npy").astype(float)      # |L2 coef| classifier handle

    fig = plt.figure(figsize=(183 * MM, 196 * MM))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.16, 0.84, 0.80],
                          hspace=0.20, wspace=0.66, left=0.055, right=0.975,
                          top=0.952, bottom=0.058)
    fig.text(0.055, 0.975, "Functional connectome: the group-mean biology of disease and scanner",
             fontsize=10.5, fontweight="bold")

    # a, b: chord connectograms of the SIGNED group-mean difference (hero) ------------------
    ax_a = fig.add_subplot(gs[0, 0:3])
    vlim_d = chord_signed(ax_a, dis, c2d, pos, sectors, top=80,
                          title="Disease   mean(SZ) − mean(HC)", letter="a",
                          pos_lbl="red: SZ > HC (hyper)", neg_lbl="blue: SZ < HC (hypo)")
    ax_b = fig.add_subplot(gs[0, 3:6])
    chord_signed(ax_b, scan, c2d, pos, sectors, top=80,
                 title="Scanner   mean(China) − mean(US)", letter="b",
                 pos_lbl="red: China > US", neg_lbl="blue: China < US")

    # c, d: signed hot-cold 53x53 matrices ------------------------------------------------
    matrix_signed(fig, fig.add_subplot(gs[1, 0:2]), dis, order, bnds,
                  "Disease FNC difference (SZ−HC)", "c")
    matrix_signed(fig, fig.add_subplot(gs[1, 2:4]), scan, order, bnds,
                  "Scanner FNC difference (China−US)", "d")

    # e: reconciliation -- the classifier handle is a different object from the biology -----
    ax = fig.add_subplot(gs[1, 4:6])
    x = dcoef; y = np.abs(dis_model[IU])
    ax.scatter(x, y, s=3.2, c="#6B7077", alpha=0.45, linewidths=0)
    ax.set_xlabel("classifier handle  |L2 coef|", fontsize=6.4)
    ax.set_ylabel("group biology  |SZ−HC mean diff|", fontsize=6.4)
    ax.set_title("e   Handle ≠ biology", loc="left", fontweight="bold", fontsize=8.2, pad=4)
    ax.text(0.96, 0.94, f"$r$ = {r_dis:.2f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, fontweight="bold", color="#B64342")
    ax.text(0.96, 0.80, f"scanner $r$ = {r_scan:.2f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=5.6, color="#6B7077")
    ax.tick_params(labelsize=5.4)

    # f: NeuroMark atlas provenance --------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0:2])
    try:
        img = plt.imread(str(atlas_png()))
        ax.imshow(img); ax.axis("off")
    except Exception as e:
        ax.axis("off"); print(f"[atlas] embed failed: {repr(e)[:60]}")
    ax.set_title("f   53-ICN NeuroMark atlas (your parcellation)", loc="left",
                 fontweight="bold", fontsize=8.0, pad=2)

    # g, h: signed 7x7 domain-block summaries ---------------------------------------------
    block_signed(fig, fig.add_subplot(gs[2, 2:4]), dis, c2d, "Disease blocks (SZ−HC)", "g")
    block_signed(fig, fig.add_subplot(gs[2, 4:6]), scan, c2d, "Scanner blocks (China−US)", "h")

    fig.text(0.055, 0.020,
             "Nodes = 53 NeuroMark ICNs grouped by functional network (a,b); arcs/cells red = positive, "
             "blue = negative group difference.  The model's own training FNC reproduces the independent "
             f"COBRE+FBIRN disease pattern at r = {r_canon:.2f}, yet the |L2 classifier coefficient| is "
             f"nearly orthogonal to it (e, r = {r_dis:.2f}): decodability ≠ the marginal biology.",
             fontsize=5.2, color="#5A5E63")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig_brainA.{ext}", dpi=380 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    import pandas as pd
    pd.DataFrame({"r_absMeanDiff_vs_L2coef_disease": [r_dis],
                  "r_absMeanDiff_vs_L2coef_scanner": [r_scan],
                  "r_canonical_vs_model_disease": [r_canon]}).to_csv(
        SRC / "fig4_reconciliation.csv", index=False)
    print(f"[fig4] saved fig_brainA  disease-handle r={r_dis:.3f}  scanner-handle r={r_scan:.3f}  "
          f"canon-model r={r_canon:.3f}  vlim_d={vlim_d:.3f}")


if __name__ == "__main__":
    main()
