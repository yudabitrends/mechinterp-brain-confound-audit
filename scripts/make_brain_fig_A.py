"""Standalone brain/FNC atlas figure (8 panels) — concentrates every brain-space
view of the scanner-vs-disease confound geometry into one display.

  a  disease-predictive connectome (glass brain, 4 views)
  b  scanner-predictive connectome (glass brain, 4 views)
  c  disease-predictive FNC importance matrix (53x53, 7 domains)
  d  scanner-predictive FNC importance matrix (53x53)
  e  top-100 edge overlay (disease / scanner / shared)
  f  disease hub regions (glass-brain markers, sized by incident importance)
  g  scanner hub regions (glass-brain markers)
  h  network-domain incidence of each signal's top-100 edges

Data: outputs/p_fnc_edge_importance/ (+ NeuroMark1 domain template). 53-ICN atlas,
N=1,246 SZ subjects / 16 sites / 1,378 FNC edges. Run: python scripts/make_fig_brain_atlas.py
"""
from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
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
DAT = REPO / "outputs" / "fnc_importance"
OUT = REPO / "manuscript" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
SRC = REPO / "manuscript" / "source_data"; SRC.mkdir(parents=True, exist_ok=True)

C_DIS, C_SCAN, C_BOTH, C_INK = "#1F5FA6", "#B64342", "#7B4EA3", "#222428"
N_ICN = 53
DOM_ORDER = ["SC", "AU", "SM", "VI", "CC", "DM", "CB"]
DOM_NAME = {"SC": "Subcortical", "AU": "Auditory", "SM": "Sensorimotor", "VI": "Visual",
            "CC": "Cognitive control", "DM": "Default mode", "CB": "Cerebellar"}
DOM_COL = {"SC": "#7E4FA0", "AU": "#E0A526", "SM": "#3FA0C4", "VI": "#4DAF63",
           "CC": "#D8612C", "DM": "#C0476B", "CB": "#7A7D82"}
_DFILE = Path("/data/qneuromark/Network_templates/NeuroMark1/Neuromark_fMRI_1.0.txt")


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


def trunc(base, lo=0.32, hi=1.0):
    c = plt.get_cmap(base)
    return LinearSegmentedColormap.from_list(f"{base}_t", c(np.linspace(lo, hi, 256)))


def vec_to_mat(vec):
    M = np.zeros((N_ICN, N_ICN)); iu = np.triu_indices(N_ICN, k=1)
    M[iu] = vec; M[(iu[1], iu[0])] = vec
    return M


def rankn(v):
    return np.argsort(np.argsort(v)) / (len(v) - 1)


def block_matrix(vec, c2d):
    """7x7 network-block mean importance (the level at which structure is real;
    edge-level 53x53 is too sparse/diffuse to read)."""
    M = vec_to_mat(vec)
    idx = {d: [k for k in range(N_ICN) if c2d[k] == d] for d in DOM_ORDER}
    B = np.zeros((7, 7))
    for i, di in enumerate(DOM_ORDER):
        for j, dj in enumerate(DOM_ORDER):
            B[i, j] = M[np.ix_(idx[di], idx[dj])].mean()
    return B


def block_panel(ax, B, cmap, title, lab):
    Bn = (B - B.min()) / (np.ptp(B) + 1e-9)
    ax.imshow(Bn, cmap=cmap, vmin=0, vmax=1, origin="upper", interpolation="nearest")
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(DOM_ORDER, fontsize=5.8); ax.set_yticklabels(DOM_ORDER, fontsize=5.8)
    ax.tick_params(length=0)
    for i in range(7):
        for j in range(7):
            if Bn[i, j] >= 0.72:
                ax.text(j, i, f"{B[i, j]:.2f}", ha="center", va="center",
                        fontsize=4.6, color="white", fontweight="bold")
    ax.set_title(f"{lab}   {title}", loc="left", fontweight="bold", fontsize=8.0, pad=4)


def adjacency(coef, top, n):
    iu = np.triu_indices(N_ICN, k=1); sel = top[:n]; cs = coef[sel]
    w = np.zeros(len(coef)); w[sel] = 0.45 + 0.55 * (cs - cs.min()) / (np.ptp(cs) + 1e-9)
    A = np.zeros((N_ICN, N_ICN)); A[iu] = w; A[(iu[1], iu[0])] = w
    return A


def order_and_bounds(c2d):
    order = []
    for dom in DOM_ORDER:
        order += [c for c in range(N_ICN) if c2d[c] == dom]
    bnds, s = [], 0
    for dom in DOM_ORDER:
        n = int((c2d == dom).sum()); bnds.append((s, s + n, dom)); s += n
    return np.array(order), bnds


def matrix_panel(ax, M, order, bnds, cmap, title, lab):
    Mo = M[np.ix_(order, order)]
    ax.imshow(Mo, cmap=cmap, vmin=0.5, vmax=1.0, origin="upper", interpolation="nearest")
    for (s, e, dom) in bnds:
        ax.add_patch(Rectangle((s - .5, s - .5), e - s, e - s, fill=False, ec=C_INK, lw=0.6))
        ax.text(-1.4, (s + e) / 2 - .5, dom, ha="right", va="center", fontsize=5.0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-6, N_ICN - 0.5)
    ax.set_title(f"{lab}   {title}", loc="left", fontweight="bold", fontsize=8.5, pad=4)


def netplot_connectome(fig, spec, coef, top, coords, c2d, edge_color, n_top=70, views=("L", "S", "R")):
    """Embed a modern 3D node-edge connectome (netplotbrain, matplotlib Axes3D -> headless-safe, no VTK) into the
    gridspec region `spec`, split into len(views) sub-axes on a translucent MNI template. Nodes are coloured by
    functional domain and sized by incident importance. We neutralise netplotbrain's internal fig.tight_layout()
    so it cannot disturb the surrounding multi-panel gridspec."""
    os.environ.setdefault("TEMPLATEFLOW_HOME", "/data/users1/ybi/mechinterp_brain/templateflow")
    import netplotbrain, pandas as pd
    A = adjacency(coef, top, n_top)
    nodes = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2]})
    ncol = [DOM_COL[c2d[i]] for i in range(N_ICN)]
    nsz = 5 + 16 * (A.sum(0) / (A.sum(0).max() + 1e-9))
    sub = spec.subgridspec(1, len(views), wspace=0.0)
    axes = []
    _tl = fig.tight_layout; fig.tight_layout = lambda *a, **k: None     # guard the shared gridspec
    try:
        for k, v in enumerate(views):                                  # one netplotbrain call per view/axes
            ax = fig.add_subplot(sub[0, k], projection="3d")
            netplotbrain.plot(nodes=nodes, edges=A, fig=fig, ax=ax, view=v,
                              template="MNI152NLin2009cAsym", templatestyle="glass",
                              template_glass_maxalpha=0.03, template_glass_pointsize=1.6,  # visible brain shell
                              node_color=ncol, node_size=nsz, node_alpha=1.0,
                              edge_color=edge_color, edge_alpha=0.55, edge_widthscale=0.55,
                              arrowaxis=None, subtitles=None, title=None)
            axes.append(ax)
    finally:
        fig.tight_layout = _tl
    return axes


def main():
    from nilearn import plotting

    scanner = np.load(DAT / "scanner_coef_abs.npy").astype(float)
    disease = np.load(DAT / "disease_coef_abs.npy").astype(float)
    s_top = np.load(DAT / "scanner_top100.npy"); d_top = np.load(DAT / "disease_top100.npy")
    coords = np.load(DAT / "icn_mni_coords.npy")
    meta = json.loads((DAT / "meta.json").read_text()); jac = meta["jaccard_top100_l2"]
    c2d = domains(); node_colors = [DOM_COL[c2d[i]] for i in range(N_ICN)]
    order, bnds = order_and_bounds(c2d)

    fig = plt.figure(figsize=(183 * MM, 212 * MM))
    # The full-width connectome rows (0,1) are width-limited, so their generous height held whitespace; trim
    # total height to ease the page-fit of this dense plate without shrinking any brain render.
    gs = fig.add_gridspec(4, 6, height_ratios=[1.05, 1.05, 0.92, 0.92],
                          hspace=0.40, wspace=0.62, left=0.045, right=0.985,
                          top=0.955, bottom=0.045)
    fig.text(0.045, 0.975, "Functional connectome: where the scanner confound and disease live",
             fontsize=10.5, fontweight="bold")

    # a, b: modern 3D node-edge connectomes on a translucent MNI template (netplotbrain) -------------------
    for row, (coef, top, col, lab, name) in enumerate([
            (disease, d_top, C_DIS, "a", "Disease-predictive connectome"),
            (scanner, s_top, C_SCAN, "b", "Scanner-predictive connectome")]):
        netplot_connectome(fig, gs[row, :], coef, top, coords, c2d, edge_color=col)
        pos = gs[row, :].get_position(fig)
        fig.text(pos.x0 + 0.004, pos.y1 - 0.004, f"{lab}   {name}", fontsize=9.5, fontweight="bold",
                 color=col, ha="left", va="top")

    # c, d: network-block importance matrices (7x7; the level where structure
    #       is interpretable — edge-level 53x53 is too diffuse to read) -------
    block_panel(fig.add_subplot(gs[2, 0:2]), block_matrix(disease, c2d),
                "Blues", "Disease network-block importance", "c")
    block_panel(fig.add_subplot(gs[2, 2:4]), block_matrix(scanner, c2d),
                "Reds", "Scanner network-block importance", "d")

    # e: top-100 overlay matrix -------------------------------------------
    ax = fig.add_subplot(gs[2, 4:6])
    iu = np.triu_indices(N_ICN, k=1)
    cat = np.zeros(len(scanner), dtype=int); cat[d_top] = 1; cat[s_top] += 2
    Mc = np.zeros((N_ICN, N_ICN), int); Mc[iu] = cat; Mc[(iu[1], iu[0])] = cat
    Mc = Mc[np.ix_(order, order)]
    cmap = ListedColormap(["#F2F3F5", C_DIS, C_SCAN, C_BOTH])
    ax.imshow(Mc, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4),
              origin="upper", interpolation="nearest")
    for (s, e, dom) in bnds:
        ax.add_patch(Rectangle((s - .5, s - .5), e - s, e - s, fill=False, ec=C_INK, lw=0.6))
    ax.set_xticks([]); ax.set_yticks([])
    n_both = int((cat == 3).sum())
    ax.set_title(f"e   Top-100 overlap ({n_both} shared)", loc="left", fontweight="bold",
                 fontsize=8.5, pad=4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=C_DIS, label="disease"), Patch(fc=C_SCAN, label="scanner"),
                       Patch(fc=C_BOTH, label="both")], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=5.6, handlelength=1.0,
              columnspacing=0.8)

    # f, g: top hub regions as labelled ranked bars (readable: which ICN/network
    #       is a hub, vs unlabelled glass-brain markers) -----------------------
    def hub_strength(coef, top):
        A = adjacency(coef, top, 100); return A.sum(0)
    for col_i, (coef, top, color, lab, name) in enumerate([
            (disease, d_top, C_DIS, "f", "Disease hub regions"),
            (scanner, s_top, C_SCAN, "g", "Scanner hub regions")]):
        ax = fig.add_subplot(gs[3, col_i * 2:col_i * 2 + 2])
        hs = hub_strength(coef, top)
        tk = np.argsort(hs)[::-1][:8][::-1]            # top-8, ascending for barh
        yy = np.arange(len(tk))
        ax.barh(yy, hs[tk], color=color, height=0.72, edgecolor="white", linewidth=0.4)
        ax.set_yticks(yy)
        ax.set_yticklabels([f"ICN{k+1} · {DOM_NAME[c2d[k]]}" for k in tk], fontsize=5.0)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("hub strength (incident importance)", fontsize=6.0)
        ax.set_xlim(0, hs.max() * 1.02)
        ax.set_title(f"{lab}   {name}", loc="left", fontweight="bold", fontsize=8.5,
                     color=color, pad=4)

    # h: network-domain incidence -----------------------------------------
    ax = fig.add_subplot(gs[3, 4:6])
    ei, ej = iu

    def inc(top):
        c = {d: 0 for d in DOM_ORDER}
        for e in top:
            c[c2d[ei[e]]] += 1; c[c2d[ej[e]]] += 1
        return np.array([c[d] for d in DOM_ORDER])
    dc_d, dc_s = inc(d_top), inc(s_top)
    yy = np.arange(len(DOM_ORDER)); h = 0.4
    ax.barh(yy + h / 2, dc_d, height=h, color=C_DIS, label="disease")
    ax.barh(yy - h / 2, dc_s, height=h, color=C_SCAN, label="scanner")
    ax.set_yticks(yy); ax.set_yticklabels([DOM_NAME[d] for d in DOM_ORDER], fontsize=6.0)
    ax.invert_yaxis(); ax.set_xlabel("top-100 edge endpoints", fontsize=6.5)
    ax.legend(loc="lower right", fontsize=6.0, handlelength=1.0)
    ax.set_title("h   Networks recruited", loc="left", fontweight="bold", fontsize=8.5, pad=4)

    # shared domain colour key (bottom strip)
    fig.text(0.045, 0.018, "Nodes coloured by functional network:  " +
             "   ".join(DOM_NAME[d] for d in DOM_ORDER), fontsize=5.6, color="#5A5E63")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig_brainA.{ext}", dpi=380 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    import pandas as pd
    pd.DataFrame({"domain": DOM_ORDER, "disease_incidence": dc_d,
                  "scanner_incidence": dc_s}).to_csv(SRC / "fig_brain_atlas_incidence.csv", index=False)
    print(f"[brain-atlas] saved fig_brainA  jaccard={jac:.3f} both={n_both}")


if __name__ == "__main__":
    main()
