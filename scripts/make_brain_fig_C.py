#!/usr/bin/env python
"""Figure C (8 panels): the clean acquisition axis and cross-disorder replication, on the brain.
a US within-country (COBRE vs Scanner2) site-axis connectome; b China (GZ vs ZMD) site-axis connectome;
c autism (ABIDE) scanner-edge connectome; d autism disease-edge connectome; e/f/g network-block matrices for
US-site / China-site / ABIDE-scanner; h cross-axis consistency (edge-importance correlation across the SZ
population scanner axis, the two within-country site axes, and the ABIDE scanner axis). Reuses make_brain_fig_A
helpers. Run in `project`. Writes manuscript/figures/fig_brainC.pdf.
"""
import os, sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import make_brain_fig_A as A   # helpers + rcParams (TeX Gyre Heros, palette)

MM = 1 / 25.4
REPO = Path(__file__).resolve().parent.parent
SITE = REPO / "outputs" / "fnc_importance_site"
ABIDE = REPO / "outputs" / "fnc_importance_abide"
SZ = REPO / "outputs" / "fnc_importance"
OUT = REPO / "manuscript" / "figures"
C_SCAN, C_DIS = A.C_SCAN, A.C_DIS


def connectome(ax, coef, top, cmap, col, lab, name, c2d, coords):
    from nilearn import plotting
    Adj = A.adjacency(coef, top, 60)
    nd = 9 + 42 * (Adj.sum(0) / (Adj.sum(0).max() + 1e-9))
    plotting.plot_connectome(Adj, coords, node_color=[A.DOM_COL[c2d[i]] for i in range(A.N_ICN)],
                             node_size=nd, edge_cmap=A.trunc(cmap), edge_vmin=0, edge_vmax=1,
                             edge_threshold="0.1%", display_mode="lzr", axes=ax, colorbar=False,
                             node_kwargs={"edgecolors": "white", "linewidths": 0.3},
                             edge_kwargs={"linewidth": 1.4, "alpha": 0.85})
    ax.set_title(f"{lab}   {name}", loc="left", fontweight="bold", fontsize=8.0, color=col, y=0.96)


def main():
    c2d = A.domains()
    coords = np.load(f"{SITE}/icn_mni_coords.npy")
    us = np.load(f"{SITE}/US_COBRE_vs_Scanner2_coef_abs.npy").astype(float)
    us_t = np.load(f"{SITE}/US_COBRE_vs_Scanner2_top100.npy")
    cn = np.load(f"{SITE}/China_GZ_vs_ZMD_coef_abs.npy").astype(float)
    cn_t = np.load(f"{SITE}/China_GZ_vs_ZMD_top100.npy")
    ab_s = np.load(f"{ABIDE}/scanner_coef_abs.npy").astype(float); ab_st = np.load(f"{ABIDE}/scanner_top100.npy")
    ab_d = np.load(f"{ABIDE}/disease_coef_abs.npy").astype(float); ab_dt = np.load(f"{ABIDE}/disease_top100.npy")
    sz_s = np.load(f"{SZ}/scanner_coef_abs.npy").astype(float)

    fig = plt.figure(figsize=(183 * MM, 206 * MM))
    # Connectome rows (0,1) get more height and near-zero wspace so each glass-brain network renders wider;
    # the compact block-matrix / correlation panels (rows 2,3) read fine at a smaller share of the area.
    gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 1.15, 0.78, 0.78], hspace=0.42, wspace=0.12,
                          left=0.035, right=0.985, top=0.945, bottom=0.05)
    fig.text(0.035, 0.965, "Scanner edges are distributed and cohort-idiosyncratic: the spatial face of redundancy",
             fontsize=10.5, fontweight="bold")

    connectome(fig.add_subplot(gs[0, 0]), us, us_t, "Reds", C_SCAN, "a", "US within-country site axis", c2d, coords)
    connectome(fig.add_subplot(gs[0, 1]), cn, cn_t, "Reds", C_SCAN, "b", "China within-country site axis", c2d, coords)
    connectome(fig.add_subplot(gs[1, 0]), ab_s, ab_st, "Reds", C_SCAN, "c", "Autism (ABIDE) scanner axis", c2d, coords)
    connectome(fig.add_subplot(gs[1, 1]), ab_d, ab_dt, "Blues", C_DIS, "d", "Autism (ABIDE) disease edges", c2d, coords)

    A.block_panel(fig.add_subplot(gs[2, 0]), A.block_matrix(us, c2d), "Reds", "US site network-blocks", "e")
    A.block_panel(fig.add_subplot(gs[2, 1]), A.block_matrix(cn, c2d), "Reds", "China site network-blocks", "f")
    A.block_panel(fig.add_subplot(gs[3, 0]), A.block_matrix(ab_s, c2d), "Reds", "Autism scanner network-blocks", "g")

    # h: cross-axis consistency of scanner/acquisition edge importance
    ax = fig.add_subplot(gs[3, 1])
    names = ["SZ pop", "US site", "China site", "ASD scan"]
    V = np.vstack([sz_s, us, cn, ab_s])
    R = np.corrcoef(V)
    im = ax.imshow(R, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(names, fontsize=5.4, rotation=40, ha="right"); ax.set_yticklabels(names, fontsize=5.4)
    ax.tick_params(length=0)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center", fontsize=5.0,
                    color="white" if R[i, j] < 0.6 else "black")
    ax.set_title("h   Cross-axis edge correlation $\\approx 0$", loc="left", fontweight="bold", fontsize=8.0, pad=4)

    fig.text(0.06, 0.016, "Nodes coloured by functional network:  " +
             "   ".join(A.DOM_NAME[d] for d in A.DOM_ORDER), fontsize=5.6, color="#5A5E63")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_brainC.{ext}", dpi=360 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_brainC] saved; cross-axis corr SZ-vs-US={R[0,1]:.2f} SZ-vs-China={R[0,2]:.2f} "
          f"SZ-vs-ASD={R[0,3]:.2f}")


if __name__ == "__main__":
    main()
