#!/usr/bin/env python
"""Structural (sMRI/VBM) brain panel: where the scanner confound lives structurally vs. where disease
lives. Reads the voxel-wise AUC NIfTIs from compute_struct_maps.py. nilearn render. Run in `project` env.

Big panel (4 sub-panels):
  (a) glass-brain of voxel-wise SCANNER discriminability (US-vs-China |AUC-0.5|) on GM
  (b) glass-brain of voxel-wise DISEASE (SZ-vs-HC) discriminability
  (c) axial montage of the scanner map (the localized 'scanner machine')
  (d) per-voxel scanner-AUC vs disease-AUC density (the structural modality dissociation)
Writes manuscript/figures/fig_struct_panel.pdf.
"""
import os
from pathlib import Path
import numpy as np, nibabel as nib
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_HEROS = Path("/home/users/ybi3/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre")
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
MAPS = "/data/users1/ybi/mechinterp_brain/struct_maps"
OUT = Path("/home/users/ybi3/mechinterp_brain/manuscript/figures"); OUT.mkdir(parents=True, exist_ok=True)
C_SCAN, C_DIS, INK = "#B64342", "#1F5FA6", "#222428"


def main():
    from nilearn import plotting, image
    scan = nib.load(f"{MAPS}/struct_scanner_auc.nii.gz")     # signed (AUC-0.5)
    dis = nib.load(f"{MAPS}/struct_disease_auc.nii.gz")
    vals = np.load(f"{MAPS}/struct_auc_vals.npz")
    a_scan, a_dx = vals["scanner"], vals["disease"]          # raw AUC on masked voxels
    thr = 0.10                                               # display |AUC-0.5| threshold

    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.9], hspace=0.42, wspace=0.18,
                          left=0.05, right=0.97, top=0.92, bottom=0.07)
    fig.text(0.05, 0.965, "Structural (sMRI/VBM): where scanner vs. disease live in gray matter",
             fontsize=10.5, fontweight="bold")

    # (a) scanner glass brain
    ax = fig.add_subplot(gs[0, :])
    plotting.plot_glass_brain(scan, display_mode="lyrz", axes=ax, colorbar=True, threshold=thr,
                              cmap="autumn_r", vmax=0.45, plot_abs=False,
                              title=None)
    ax.set_title("a   Scanner discriminability (US vs. China, |AUC$-$0.5|)", loc="left",
                 fontweight="bold", fontsize=9, color=C_SCAN, y=0.97)

    # (b) disease glass brain
    ax = fig.add_subplot(gs[1, :])
    plotting.plot_glass_brain(dis, display_mode="lyrz", axes=ax, colorbar=True, threshold=thr,
                              cmap="winter_r", vmax=0.45, plot_abs=False)
    ax.set_title("b   Disease discriminability (SZ vs. HC, |AUC$-$0.5|)", loc="left",
                 fontweight="bold", fontsize=9, color=C_DIS, y=0.97)

    # (c) scanner axial montage
    ax = fig.add_subplot(gs[2, 0])
    plotting.plot_stat_map(scan, display_mode="z", cut_coords=6, axes=ax, colorbar=False,
                           threshold=thr, cmap="autumn_r", vmax=0.45, annotate=False, black_bg=False)
    ax.set_title("c   Scanner map (axial)", loc="left", fontweight="bold", fontsize=9, color=C_SCAN, y=1.0)

    # (d) per-voxel scanner vs disease AUC density
    ax = fig.add_subplot(gs[2, 1])
    h = ax.hist2d(np.abs(a_scan - 0.5), np.abs(a_dx - 0.5), bins=80, cmin=1, cmap="magma_r")
    mx = max(np.abs(a_scan - 0.5).max(), np.abs(a_dx - 0.5).max())
    ax.plot([0, mx], [0, mx], ls=":", lw=0.8, color="#777")
    ax.set(xlabel="scanner |AUC$-$0.5|", ylabel="disease |AUC$-$0.5|", xlim=(0, mx), ylim=(0, mx))
    ax.set_title("d   Per-voxel dissociation", loc="left", fontweight="bold", fontsize=9, y=1.0)
    ax.annotate("scanner $\\gg$ disease\nin most GM voxels", (0.62 * mx, 0.18 * mx), fontsize=6.2,
                color=INK, ha="center")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_struct_panel.{ext}", dpi=360 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"scanner |AUC-0.5|: mean {np.abs(a_scan-0.5).mean():.3f} max {np.abs(a_scan-0.5).max():.3f}; "
          f"disease: mean {np.abs(a_dx-0.5).mean():.3f} max {np.abs(a_dx-0.5).max():.3f}")
    print(f"wrote {OUT}/fig_struct_panel.pdf")


if __name__ == "__main__":
    main()
