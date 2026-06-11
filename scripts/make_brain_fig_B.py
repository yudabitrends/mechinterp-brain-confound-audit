#!/usr/bin/env python
"""Figure B (8 panels): structural anatomy of the confound + the network atlas.
a VBM voxel-wise scanner-AUC glass brain; b VBM disease-AUC glass brain; c VBM scanner axial montage;
d VBM scanner sagittal montage; e FreeSurfer subcortical SCANNER discriminability markers; f FreeSurfer
subcortical DISEASE markers; g Neuromark 53-ICN atlas on the brain; h per-voxel scanner-vs-disease density.
Run in `project`. Writes manuscript/figures/fig_brainB.pdf.
"""
import os, sys, glob, re
from pathlib import Path
import numpy as np, nibabel as nib
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
import make_brain_fig_A as A  # rcParams + palette
import torch

MM = 1 / 25.4
MAPS = "/data/users1/ybi/mechinterp_brain/struct_maps"
FS = "/data/users1/ybi/geometric_multivit/freesurfer_subjects"
NMARK = "/data/qneuromark/Network_templates/NeuroMark1/Neuromark_fMRI_1.0.nii"
OUT = Path("/home/users/ybi3/mechinterp_brain/manuscript/figures")
C_SCAN, C_DIS = A.C_SCAN, A.C_DIS
# approx MNI centroids (mm) for the main aseg subcortical structures
CENT = {"Thalamus": (12, -18, 8), "Caudate": (13, 10, 9), "Putamen": (24, 1, 2), "Pallidum": (18, -2, 0),
        "Hippocampus": (26, -20, -14), "Amygdala": (22, -4, -16), "Accumbens-area": (9, 10, -7)}


def fs_subcortical_auc():
    """per-structure scanner(US/China) & disease(SZ/HC) AUC from FreeSurfer aseg volumes."""
    d = torch.load("/home/users/ybi3/mechinterp_brain/outputs/activations/fused_HOLD_ALL.pt", weights_only=True)
    fmap = {str(s): (int(y), str(p)) for s, y, p in zip(d["subject_id"], np.asarray(d["y_dx"]),
                                                        np.asarray(d["population"]))}
    structs = [f"{h}-{n}" for n in CENT for h in ("Left", "Right")]
    rows = []
    for sub in os.listdir(FS):
        ap = f"{FS}/{sub}/stats/aseg.stats"
        if not os.path.exists(ap): continue
        pref = sub.split("_")[0]
        if pref not in ("COBRE", "FBIRN", "ChineseSZ", "Scanner1", "Scanner2", "Scanner3"): continue
        pop = "China" if pref == "ChineseSZ" else "US"
        bare = re.sub(r"^(COBRE_COBRE_|FBIRN_FBIRN_|ChineseSZ_|Scanner[0-9]_)", "", sub)
        dx = fmap[bare][0] if bare in fmap else (1 if "SZ-" in sub else 0 if "NC-" in sub else None)
        vol = {}
        for ln in open(ap):
            p = ln.split()
            if len(p) >= 5 and p[4] in structs:
                try: vol[p[4]] = float(p[3])
                except ValueError: pass
        rows.append((pop, dx, vol))
    # bilateral mean per structure
    out = {}
    for n in CENT:
        v = np.array([0.5 * (r[2].get(f"Left-{n}", np.nan) + r[2].get(f"Right-{n}", np.nan)) for r in rows])
        pop = np.array([r[0] for r in rows]); dx = np.array([r[1] if r[1] is not None else -1 for r in rows])
        ok = ~np.isnan(v)
        scan = roc_auc_score((pop[ok] == "China").astype(int), v[ok])
        dm = ok & (dx >= 0)
        dis = roc_auc_score(dx[dm], v[dm])
        out[n] = (max(scan, 1 - scan), max(dis, 1 - dis))
    return out


def main():
    from nilearn import plotting
    scan = nib.load(f"{MAPS}/struct_scanner_auc.nii.gz"); dis = nib.load(f"{MAPS}/struct_disease_auc.nii.gz")
    vals = np.load(f"{MAPS}/struct_auc_vals.npz"); a_s, a_d = vals["scanner"], vals["disease"]
    fsa = fs_subcortical_auc()

    from nilearn import datasets, surface
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    FS = datasets.fetch_surf_fsaverage("fsaverage5")

    def surf_row(row_spec, img, cmap, vmax, label, color, thr=0.10):
        """Render a volumetric map on the inflated cortical surface (lateral+medial, both hemispheres) as four
        embedded 3D axes plus a slim colorbar. Pure matplotlib (no VTK), so it is headless-safe. The volume is
        projected with vol_to_surf, which captures only cortical signal -- deep/subcortical signal is shown by
        the montages (c, d) below."""
        tex = {h: surface.vol_to_surf(img, FS["pial_" + h]) for h in ("left", "right")}
        pos = row_spec.get_position(fig)                       # label via fig.text (3D-axes titles are unreliable)
        fig.text(pos.x0 + 0.004, pos.y1 - 0.004, label, fontsize=8.0, fontweight="bold", color=color,
                 ha="left", va="top")
        sub = row_spec.subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.0)
        for i, (h, v) in enumerate([("left", "lateral"), ("left", "medial"),
                                    ("right", "medial"), ("right", "lateral")]):
            sax = fig.add_subplot(sub[0, i], projection="3d")
            plotting.plot_surf_stat_map(FS["infl_" + h], tex[h], hemi=h, view=v, colorbar=False,
                threshold=thr, vmax=vmax, cmap=cmap, bg_map=FS["sulc_" + h], axes=sax)
            try: sax.set_box_aspect(None, zoom=1.85)   # zoom in so each brain fills its cell (no figure growth)
            except Exception: pass
        cax = fig.add_subplot(sub[0, 4]); sm = ScalarMappable(Normalize(thr, vmax), cmap); sm.set_array([])
        cb = fig.colorbar(sm, cax=cax); cb.set_ticks([thr, vmax]); cb.ax.tick_params(labelsize=5, length=2)

    fig = plt.figure(figsize=(183 * MM, 192 * MM))
    # a,b are now inflated cortical-SURFACE renders (publication-grade, headless via matplotlib); the voxel
    # montages c,d are retained because the surface projection drops deep/subcortical signal, which they show.
    gs = fig.add_gridspec(6, 2, height_ratios=[0.74, 0.74, 0.48, 0.48, 0.95, 0.95], hspace=0.26, wspace=0.04,
                          left=0.025, right=0.99, top=0.935, bottom=0.04)
    fig.text(0.025, 0.968, "Structural anatomy of the confound and the functional-network atlas",
             fontsize=10.5, fontweight="bold")

    surf_row(gs[0, :], scan, "autumn_r", 0.45, "a   VBM scanner (US vs.\\ China, |AUC$-$0.5|), cortical surface", C_SCAN)
    surf_row(gs[1, :], dis, "winter_r", 0.45, "b   VBM disease (SZ vs.\\ HC), cortical surface", C_DIS)

    ax = fig.add_subplot(gs[2, :]); plotting.plot_stat_map(scan, display_mode="z", cut_coords=7, axes=ax,
        colorbar=False, threshold=0.10, cmap="autumn_r", vmax=0.45, annotate=False, black_bg=False)
    ax.set_title("c   VBM scanner, deep/subcortical (axial montage)", loc="left", fontweight="bold", fontsize=8.0, color=C_SCAN, y=1.0)
    ax = fig.add_subplot(gs[3, :]); plotting.plot_stat_map(scan, display_mode="x", cut_coords=7, axes=ax,
        colorbar=False, threshold=0.10, cmap="autumn_r", vmax=0.45, annotate=False, black_bg=False)
    ax.set_title("d   VBM scanner, deep/subcortical (sagittal montage)", loc="left", fontweight="bold", fontsize=8.0, color=C_SCAN, y=1.0)

    coords = np.array([CENT[n] for n in CENT])
    sc_v = np.array([fsa[n][0] - 0.5 for n in CENT]); di_v = np.array([fsa[n][1] - 0.5 for n in CENT])
    vmx = float(max(sc_v.max(), di_v.max(), 0.05)) * 1.05
    ax = fig.add_subplot(gs[4, 0]); plotting.plot_markers(sc_v, coords, node_cmap="autumn_r", node_vmin=0,
        node_vmax=vmx, display_mode="lzr", axes=ax, colorbar=True, node_size=40 + 420 * sc_v / vmx)
    ax.set_title("e   FreeSurfer subcortical scanner |AUC$-$0.5|", loc="left", fontweight="bold", fontsize=8.0, color=C_SCAN, y=0.97)
    ax = fig.add_subplot(gs[4, 1]); plotting.plot_markers(di_v, coords, node_cmap="winter_r", node_vmin=0,
        node_vmax=vmx, display_mode="lzr", axes=ax, colorbar=True, node_size=40 + 420 * di_v / vmx)
    ax.set_title("f   FreeSurfer subcortical disease |AUC$-$0.5|", loc="left", fontweight="bold", fontsize=8.0, color=C_DIS, y=0.97)

    ax = fig.add_subplot(gs[5, 0])
    try:
        plotting.plot_prob_atlas(NMARK, display_mode="z", cut_coords=5, axes=ax, colorbar=False,
                                 view_type="filled_contours", linewidths=0.4)
    except Exception as e:
        ax.text(0.5, 0.5, f"atlas render skipped", ha="center", fontsize=6); print("atlas:", e)
    ax.set_title("g   Neuromark 53-ICN atlas", loc="left", fontweight="bold", fontsize=8.0, y=1.0)

    ax = fig.add_subplot(gs[5, 1])
    mx = max(np.abs(a_s - 0.5).max(), np.abs(a_d - 0.5).max())
    ax.hist2d(np.abs(a_s - 0.5), np.abs(a_d - 0.5), bins=70, cmin=1, cmap="magma_r")
    ax.plot([0, mx], [0, mx], ls=":", lw=0.8, color="#777")
    ax.set(xlabel="scanner |AUC$-$0.5|", ylabel="disease |AUC$-$0.5|", xlim=(0, mx), ylim=(0, mx))
    ax.set_title("h   Per-voxel dissociation", loc="left", fontweight="bold", fontsize=8.0, y=1.0)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_brainB.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("[fig_brainB] saved; subcortical scanner |AUC-0.5|:",
          {n: round(fsa[n][0] - 0.5, 2) for n in CENT})


if __name__ == "__main__":
    main()
