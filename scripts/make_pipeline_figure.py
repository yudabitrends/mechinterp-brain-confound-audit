#!/usr/bin/env python
"""Fig 1 pipeline schematic (full-width, flat vector style) for the TMI paper.
4 stages left->right: data -> frozen classifier -> mechanistic toolkit -> findings.
Nature-style (TeX Gyre Heros). Writes manuscript/figures/fig1_pipeline.pdf.
"""
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_H = Path("/home/users/ybi3/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre")
for _f in ("texgyreheros-regular.otf", "texgyreheros-bold.otf"):
    try: fm.fontManager.addfont(str(_H / _f))
    except Exception: pass
mpl.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "mathtext.fontset": "stixsans"})
MM = 1 / 25.4
C = dict(sMRI="#B64342", FNC="#2471A3", fuse="#7D3C98", tool="#566573",
         dis="#1E8449", scan="#B64342", grey="#95a5a6", ink="#2c3e50")
OUT = Path("manuscript/figures"); OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, ec, fc="white", lw=1.1, rad=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={rad}",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def txt(ax, x, y, s, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("fontsize", 6.4)
    kw.setdefault("color", C["ink"]); kw.setdefault("zorder", 3); ax.text(x, y, s, **kw)


def arrow(ax, x0, x1, y, label=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color=C["ink"], zorder=2))
    if label:
        txt(ax, (x0 + x1) / 2, y + 1.6, label, fontsize=5.6, color="#7f8c8d")


def main():
    fig, ax = plt.subplots(figsize=(180 * MM, 50 * MM))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    # Stage 1 — Data
    box(ax, 2, 17, 14, 7, C["sMRI"]); txt(ax, 9, 20.5, "sMRI volume", color=C["sMRI"], fontsize=6.2)
    box(ax, 2, 7, 14, 7, C["FNC"]); txt(ax, 9, 10.5, "FNC matrix\n53$\\times$53", color=C["FNC"], fontsize=6.2)
    txt(ax, 9, 3.2, "4 cohorts, US+China\n$N{=}1{,}253$, held-out 70/30", fontsize=5.4)
    txt(ax, 9, 26.6, "labels: scanner + diagnosis", fontsize=5.4, color="#7f8c8d")

    # Stage 2 — frozen classifier
    box(ax, 24, 5, 22, 20, C["fuse"], fc="#faf6fc", lw=1.3)
    txt(ax, 35, 23, "MultiViT2 (frozen)\nheld-out AUC 0.84", fontsize=6.0, color=C["fuse"])
    box(ax, 26, 16.5, 9, 4, C["sMRI"]); txt(ax, 30.5, 18.5, "sMRI ViT", color=C["sMRI"], fontsize=5.6)
    box(ax, 26, 9.5, 9, 4, C["FNC"]); txt(ax, 30.5, 11.5, "FNC Tx", color=C["FNC"], fontsize=5.6)
    box(ax, 38, 12.5, 6.5, 5, C["fuse"]); txt(ax, 41.2, 15, "fusion\n$h{\\to}$dx", color=C["fuse"], fontsize=5.4)
    for yy in (18.5, 11.5):
        ax.add_patch(FancyArrowPatch((35, yy), (38, 15.2 if yy > 15 else 14.8), arrowstyle="-|>",
                     mutation_scale=6, lw=0.8, color="#999", zorder=2))
    txt(ax, 35, 6.6, "forward hooks (26)", fontsize=5.2, color="#7f8c8d")

    # Stage 3 — toolkit
    tools = ["SAE atlas (4,096)", "linear probing", "feature / circuit ablation", "path-patch + DAS / IIA"]
    for i, t in enumerate(tools):
        yy = 21.5 - i * 5.0
        box(ax, 54, yy, 20, 4.0, C["tool"], fc="#f4f6f7"); txt(ax, 64, yy + 2.0, t, fontsize=5.8)

    # Stage 4 — findings
    box(ax, 80, 16, 18, 9, C["scan"], fc="#fdf3f2")
    txt(ax, 89, 22.4, "distributed", fontweight="bold", color=C["scan"], fontsize=6.4)
    txt(ax, 89, 18.7, "~834 features\nablation null  $\\times$", fontsize=5.6)
    box(ax, 80, 5, 18, 9, C["dis"], fc="#f1f9f4")
    txt(ax, 89, 11.4, "causally compressed", fontweight="bold", color=C["dis"], fontsize=6.0)
    txt(ax, 89, 7.7, "low-dim subspace, IIA 0.89\nremovable  $\\checkmark$", fontsize=5.6)

    arrow(ax, 16.5, 23.5, 15, "audit")
    arrow(ax, 46.5, 53.5, 15, "hooks")
    arrow(ax, 74.5, 79.5, 15, "")
    plt.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_pipeline.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig1_pipeline.pdf")


if __name__ == "__main__":
    main()
