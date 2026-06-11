#!/usr/bin/env python
"""Main results figure (4 panels, full-width) for the TMI paper. Tells the arc:
A localize -> B distributed (feature ablation fails) -> C causal compression (DAS/IIA)
-> D subspace removal harmonizes. Nature-style (TeX Gyre Heros). Writes manuscript/figures/fig_main.pdf.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_HEROS = Path("/home/users/ybi3/texlive/2026/texmf-dist/fonts/opentype/public/tex-gyre")
for _f in ("texgyreheros-regular.otf", "texgyreheros-bold.otf", "texgyreheros-italic.otf"):
    try: fm.fontManager.addfont(str(_HEROS / _f))
    except Exception: pass
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "mathtext.fontset": "stixsans",
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7, "legend.frameon": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.6, "ytick.major.size": 2.6,
})
MM = 1 / 25.4
SAE = Path("outputs/sae_ckpts")
OUT = Path("manuscript/figures"); OUT.mkdir(parents=True, exist_ok=True)
C = dict(sMRI="#B64342", FNC="#2471A3", cross="#7D3C98", scanner="#B64342", disease="#1E8449",
         grey="#7f8c8d", site="#E67E22")


def panelA(ax):
    d = pd.read_csv(SAE / "phase2_probe_HOLDtrain_seed0_test.csv")
    d["branch"] = np.where(d.hook.str.startswith("sMRI"), "sMRI",
                  np.where(d.hook.str.startswith("sFNC"), "FNC", "cross"))
    for br in ["sMRI", "FNC", "cross"]:
        s = d[d.branch == br]
        ax.scatter(s.pop_auc_sae, s.dx_auc_sae, s=24, color=C[br], alpha=0.85,
                   edgecolor="white", lw=0.4, label=br)
    ax.plot([0.5, 1], [0.5, 1], ls=":", lw=0.7, color=C["grey"])
    ax.set(xlim=(0.5, 1.0), ylim=(0.5, 1.0), xlabel="scanner AUC", ylabel="disease AUC")
    ax.set_title("A  Where: modality dissociation", loc="left", fontweight="bold")
    ax.legend(loc="lower left", handletextpad=0.2, borderpad=0.2)
    ax.annotate("structural branch:\nscanner $\\gg$ disease", (0.86, 0.575), fontsize=6.3,
                color=C["sMRI"], ha="center", va="center")


def panelB(ax):
    d = pd.read_csv(SAE / "phase6_sfc_ablation.csv").set_index("condition")
    order = ["baseline", "graph_ablate_scanner", "random_same_size_ctrl"]
    lab = ["baseline", "ablate scanner\nfeatures (5,726)", "random\ncontrol"]
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w/2, d.loc[order, "scanner_pop_auc"], w, color=C["scanner"], label="scanner")
    ax.bar(x + w/2, d.loc[order, "disease_logits"], w, color="none", edgecolor=C["disease"],
           hatch="////", lw=0.8, label="disease")
    ax.axhline(0.5, ls=":", lw=0.7, color=C["grey"])
    ax.set(ylim=(0.0, 1.0), xticks=x, ylabel="held-out AUC")
    ax.set_xticklabels(lab, fontsize=6.3)
    ax.set_title("B  What: distributed → feature ablation fails", loc="left", fontweight="bold")
    ax.legend(ncol=2, loc="lower center", handletextpad=0.3, columnspacing=1.2,
              bbox_to_anchor=(0.5, 1.0))


def panelC(ax):
    d = pd.read_csv(SAE / "phase5_das_iia.csv")
    ax.plot(d.k, d.scanner_iia, "-o", ms=4, color=C["scanner"], alpha=0.55, label="scanner IIA (single-run sweep)")
    ax.errorbar([1], [0.89], yerr=[0.01], fmt="*", ms=9, color=C["scanner"], capsize=2,
                zorder=6, label="$k{=}1$ headline (4 seeds)")
    ax.plot(d.k, d.disease_preserved, "-s", ms=3.5, color=C["disease"], label="disease preserved")
    ax.plot(d.k, d.scanner_iia_rand, "--^", ms=3.5, color=C["grey"], label="random rotation")
    ax.set_xscale("log", base=2)
    ax.set(xlabel="subspace dim. $k$", ylabel="interchange accuracy", ylim=(0.0, 1.05))
    ax.set_xticks(d.k); ax.set_xticklabels([int(k) for k in d.k])
    ax.axhline(0.5, ls=":", lw=0.7, color=C["grey"])
    ax.set_title("C  Why removable: low-dim. causal compression", loc="left", fontweight="bold")
    ax.legend(loc="center right", handletextpad=0.3)
    ax.annotate("floor 0.12; saturates by $k{=}4$", (1.05, 0.22), fontsize=5.8, color="#444", ha="left")


def panelD(ax):
    s = pd.read_csv(SAE / "phase3b_rounds_sweep_HOLD.csv")
    ax.plot(s.rounds, s.scanner_pop_auc, "-o", ms=4, color=C["scanner"], label="scanner (erasure)")
    ax.plot(s.rounds, s.disease_auc, "-s", ms=3.5, color=C["disease"], label="disease (erasure)")
    ax.axhline(0.5, ls=":", lw=0.7, color=C["grey"])
    h = pd.read_csv(SAE / "harmonize_compare.csv").set_index("method")
    pts = [("ComBat", "D"), ("site_regression", "P"), ("random_erasure", "X")]
    for m, mk in pts:
        if m in h.index:
            ax.scatter(101, h.loc[m, "scanner_pop_auc"], marker=mk, s=34, color=C["scanner"],
                       edgecolor="k", lw=0.3, zorder=5)
            ax.scatter(101, h.loc[m, "disease_auc"], marker=mk, s=34, color=C["disease"],
                       edgecolor="k", lw=0.3, zorder=5)
    ax.set(xlabel="erased rank (INLP rounds)  /  baselines", ylabel="held-out AUC", ylim=(0.45, 1.0))
    ax.set_title("D  So: subspace removal harmonizes", loc="left", fontweight="bold")
    ax.legend(loc="center left", handletextpad=0.3)
    ax.annotate("ComBat$\\,\\diamond$  site-reg$\\,+$\nrandom$\\,\\times$", (101, 0.96), fontsize=6,
                ha="right", va="top", color="#444")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 120 * MM))
    panelA(axes[0, 0]); panelB(axes[0, 1]); panelC(axes[1, 0]); panelD(axes[1, 1])
    fig.tight_layout(w_pad=1.6, h_pad=1.8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_main.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig_main.pdf")


if __name__ == "__main__":
    main()
