#!/usr/bin/env python
"""Two reviewer-response figures from cached CSVs (base env, no model):
  fig_decode_causal.pdf  (R2) — decodability vs causal interchange-IIA across a direction battery.
  fig_ablation_sweep.pdf (R5) — scanner stays flat while the disease decision collapses vs #ablated.
TeX Gyre Heros style. Reads outputs/sae_ckpts/{decode_vs_causal,phase6_count_sweep}.csv.
"""
from pathlib import Path
import numpy as np, pandas as pd
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
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.6, "mathtext.fontset": "stixsans",
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7, "legend.frameon": False,
})
MM = 1 / 25.4
SAE = Path("outputs/sae_ckpts"); OUT = Path("manuscript/figures"); OUT.mkdir(parents=True, exist_ok=True)


def fig_decode_causal():
    d = pd.read_csv(SAE / "decode_vs_causal.csv")
    col = {"probe": "#C0392B", "inlp": "#2471A3", "pca": "#E67E22", "random": "#7f8c8d"}
    lab = {"probe": "scanner probe (=DAS axis)", "inlp": "INLP nullspace dirs",
           "pca": "principal components", "random": "random dirs"}
    fig, ax = plt.subplots(figsize=(88 * MM, 78 * MM))
    for k in ["random", "pca", "inlp", "probe"]:
        s = d[d.kind == k]
        ax.scatter(s.decodability, s.scanner_iia, s=42 if k == "probe" else 24,
                   color=col[k], alpha=0.9, edgecolor="white", lw=0.4,
                   marker="*" if k == "probe" else "o", label=lab[k], zorder=3 if k == "probe" else 2)
    ax.axhline(0.124, ls=":", lw=0.7, color="#7f8c8d")
    ax.annotate("causal floor 0.12", (0.5, 0.135), fontsize=6, color="#7f8c8d")
    ax.annotate("decodable but\nNOT causal", (0.80, 0.27), fontsize=6.3, color="#2471A3", ha="center")
    ax.set(xlabel="correlational scanner decodability (AUC)",
           ylabel="causal interchange-IIA (disease-preserved)", xlim=(0.48, 1.0), ylim=(0.0, 1.0))
    ax.set_title("Decodability $\\neq$ causality", loc="left", fontweight="bold")
    ax.legend(loc="upper left", handletextpad=0.2, borderpad=0.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_decode_causal.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig); print(f"wrote {OUT}/fig_decode_causal.pdf")


def fig_ablation_sweep():
    s = pd.read_csv(SAE / "phase6_count_sweep.csv")
    fig, ax = plt.subplots(figsize=(88 * MM, 78 * MM))
    ax.plot(s.n_ablated, s.scanner_pop_auc, "-o", ms=4, color="#C0392B", label="scanner (held-out)")
    ax.plot(s.n_ablated, s.disease_logits, "-s", ms=3.5, color="#1E8449", label="disease decision (logit)")
    ax.axhline(0.5, ls=":", lw=0.7, color="#7f8c8d")
    ax.set(xlabel="# scanner features ablated", ylabel="held-out AUC", ylim=(0.45, 1.0))
    ax.set_title("Scanner survives every ablation scale; disease collapses", loc="left", fontweight="bold")
    ax.legend(loc="center right", handletextpad=0.3)
    ax.annotate("100 features already\nbreak the decision", (s.n_ablated.iloc[1] + 200, 0.66),
                fontsize=6.3, color="#1E8449", va="center")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_ablation_sweep.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig); print(f"wrote {OUT}/fig_ablation_sweep.pdf")


if __name__ == "__main__":
    fig_decode_causal()
    fig_ablation_sweep()
