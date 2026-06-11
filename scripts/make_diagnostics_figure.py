#!/usr/bin/env python
"""Composite 'audit diagnostics' figure (Nature-figure standards): aggregates four single-panel
checks into one 2x2 quantitative grid so no lone panels remain.
  (a) SAE atlas stability       -- seed-reproducible features rise into mid-depth functional blocks
  (b) decodability != causality -- INLP dirs decodable yet causally inert; only the probe/DAS axis is both
  (c) ablation-count sweep      -- scanner flat at every scale while the disease decision collapses
  (d) trained DAS vs depth      -- the causal handle consolidates only at the decision representation
Restrained palette (scanner red / disease green / neutral grey), TeX Gyre Heros, editable vector PDF.
Reads outputs/sae_ckpts/{phase1_gate_HOLDtrain, decode_vs_causal, phase6_count_sweep, phase5c_das_depth}.csv.
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
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7.5, "axes.titlesize": 8.5,
    "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.4,
    "mathtext.fontset": "stixsans", "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "legend.frameon": False, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
})
MM = 1 / 25.4
SAE = Path("outputs/sae_ckpts"); OUT = Path("manuscript/figures"); OUT.mkdir(parents=True, exist_ok=True)
SCAN, DIS, GREY, FNC, SMRI = "#B64342", "#1E8449", "#767676", "#2471A3", "#C0392B"


def title(ax, lab, txt):
    ax.set_title(f"$\\bf{{{lab}}}$   {txt}", loc="left", fontsize=8.3, pad=4)


def panelA(ax):
    g = pd.read_csv(SAE / "phase1_gate_HOLDtrain.csv").groupby("hook")["n_stable"].mean().reset_index()
    def depth(h): return int(h.split("blocks.")[1].split(".")[0]) if "blocks." in h else (99 if "norm" in h else -1)
    g["depth"] = g.hook.apply(depth)
    g["branch"] = np.where(g.hook.str.startswith("sMRI"), "sMRI",
                  np.where(g.hook.str.startswith("sFNC"), "FNC", "cross"))
    for br, col in [("FNC", FNC), ("sMRI", SMRI)]:
        s = g[(g.branch == br) & (g.depth >= 0) & (g.depth < 90) & (~g.hook.str.contains("mlp"))].sort_values("depth")
        ax.plot(s.depth, s.n_stable, "-o", ms=4, lw=1.5, color=col, label=f"{br} branch")
    ax.set(xlabel="transformer block depth", ylabel="seed-stable SAE features")
    ax.legend(loc="upper left", handletextpad=0.4)
    title(ax, "a", "SAE atlas is stable")


def panelB(ax):
    d = pd.read_csv(SAE / "decode_vs_causal.csv")
    col = {"probe": SCAN, "inlp": FNC, "pca": "#E67E22", "random": GREY}
    lab = {"probe": "probe (=DAS axis)", "inlp": "INLP dirs", "pca": "PCs", "random": "random"}
    for k in ["random", "pca", "inlp", "probe"]:
        s = d[d.kind == k]
        ax.scatter(s.decodability, s.scanner_iia, s=46 if k == "probe" else 22, color=col[k],
                   marker="*" if k == "probe" else "o", edgecolor="white", lw=0.35,
                   alpha=0.92, zorder=3 if k == "probe" else 2, label=lab[k])
    ax.axhline(0.124, ls=":", lw=0.7, color=GREY)
    ax.annotate("decodable but\nnot causal", (0.80, 0.28), fontsize=6.0, color=FNC, ha="center")
    ax.set(xlabel="decodability (AUC)", ylabel="causal interchange-$\\mathrm{IIA}$", xlim=(0.48, 1.0), ylim=(0, 1.0))
    ax.legend(loc="upper left", handletextpad=0.2, labelspacing=0.25)
    title(ax, "b", "decodability $\\neq$ causality")


def panelC(ax):
    s = pd.read_csv(SAE / "phase6_count_sweep.csv")
    ax.plot(s.n_ablated, s.scanner_pop_auc, "-o", ms=4, lw=1.4, color=SCAN, label="scanner")
    ax.plot(s.n_ablated, s.disease_logits, "-s", ms=3.6, lw=1.4, color=DIS, label="disease decision")
    ax.axhline(0.5, ls=":", lw=0.7, color=GREY)
    ax.set(xlabel="# scanner features ablated", ylabel="held-out AUC", ylim=(0.45, 1.0))
    ax.legend(loc="center right", handletextpad=0.4)
    ax.annotate("100 already\nbreak disease", (s.n_ablated.iloc[1] + 250, 0.64), fontsize=6.0, color=DIS, va="center")
    title(ax, "c", "scanner survives every ablation scale")


def panelD(ax):
    d = pd.read_csv(SAE / "phase5c_das_depth.csv")
    x = np.arange(len(d))
    ax.plot(x, d.iia_trained, "-o", ms=4, lw=1.5, color=SCAN, label="trained DAS")
    ax.plot(x, d.iia_untrained, "--^", ms=3.4, lw=1.2, color=GREY, label="untrained")
    ax.axhline(0.89, ls="-", lw=1.0, color=DIS, alpha=0.8)
    ax.annotate("decision rep $0.89$", (0.2, 0.91), fontsize=6.2, color=DIS)
    ax.set_xticks(x); ax.set_xticklabels([h.replace("sFNC_encoder.", "").replace("cross_blocks.", "fus")
                                          .replace(".ca_b", "").replace("blocks.", "blk")
                                          for h in d.hook], rotation=40, ha="right", fontsize=5.6)
    ax.set(xlabel="hook (early $\\rightarrow$ decision)", ylabel="downstream scanner-$\\mathrm{IIA}$", ylim=(0, 1.0))
    ax.legend(loc="upper right", handletextpad=0.4)
    title(ax, "d", "strong causal handle only at the decision rep")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(178 * MM, 118 * MM))
    panelA(axes[0, 0]); panelB(axes[0, 1]); panelC(axes[1, 0]); panelD(axes[1, 1])
    fig.text(0.012, 0.975, "Audit diagnostics", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=2.2, h_pad=2.6)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_diagnostics.{ext}", dpi=500 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig_diagnostics.pdf")


if __name__ == "__main__":
    main()
