#!/usr/bin/env python
"""Stage D: the 4 main figures. Nature/TMI style, reusing the TeX Gyre Heros boilerplate
from MultiViT2/scripts/make_figures.py. Each figure is skipped gracefully if its source
CSV is absent. Outputs PDF+PNG to outputs/figures/.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- font + rcParams (verbatim from MultiViT2/scripts/make_figures.py) -----------------
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
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "mathtext.fontset": "stixsans",
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.6, "legend.frameon": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
})
MM = 1 / 25.4
SAE = "outputs/sae_ckpts"
OUT = Path("outputs/figures"); OUT.mkdir(parents=True, exist_ok=True)
C = {"sMRI": "#c0392b", "FNC": "#2471a3", "cross": "#7d3c98", "disease": "#1e8449", "scanner": "#c0392b"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf", flush=True)


def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(180 * MM, 52 * MM)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 28)
    blocks = [("sMRI-ViT\n+ FNC-transformer\n+ cross-attn fusion\n(MultiViT2)", 9, C["cross"]),
              ("forward hooks\n26 layers", 29, "#566573"),
              ("Sparse\nAutoencoders\n(token-level)", 48, "#566573"),
              ("probes:\ndisease vs\nscanner", 67, C["disease"]),
              ("subspace\nerasure\n(harmonize)", 86, C["scanner"])]
    for label, x, col in blocks:
        ax.add_patch(FancyBboxPatch((x - 7.5, 9), 15, 10, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="white", ec=col, lw=1.1))
        ax.text(x, 14, label, ha="center", va="center", fontsize=6.6, color=col)
        if x < 86:
            ax.add_patch(FancyArrowPatch((x + 7.6, 14), (x + 12.4, 14), arrowstyle="-|>",
                                         mutation_scale=8, lw=0.8, color="#34495e"))
    ax.text(50, 25, "Mechanistic dissection of scanner confounds in a multimodal brain-disorder classifier",
            ha="center", fontsize=8.5)
    save(fig, "fig1_pipeline")


def fig2_atlas():
    gp = Path(f"{SAE}/phase1_gate_HOLDtrain.csv")
    if not gp.exists():
        print("fig2: gate csv missing, skip"); return
    g = pd.read_csv(gp); g = g.groupby("hook")["n_stable"].mean().reset_index()
    def depth(h):
        return int(h.split("blocks.")[1].split(".")[0]) if "blocks." in h else (99 if "norm" in h else -1)
    g["depth"] = g.hook.apply(depth); g["branch"] = np.where(g.hook.str.startswith("sMRI"), "sMRI",
                                       np.where(g.hook.str.startswith("sFNC"), "FNC", "cross"))
    fig, ax = plt.subplots(figsize=(90 * MM, 62 * MM))
    for br in ["FNC", "sMRI"]:
        s = g[(g.branch == br) & (g.depth >= 0) & (g.depth < 90) & (~g.hook.str.contains("mlp"))].sort_values("depth")
        ax.plot(s.depth, s.n_stable, "-o", ms=4, lw=1.4, color=C[br], label=f"{br} residual")
    ax.set_xlabel("transformer block depth"); ax.set_ylabel("seed-stable SAE features")
    ax.set_title("Atlas stability rises with depth (FNC)"); ax.legend()
    save(fig, "fig2_atlas_stability")


def fig3_dissociation():
    pp = Path(f"{SAE}/phase2_probe_HOLDtrain_seed0_test.csv")
    if not pp.exists():
        print("fig3: phase2 test csv missing, skip"); return
    d = pd.read_csv(pp)
    d["branch"] = np.where(d.hook.str.startswith("sMRI"), "sMRI",
                  np.where(d.hook.str.startswith("sFNC"), "FNC", "cross"))
    fig, ax = plt.subplots(figsize=(80 * MM, 76 * MM))
    for br in ["sMRI", "FNC", "cross"]:
        s = d[d.branch == br]
        ax.scatter(s.pop_auc_sae, s.dx_auc_sae, s=26, color=C[br], label=br, alpha=0.85, edgecolor="white", lw=0.4)
    ax.plot([0.5, 1], [0.5, 1], ls=":", lw=0.7, color="#95a5a6")
    ax.set_xlabel("scanner (population) AUC"); ax.set_ylabel("disease AUC")
    ax.set_title("Modality dissociation (held-out)"); ax.legend(loc="lower left")
    ax.set_xlim(0.5, 1.0); ax.set_ylim(0.5, 1.0)
    save(fig, "fig3_modality_dissociation")


def fig4_causal():
    hp = Path(f"{SAE}/harmonize_compare.csv")
    sp = Path(f"{SAE}/phase3b_rounds_sweep_HOLD.csv")  # held-out (non-ceiling); falls back below
    if not sp.exists():
        sp = Path(f"{SAE}/phase3b_rounds_sweep.csv")
    if not hp.exists():
        print("fig4: harmonize csv missing, skip"); return
    h = pd.read_csv(hp)
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 64 * MM))
    # panel A: harmonization comparison bars
    ax = axes[0]; m = h.set_index("method")
    order = [x for x in ["raw", "ComBat", "site_regression", "random_erasure", "INLP_scanner_erasure"] if x in m.index]
    x = np.arange(len(order)); w = 0.27
    ax.bar(x - w, m.loc[order, "scanner_pop_auc"], w, label="scanner", color=C["scanner"])
    ax.bar(x, m.loc[order, "site_auc"], w, label="site", color="#e67e22")
    ax.bar(x + w, m.loc[order, "disease_auc"], w, label="disease", color=C["disease"])
    ax.axhline(0.5, ls=":", lw=0.7, color="#95a5a6")
    ax.set_xticks(x); ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=6.2)
    ax.set_ylabel("held-out AUC"); ax.set_title("Harmonization (held-out test)"); ax.legend(ncol=3, fontsize=6)
    # panel B: erasure tradeoff curve
    ax = axes[1]
    if sp.exists():
        s = pd.read_csv(sp)
        ax.plot(s.rounds, s.scanner_pop_auc, "-o", ms=4, color=C["scanner"], label="scanner")
        ax.plot(s.rounds, s.disease_auc, "-s", ms=4, color=C["disease"], label="disease")
        ax.axhline(0.5, ls=":", lw=0.7, color="#95a5a6")
        ax.set_xlabel("erased rank (INLP rounds)"); ax.set_ylabel("AUC")
        ax.set_title("Erasure tradeoff"); ax.legend()
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "tradeoff sweep pending", ha="center")
    save(fig, "fig4_causal_intervention")


if __name__ == "__main__":
    fig1_pipeline(); fig2_atlas(); fig3_dissociation(); fig4_causal()
    print("done.")
