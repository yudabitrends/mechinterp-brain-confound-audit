# A causal-exposure release criterion for scanner confounds in multi-site brain-disorder classifiers

Code for the paper *"Residual decodability is not enough: a causal-exposure release criterion for scanner
confounds in multi-site brain-disorder classifiers"* (under review at *Medical Image Analysis*).

We open a **frozen** multi-site brain-disorder classifier (sMRI 3D-ViT + functional-connectivity transformer +
cross-attention fusion) with mechanistic interpretability and ask whether the **scanner/site** confound is merely
*decodable* or actually *causally used* by the disease decision. The central finding: the confound is
**redundantly distributed** (no sparse handle; feature/circuit ablation is null at every scale) yet **causally
compressed** onto a single decision direction, which is why feature-level repair fails while a decision-level
linear correction succeeds and transfers to unseen scanners. The practical consequence is a **release criterion**:
certify a model on residual **causal exposure** (measured on the model's own decision), not on residual
decodability.

> Throughout, "causal" denotes interventional influence **within the frozen model's computation graph**
> (intervening on activations and reading the model's own decision), not clinical or biological causation.

## Repository layout

```
src/mib/                 reusable audit library
  sae.py                 sparse autoencoders (activation dictionaries)
  extract.py             activation-extraction harness for a trained checkpoint
  probe.py               linear probing of SAE features (scanner vs disease)
  patch.py               causal feature ablation / activation patching
  das.py                 Distributed Alignment Search (DAS) + interchange-intervention accuracy (IIA)
  edge_attribution.py    sparse-feature-circuit node attribution (attribution patching)
  metrics.py             SAE quality + seed-stability metrics
  abide_data.py          ABIDE (autism) adapter for the cross-disorder replication

scripts/                 pipeline, controls, experiments, and figures (~65 scripts)
  phase1..phase6*        main pipeline: modality gate -> SAE atlas -> probing ->
                         feature/circuit ablation -> path patching -> DAS -> harmonization
  ctrl_*                 controls and headline experiments, including:
    ctrl_entangled_full.py        passenger-vs-shortcut (entangled regime)
    ctrl_spd_intervention.py      connectome-native SPD (log-Euclidean) intervention operator
    ctrl_offsite_closedloop.py    unseen-site deployment gate (decodability vs causal-exposure;
                                  matched-control false-positive-rate gap + bootstrap CIs); also
                                  the in-distribution reconciliation and model-selection loops
    ctrl_highsignal_compression.py  k=1 DAS interchange on a high-signal attribute (sex) control
    ctrl_modern_harmonize.py      covariate-preserving ComBat (neuroHarmonize) unseen-site test
    ctrl_decode_vs_causal.py      decodable-but-noncausal direction families
    ctrl_natural_abide.py         ABIDE under-alarm direction
    ctrl_fd_residualize.py        head-motion (framewise-displacement) control
  make_*                 figure generation (including the brain plates)
  cross_arch.py          cross-architecture replication (MLP / BrainNetCNN / Transformer)
  harmonize_compare.py   ComBat / site-regression / INLP / LEACE comparison
  bootstrap_ci.py        bootstrap confidence intervals
  run_holdout.sbatch     SLURM driver for the held-out extraction -> SAE -> probe pipeline

tests/                   pytest unit tests (das, edge_attribution, patch, sae)
manuscript/source_data/  aggregate source data for a figure panel (no subject-level data)
```

Scripts add `src/` to `sys.path`; run them from the repository root.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.8/3.10. `neuroCombat` is required only for the ComBat baseline; `nibabel`/`nilearn` only for
the structural and brain-figure scripts.

## Data availability

This repository contains **code only**. The neuroimaging cohorts (COBRE, FBIRN, PK\_MPRC, a Chinese SZ cohort, and
ABIDE I+II) are **not** redistributed here: we release neither raw images nor subject-level demographics. ABIDE
I+II are publicly available via the ABIDE/NITRC repositories; the other cohorts are governed by their respective
data-use agreements. Scripts reference site-local data paths (e.g. NeuroMark templates and preprocessed
derivatives) that must be supplied by the user. The trained classifier and activation-extraction stack depend on
the authors' `MultiViT2` model codebase, which is maintained separately.

Released artifacts are model-internal derived representations and aggregate results that cannot be inverted to
recover any individual scan.

## Reproducing the analysis (sketch)

1. Train / obtain the frozen classifier and extract activations (`scripts/run_holdout.sbatch`, `src/mib/extract.py`).
2. Train the SAE atlas and probe features (`scripts/train_sae.py`, `src/mib/sae.py`, `src/mib/probe.py`).
3. Feature/circuit ablation (`src/mib/patch.py`, `scripts/phase3*`); DAS / IIA (`src/mib/das.py`, `scripts/phase5*`).
4. Harmonization comparison and the controls/experiments (`scripts/harmonize_compare.py`, `scripts/ctrl_*`).
5. Figures (`scripts/make_*`).

## Citation

A citation entry (DOI) will be added upon publication. Please cite the *Medical Image Analysis* paper if you use
this code.

## License

MIT — see [LICENSE](LICENSE).
