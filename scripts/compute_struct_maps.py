#!/usr/bin/env python
"""Voxel-wise structural (VBM modulated GM) discriminability maps for the structural brain panel:
per-voxel scanner (US vs China) and disease (HC vs SZ) AUC, on the held-out-aware full cohort.
Saves two NIfTIs (MNI 121x145x121) for nilearn rendering. Run as a SLURM CPU job (loads ~1253 vols)."""
import os, sys
import numpy as np, nibabel as nib
from scipy.stats import rankdata
sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from geomultivit.data.multicohort import load_geometric_cohort

H5 = "/home/users/ybi3/data/szdataset_modified.h5"
OUT = "/data/users1/ybi/mechinterp_brain/struct_maps"


def voxel_auc(X, y):
    """Per-column AUC (Mann-Whitney) of X (N,M) predicting binary y, chunked over columns."""
    y = np.asarray(y); npos = int(y.sum()); nneg = len(y) - npos
    auc = np.empty(X.shape[1], np.float32)
    for s in range(0, X.shape[1], 40000):
        c = X[:, s:s + 40000]
        r = np.apply_along_axis(rankdata, 0, c)             # ranks within each column
        auc[s:s + c.shape[1]] = (r[y == 1].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)
    return auc


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_geometric_cohort(h5_path=H5, splits=("train", "test"), accessible_only=False).df
    df = df[df.cohort.isin(["COBRE", "FBIRN", "ChineseSZ", "PK_MPRC"])].drop_duplicates("SubjectID")
    df = df[df["sMRIPath"].apply(lambda p: os.path.exists(str(p)))].reset_index(drop=True)
    df = df[df.population.isin(["US", "China"])].reset_index(drop=True)
    print(f"N={len(df)} subjects", flush=True)
    ref = nib.load(str(df.iloc[0]["sMRIPath"])); shape = ref.shape; affine = ref.affine

    # group GM mask: present (>0.05) in >=90% of subjects
    acc = np.zeros(shape, np.float32); n = 0
    for p in df["sMRIPath"]:
        acc += (nib.load(str(p)).get_fdata() > 0.05).astype(np.float32); n += 1
        if n % 200 == 0:
            print(f"  mask pass {n}", flush=True)
    mask = acc >= 0.9 * n
    midx = np.where(mask.ravel())[0]
    print(f"mask voxels = {len(midx)} ({100*len(midx)/mask.size:.1f}% of volume)", flush=True)

    # load masked voxels into (N, M)
    X = np.empty((len(df), len(midx)), np.float32)
    for i, p in enumerate(df["sMRIPath"]):
        X[i] = nib.load(str(p)).get_fdata().ravel()[midx]
        if i % 200 == 0:
            print(f"  load {i}", flush=True)
    yscan = (df.population.to_numpy() == "China").astype(int)
    ydx = df.Diagnosis.to_numpy().astype(int)
    print("computing voxel AUC (scanner)...", flush=True); a_scan = voxel_auc(X, yscan)
    print("computing voxel AUC (disease)...", flush=True); a_dx = voxel_auc(X, ydx)

    for name, a in [("scanner", a_scan), ("disease", a_dx)]:
        vol = np.full(mask.size, 0.5, np.float32); vol[midx] = a; vol = vol.reshape(shape)
        # center on 0.5 -> signed importance for display
        nib.save(nib.Nifti1Image((vol - 0.5).astype(np.float32), affine, ref.header),
                 f"{OUT}/struct_{name}_auc.nii.gz")
        print(f"{name}: |AUC-0.5| max={np.abs(a-0.5).max():.3f} mean={np.abs(a-0.5).mean():.3f} "
              f"frac>0.1={np.mean(np.abs(a-0.5)>0.1):.3f}", flush=True)
    np.savez(f"{OUT}/struct_auc_vals.npz", scanner=a_scan, disease=a_dx)
    print(f"wrote {OUT}/struct_{{scanner,disease}}_auc.nii.gz + vals", flush=True)


if __name__ == "__main__":
    main()
