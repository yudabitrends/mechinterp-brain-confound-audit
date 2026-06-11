"""ABIDE (autism) data adapter for the MultiViT2 replication.

Builds a per-subject manifest joining precomputed sFNC (.mat tensors) + sMRI NIfTI dirs +
phenotype (diagnosis), and a Dataset yielding the same {sMRI,(1,96,112,96); sFNC,(53,53);
label; subject_id} contract as geomultivit's MultiModalH5Dataset, so the existing training /
extraction code runs unchanged. Positive class = ASD (DX_GROUP 1 -> 1; HC 2 -> 0).
"""
from __future__ import annotations

import os, sys
import numpy as np, pandas as pd, scipy.io as sio, torch
from torch.utils.data import Dataset

sys.path.insert(0, "/home/users/ybi3/MultiViT2/geometric_multivit/src")
from geomultivit.data.preprocess import volume_to_tensor

VBM = "VBM_modulated_SPM12_SM6.nii"
A1 = dict(mat="/data/qneuromark/Results/SFNC/ABIDE1/ABIDE_TRall.mat",
         smri="/data/qneuromark/Data/Autism/ABIDE1/ZN_Neuromark/ZN_Prep_sMRI",
         pheno="/data/qneuromark/Data/Autism/ABIDE1/Data_info/Phenotypic_V1_0b_preprocessed1.csv")
A2 = dict(mat="/data/qneuromark/Results/SFNC/ABIDE2/ABIDE2_TRall_new.mat",
         smri="/data/qneuromark/Data/Autism/ABIDE2/ZN_Neuromark/ZN_Prep_sMRI",
         pheno="/data/qneuromark/Data/Autism/ABIDE2/Data_info/ABIDEII_Composite_Phenotypic.csv")


def _s(x):
    """Extract a scalar string from a nested MATLAB object cell (or '' if empty)."""
    a = np.asarray(x).ravel()
    return str(a[0]) if a.size and np.asarray(a[0]).size else ""


def build_abide_manifest() -> pd.DataFrame:
    rows = []
    # --- ABIDE1: join key = FILE_ID ---
    m1 = sio.loadmat(A1["mat"])
    site1 = [_s(v) for v in m1["analysis_SCORE_str"][:, 4]]
    fid1 = [_s(v) for v in m1["analysis_SCORE_str"][:, 5]]
    ph1 = pd.read_csv(A1["pheno"]).set_index("FILE_ID")["DX_GROUP"].to_dict()
    for i, (fid, site) in enumerate(zip(fid1, site1)):
        if not fid or fid not in ph1:
            continue
        rows.append(dict(SubjectID=fid, cohort="ABIDE_I", site=f"A1_{site}",
                         Diagnosis=1 if int(ph1[fid]) == 1 else 0,
                         sMRIPath=os.path.join(A1["smri"], fid, VBM), sfnc_idx=i))
    # --- ABIDE2: join key = SUB_ID; dir = {site}_{subid} ---
    m2 = sio.loadmat(A2["mat"])
    site2 = [_s(v) for v in m2["analysis_SCORE_str"][:, 0]]
    sub2 = m2["analysis_ID"].ravel().astype(int)
    ph2 = pd.read_csv(A2["pheno"], encoding="latin-1").set_index("SUB_ID")["DX_GROUP"].to_dict()
    for i, (sid, site) in enumerate(zip(sub2, site2)):
        if int(sid) not in ph2:
            continue
        d = f"{site}_{sid}"
        rows.append(dict(SubjectID=d, cohort="ABIDE_II", site=f"A2_{site}",
                         Diagnosis=1 if int(ph2[int(sid)]) == 1 else 0,
                         sMRIPath=os.path.join(A2["smri"], d, VBM), sfnc_idx=i))
    df = pd.DataFrame(rows)
    df = df[df["sMRIPath"].apply(os.path.exists)].reset_index(drop=True)
    return df


class ABIDEDataset(Dataset):
    """Yields {sMRI,(1,D,H,W); sFNC,(53,53); label; subject_id} from the ABIDE manifest.
    sFNC tensors are loaded once per cohort and indexed by sfnc_idx."""

    def __init__(self, df, volume_shape=(96, 112, 96), cache_dir=None, augment=False):
        self.df = df.reset_index(drop=True)
        self.volume_shape = tuple(volume_shape)
        self.augment = augment
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._fnc = {"ABIDE_I": sio.loadmat(A1["mat"])["sFNC"],
                     "ABIDE_II": sio.loadmat(A2["mat"])["sFNC"]}

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        return self.df["Diagnosis"].to_numpy(np.int64)

    def _vol(self, path, sid):
        if self.cache_dir:
            cp = os.path.join(self.cache_dir, f"{sid}.npy")
            if os.path.exists(cp):
                return np.load(cp)
        import nibabel as nib
        vol = volume_to_tensor(np.asarray(nib.load(path).get_fdata(), np.float32), self.volume_shape)
        if self.cache_dir:
            np.save(os.path.join(self.cache_dir, f"{sid}.npy"), vol)
        return vol

    def __getitem__(self, i):
        r = self.df.iloc[i]
        vol = self._vol(str(r["sMRIPath"]), str(r["SubjectID"]))
        if self.augment and np.random.rand() < 0.5:
            vol = np.ascontiguousarray(vol[::-1, :, :])
        fnc = np.asarray(self._fnc[r["cohort"]][int(r["sfnc_idx"])], np.float32)
        return {"sMRI": torch.from_numpy(vol).unsqueeze(0).float(),
                "sFNC": torch.from_numpy(np.ascontiguousarray(fnc)).float(),
                "label": torch.tensor(int(r["Diagnosis"]), dtype=torch.long),
                "subject_id": str(r["SubjectID"])}
