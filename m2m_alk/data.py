from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from monai.transforms import (
    CastToType,
    Compose,
    NormalizeIntensity,
    RandAffine,
    RandFlip,
    RandGaussianNoise,
    RandRotate90,
    ScaleIntensityRange,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .utils import read_table, torch_load_tensor


def build_ct_transform(train: bool = False) -> Compose:
    """Build MONAI transforms matching Supplementary Table S2."""
    transforms = [
        ScaleIntensityRange(a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ]
    if train:
        transforms.extend(
            [
                RandAffine(
                    prob=0.5,
                    translate_range=(3, 3, 3),
                    rotate_range=(0.2, 0.2, 0.2),
                    scale_range=(0.1, 0.1, 0.1),
                ),
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                RandFlip(prob=0.5, spatial_axis=0),
                RandGaussianNoise(prob=0.2),
            ]
        )
    transforms.append(CastToType(dtype=np.float32))
    return Compose(transforms)


def build_ct_display_transform() -> Compose:
    """Intensity scaling for Grad-CAM overlays without z-score normalization."""
    return Compose(
        [
            ScaleIntensityRange(a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
            CastToType(dtype=np.float32),
        ]
    )


class NiftiROIDataset(Dataset):
    """Dataset for 32 x 32 x 32 CT ROI NIfTI volumes.

    The metadata table is expected to contain sample identifiers in the first column
    and labels in the second column by default. Files are read from
    ``roi_root / split_name / f"{sample_id}.nii.gz"``.
    """

    def __init__(
        self,
        metadata_path: str | Path,
        roi_root: str | Path,
        split_name: str,
        transform=None,
        sample_col: int | str = 0,
        label_col: int | str = 1,
        return_display: bool = False,
        display_transform=None,
    ) -> None:
        df = read_table(metadata_path)
        self.names = df.iloc[:, sample_col].astype(str).tolist() if isinstance(sample_col, int) else df[sample_col].astype(str).tolist()
        self.labels = df.iloc[:, label_col].astype(int).tolist() if isinstance(label_col, int) else df[label_col].astype(int).tolist()
        self.data_dir = Path(roi_root) / split_name
        self.transform = transform
        self.return_display = return_display
        self.display_transform = display_transform

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        image_path = self.data_dir / f"{name}.nii.gz"
        if not image_path.exists():
            raise FileNotFoundError(f"CT ROI not found: {image_path}")
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))[np.newaxis, ...]
        model_img = self.transform(img) if self.transform else img.astype(np.float32)
        label = torch.tensor(self.labels[idx]).long()
        if self.return_display:
            display_img = self.display_transform(img) if self.display_transform else img.astype(np.float32)
            return model_img, display_img, label, name
        return model_img, label, name


class MultiModalFeatureDataset(Dataset):
    """Patient-level WSI feature + CT feature dataset.

    Default column order:
    ``sample_id, wsi_feature_path, label, ct_feat_0, ct_feat_1, ...``.
    """

    def __init__(
        self,
        df,
        sample_col: int | str = 0,
        wsi_path_col: int | str = 1,
        label_col: int | str = 2,
        first_ct_feature_col: int = 3,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.sample_col = sample_col
        self.wsi_path_col = wsi_path_col
        self.label_col = label_col
        self.first_ct_feature_col = first_ct_feature_col

        self.names = self._col(sample_col).astype(str).values
        self.wsi_paths = self._col(wsi_path_col).astype(str).values
        self.labels = self._col(label_col).astype(int).values
        self.ct_features = self.df.iloc[:, first_ct_feature_col:].values.astype(np.float32)

    def _col(self, col: int | str):
        return self.df.iloc[:, col] if isinstance(col, int) else self.df[col]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        wsi_feat = torch_load_tensor(self.wsi_paths[idx], map_location="cpu")
        ct_feat = torch.from_numpy(self.ct_features[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return wsi_feat, ct_feat, label, self.names[idx]


def fit_transform_ct_features(
    train_df,
    val_df=None,
    external_df=None,
    first_ct_feature_col: int = 3,
) -> Tuple[StandardScaler, tuple]:
    """Fit CT feature standardization on the training set only."""
    scaler = StandardScaler()
    output = []
    train_df = train_df.copy()
    train_df.iloc[:, first_ct_feature_col:] = scaler.fit_transform(
        train_df.iloc[:, first_ct_feature_col:]
    )
    output.append(train_df)
    for df in (val_df, external_df):
        if df is None:
            output.append(None)
            continue
        df = df.copy()
        df.iloc[:, first_ct_feature_col:] = scaler.transform(df.iloc[:, first_ct_feature_col:])
        output.append(df)
    return scaler, tuple(output)
