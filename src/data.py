"""
src/data.py — dataset classes and dataloader helpers.

Supports two dataset modes:
  - "plantvillage"  : folder-per-class structure (single-label)
  - "plantpathology": CSV with space-separated multi-label annotations

Public API
----------
make_pv_loaders(root, img_size, batch_size, val_frac, test_frac, num_workers)
make_pp_loaders(root, img_size, batch_size, val_frac, test_frac, num_workers)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _build_transforms(img_size: int, augment: bool) -> transforms.Compose:
    """Return a torchvision transform pipeline.

    Training pipeline applies random flips and colour jitter.
    Validation/test pipeline applies only resize + centre-crop + normalise.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class PlantVillageDataset(Dataset):
    """Folder-per-class image dataset for PlantVillage.

    Parameters
    ----------
    samples   : list of (path, class_idx) tuples
    classes   : ordered list of class name strings
    transform : torchvision transform or None
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        classes: list[str],
        transform=None,
    ) -> None:
        self.samples   = samples
        self.classes   = classes
        self.transform = transform

    # class_to_idx convenience property
    @property
    def class_to_idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_pv_loaders(
    root: str | Path,
    img_size: int = 224,
    batch_size: int = 64,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train / val / test DataLoaders for PlantVillage (color subset).

    Returns
    -------
    train_loader, val_loader, test_loader, classes
    """
    color_dir = Path(root) / "data" / "plantvillage" / "plantvillage dataset" / "color"
    classes   = sorted([d.name for d in color_dir.iterdir() if d.is_dir()])

    all_samples: list[tuple[Path, int]] = []
    for idx, cls in enumerate(classes):
        for p in (color_dir / cls).glob("*.*"):
            all_samples.append((p, idx))

    # Stratified split: train / (val + test)
    labels = [s[1] for s in all_samples]
    train_samples, tmp_samples, _, tmp_labels = train_test_split(
        all_samples, labels,
        test_size=val_frac + test_frac,
        stratify=labels,
        random_state=seed,
    )
    # Split the held-out portion into val and test
    rel_test = test_frac / (val_frac + test_frac)
    val_samples, test_samples = train_test_split(
        tmp_samples,
        test_size=rel_test,
        stratify=tmp_labels,
        random_state=seed,
    )

    train_ds = PlantVillageDataset(train_samples, classes,
                                   transform=_build_transforms(img_size, augment=True))
    val_ds   = PlantVillageDataset(val_samples,   classes,
                                   transform=_build_transforms(img_size, augment=False))
    test_ds  = PlantVillageDataset(test_samples,  classes,
                                   transform=_build_transforms(img_size, augment=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, classes


class PlantPathologyDataset(Dataset):
    """Multi-label image dataset for Plant Pathology 2021.

    Parameters
    ----------
    df        : DataFrame with columns ["image", "labels"]
    img_dir   : directory containing the .jpg files
    classes   : ordered list of label strings
    transform : torchvision transform or None
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        classes: list[str],
        transform=None,
    ) -> None:
        self.df        = df.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.classes   = classes
        self.n_classes = len(classes)
        self._cls_idx  = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["image"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Build binary multi-hot target vector
        target = torch.zeros(self.n_classes, dtype=torch.float32)
        for lbl in str(row["labels"]).split():
            if lbl in self._cls_idx:
                target[self._cls_idx[lbl]] = 1.0
        return img, target


def make_pp_loaders(
    root: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train / val / test DataLoaders for Plant Pathology 2021.

    Returns
    -------
    train_loader, val_loader, test_loader, classes
    """
    root    = Path(root)
    img_dir = root / "data" / "plantpathology" / "train_images"
    csv     = root / "data" / "plantpathology" / "train.csv"

    df = pd.read_csv(csv)

    # Drop rows whose image file is missing
    present = {p.name for p in img_dir.glob("*.*")}
    df = df[df["image"].astype(str).isin(present)].reset_index(drop=True)

    # Derive ordered class list from all label tokens
    from collections import Counter
    all_tokens: list[str] = []
    for lbl_str in df["labels"]:
        all_tokens.extend(str(lbl_str).split())
    classes = [c for c, _ in Counter(all_tokens).most_common()]

    # Stratified split on the first (dominant) label per row
    primary_label = df["labels"].astype(str).str.split().str[0]
    train_df, tmp_df = train_test_split(
        df, test_size=val_frac + test_frac,
        stratify=primary_label, random_state=seed,
    )
    rel_test = test_frac / (val_frac + test_frac)
    tmp_primary = tmp_df["labels"].astype(str).str.split().str[0]
    val_df, test_df = train_test_split(
        tmp_df, test_size=rel_test,
        stratify=tmp_primary, random_state=seed,
    )

    train_ds = PlantPathologyDataset(train_df, img_dir, classes,
                                     transform=_build_transforms(img_size, augment=True))
    val_ds   = PlantPathologyDataset(val_df,   img_dir, classes,
                                     transform=_build_transforms(img_size, augment=False))
    test_ds  = PlantPathologyDataset(test_df,  img_dir, classes,
                                     transform=_build_transforms(img_size, augment=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, classes

