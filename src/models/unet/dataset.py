"""
PyTorch Dataset and DataLoader utilities for mangrove segmentation patches.

Expects patches as .npy files (img: 6×256×256 float32, mask: 256×256 uint8)
and a JSON norm_stats file with per-band mean/std computed on training data.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class MangroveDataset(Dataset):
    """
    Dataset for mangrove segmentation patches.

    Each sample is a pair (image, mask) loaded from .npy files.
    Image: 6-band Sentinel-2 composite, shape (6, 256, 256).
    Mask:  Binary mangrove label, shape (1, 256, 256).

    Parameters
    ----------
    split_file : str
        Path to a text file listing relative patch paths (one per line),
        e.g. ``data/patches/sundarbans_2024/img_0001.npy``.
    norm_stats_file : str
        Path to a JSON file with keys ``"mean"`` and ``"std"``, each a
        list of 6 floats (per-band statistics computed on training data).
    augment : bool
        If True, apply random horizontal and vertical flips.
    """

    def __init__(self, split_file: str, norm_stats_file: str, augment: bool = False):
        # --- load split list ---
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) == 0:
            raise ValueError(f"Split file is empty: {split_file}")

        # Resolve image paths and derive corresponding mask paths
        self.img_paths = []
        self.mask_paths = []
        for rel_path in lines:
            img_abs = os.path.join(REPO_ROOT, rel_path)
            # mask file has the same name but with "mask_" instead of "img_"
            mask_abs = os.path.join(
                os.path.dirname(img_abs),
                os.path.basename(img_abs).replace("img_", "mask_"),
            )
            self.img_paths.append(img_abs)
            self.mask_paths.append(mask_abs)

        # --- load normalisation statistics ---
        if not os.path.isfile(norm_stats_file):
            raise FileNotFoundError(f"Norm stats file not found: {norm_stats_file}")

        with open(norm_stats_file, "r") as f:
            stats = json.load(f)

        # Shape (6, 1, 1) for broadcasting over (6, H, W)
        self.mean = np.array(stats["mean"], dtype=np.float32).reshape(6, 1, 1)
        self.std = np.array(stats["std"], dtype=np.float32).reshape(6, 1, 1)

        self.augment = augment

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # --- load arrays ---
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image patch not found: {img_path}")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask patch not found: {mask_path}")

        img = np.load(img_path).astype(np.float32)   # (6, 256, 256)
        mask = np.load(mask_path).astype(np.uint8)    # (256, 256)

        # --- normalise ---
        img = (img - self.mean) / (self.std + 1e-8)

        # --- augmentation ---
        if self.augment:
            if np.random.random() > 0.5:
                img = img[:, :, ::-1]     # horizontal flip
                mask = mask[:, ::-1]
            if np.random.random() > 0.5:
                img = img[:, ::-1, :]     # vertical flip
                mask = mask[::-1, :]
            # ensure memory layout is contiguous after flipping
            img = np.ascontiguousarray(img)
            mask = np.ascontiguousarray(mask)

        # --- convert to tensors ---
        img_tensor = torch.from_numpy(img.copy()).float()             # (6, 256, 256)
        mask_tensor = torch.from_numpy(mask.copy()).float().unsqueeze(0)  # (1, 256, 256)

        return img_tensor, mask_tensor


def compute_pos_weight(split_file: str) -> float:
    """
    Compute class-balancing weight for BCEWithLogitsLoss.

    Iterates every mask referenced in *split_file*, counts positive
    (mangrove) vs. negative (background) pixels, and returns
    ``neg_count / pos_count``.

    Parameters
    ----------
    split_file : str
        Path to a split text file (same format as MangroveDataset).

    Returns
    -------
    float
        The positive-class weight (neg / pos ratio).
    """
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) == 0:
        raise ValueError(f"Split file is empty: {split_file}")

    pos_count = 0
    neg_count = 0

    for rel_path in lines:
        img_abs = os.path.join(REPO_ROOT, rel_path)
        mask_abs = os.path.join(
            os.path.dirname(img_abs),
            os.path.basename(img_abs).replace("img_", "mask_"),
        )
        if not os.path.isfile(mask_abs):
            raise FileNotFoundError(f"Mask not found: {mask_abs}")

        mask = np.load(mask_abs).astype(np.uint8)
        pos_count += int(np.sum(mask > 0))
        neg_count += int(np.sum(mask == 0))

    if pos_count == 0:
        raise ValueError("No positive pixels found — pos_weight is undefined.")

    ratio = neg_count / pos_count
    print(f"[pos_weight] pos={pos_count:,}  neg={neg_count:,}  ratio={ratio:.2f}")
    return ratio


def get_dataloaders(batch_size: int = 8, num_workers: int = 0) -> tuple:
    """
    Build train / val / test DataLoaders with standard settings.

    Parameters
    ----------
    batch_size : int
        Mini-batch size (default 8; reduce to 4 if OOM on 4 GB VRAM).
    num_workers : int
        DataLoader worker processes (default 0 for Windows compatibility).

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``
    """
    splits_dir = os.path.join(REPO_ROOT, "data", "splits")
    train_txt = os.path.join(splits_dir, "train.txt")
    val_txt = os.path.join(splits_dir, "val.txt")
    test_txt = os.path.join(splits_dir, "test.txt")
    norm_stats = os.path.join(splits_dir, "norm_stats.json")

    # Verify all required files exist before constructing datasets
    for path in [train_txt, val_txt, test_txt, norm_stats]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required file missing: {path}\n"
                "Run the data pipeline (extract_patches → make_splits) first."
            )

    train_ds = MangroveDataset(train_txt, norm_stats, augment=True)
    val_ds = MangroveDataset(val_txt, norm_stats, augment=False)
    test_ds = MangroveDataset(test_txt, norm_stats, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
