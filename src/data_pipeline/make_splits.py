"""
make_splits.py — Site-level train/val/test split for mangrove patches.

Split strategy (prevents data leakage):
  Train pool: sundarbans_2024 + gulf_of_kutch_2024
  Val:        10% randomly held out from train pool (seed=42)
  Test:       pichavaram_2024 (completely unseen site)
  2020 data:  kept for carbon flux calculation, excluded from splits

Outputs (written to data/splits/):
  train.txt       — one image patch path per line (relative to repo root)
  val.txt         — same format
  test.txt        — same format
  norm_stats.json — per-band mean and std computed on train set only
"""

import argparse
import glob
import json
import os
import random

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PATCHES_DIR = os.path.join(REPO_ROOT, "data", "patches")
SPLITS_DIR = os.path.join(REPO_ROOT, "data", "splits")

TRAIN_SITES = ["sundarbans_2024", "gulf_of_kutch_2024"]
TEST_SITES = ["pichavaram_2024"]
NUM_BANDS = 6


def collect_patch_paths(patches_dir: str, site_dirs: list[str]) -> list[str]:
    """Return sorted list of image patch paths (relative to repo root) for
    the given site directories."""
    paths = []
    for site_dir in site_dirs:
        pattern = os.path.join(patches_dir, site_dir, "img_*.npy")
        for full_path in sorted(glob.glob(pattern)):
            # Convert to path relative to repo root, use forward slashes
            rel = os.path.relpath(full_path, REPO_ROOT).replace("\\", "/")
            paths.append(rel)
    return paths


def compute_norm_stats(patch_paths: list[str]) -> dict:
    """Two-pass per-band mean and std. Leverages numpy vectorisation per patch
    to stay fast while keeping memory constant (one patch at a time)."""
    total_pixels = 0
    band_sum = np.zeros(NUM_BANDS, dtype=np.float64)

    # --- Pass 1: mean ---
    for rel_path in patch_paths:
        abs_path = os.path.join(REPO_ROOT, rel_path)
        img = np.load(abs_path).astype(np.float64)  # (6, 256, 256)
        npix = img.shape[1] * img.shape[2]
        total_pixels += npix
        band_sum += img.reshape(NUM_BANDS, -1).sum(axis=1)

    if total_pixels == 0:
        return {"mean": [0.0] * NUM_BANDS, "std": [0.0] * NUM_BANDS}

    band_mean = band_sum / total_pixels

    # --- Pass 2: variance ---
    band_sq_diff = np.zeros(NUM_BANDS, dtype=np.float64)
    for rel_path in patch_paths:
        abs_path = os.path.join(REPO_ROOT, rel_path)
        img = np.load(abs_path).astype(np.float64)
        pixels = img.reshape(NUM_BANDS, -1)  # (6, N)
        band_sq_diff += ((pixels - band_mean[:, None]) ** 2).sum(axis=1)

    band_std = np.sqrt(band_sq_diff / total_pixels)

    return {
        "mean": band_mean.tolist(),
        "std": band_std.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create site-level train/val/test splits and compute "
        "normalization statistics."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of training patches to hold out as validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Collect patch paths by role
    # ------------------------------------------------------------------
    train_pool = collect_patch_paths(PATCHES_DIR, TRAIN_SITES)
    test_paths = collect_patch_paths(PATCHES_DIR, TEST_SITES)

    if not train_pool:
        print(
            f"[WARNING] No training patches found in {PATCHES_DIR} for "
            f"sites {TRAIN_SITES}. Check that patches exist."
        )
    if not test_paths:
        print(
            f"[WARNING] No test patches found in {PATCHES_DIR} for "
            f"sites {TEST_SITES}. Check that patches exist."
        )

    # ------------------------------------------------------------------
    # 2. Split train pool into train + val
    # ------------------------------------------------------------------
    random.seed(args.seed)
    shuffled = list(train_pool)
    random.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * args.val_ratio)) if shuffled else 0
    val_paths = sorted(shuffled[:val_count])
    train_paths = sorted(shuffled[val_count:])

    # ------------------------------------------------------------------
    # 3. Save manifests
    # ------------------------------------------------------------------
    os.makedirs(SPLITS_DIR, exist_ok=True)

    for name, paths in [
        ("train", train_paths),
        ("val", val_paths),
        ("test", test_paths),
    ]:
        out_file = os.path.join(SPLITS_DIR, f"{name}.txt")
        with open(out_file, "w") as f:
            for p in paths:
                f.write(p + "\n")
        print(f"  {name:5s}: {len(paths):>5d} patches  ->  {out_file}")

    # ------------------------------------------------------------------
    # 4. Compute normalization stats on TRAIN set only
    # ------------------------------------------------------------------
    if train_paths:
        print("\nComputing per-band normalization stats on training set ...")
        stats = compute_norm_stats(train_paths)
    else:
        stats = {"mean": [0.0] * NUM_BANDS, "std": [0.0] * NUM_BANDS}

    stats_file = os.path.join(SPLITS_DIR, "norm_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved normalization stats -> {stats_file}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    band_names = ["B2", "B3", "B4", "B8", "B11", "B12"]
    print("\n--- Split Summary ---")
    print(f"  Train : {len(train_paths)} patches  ({', '.join(TRAIN_SITES)})")
    print(f"  Val   : {len(val_paths)} patches  (10% held out from train pool)")
    print(f"  Test  : {len(test_paths)} patches  ({', '.join(TEST_SITES)})")
    print(f"\n--- Normalization Stats (train only) ---")
    for i, name in enumerate(band_names):
        print(f"  {name:4s}  mean={stats['mean'][i]:.6f}  std={stats['std'][i]:.6f}")

    # Report 2020 data presence
    all_dirs = sorted(
        d
        for d in os.listdir(PATCHES_DIR)
        if os.path.isdir(os.path.join(PATCHES_DIR, d))
    ) if os.path.isdir(PATCHES_DIR) else []

    dirs_2020 = [d for d in all_dirs if d.endswith("_2020")]
    if dirs_2020:
        print(f"\n  2020 directories present (for carbon flux, not in splits): "
              f"{dirs_2020}")


if __name__ == "__main__":
    main()
