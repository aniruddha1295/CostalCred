"""
Extract 256x256 patches from aligned Sentinel-2 / GMW mask pairs.

Slides a window with 50% overlap (stride 128) across each image/mask pair.
Keeps all patches containing mangrove pixels and ~10% of all-zero-mask
patches as negative samples (deterministic via position-based hashing).

Outputs:
  data/patches/{site}_{year}/img_NNNN.npy   (6, 256, 256) float32 [0, 1]
  data/patches/{site}_{year}/mask_NNNN.npy  (256, 256)    uint8   {0, 1}

Usage:
  python src/data_pipeline/extract_patches.py --site sundarbans --year 2024
  python src/data_pipeline/extract_patches.py --all
"""

import argparse
import hashlib
import os

import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SITES = ["sundarbans", "gulf_of_kutch", "pichavaram"]
YEARS = [2020, 2024]

IMG_DIR = os.path.join("data", "raw", "sentinel2")
MASK_DIR = os.path.join("data", "raw", "gmw")
PATCH_DIR = os.path.join("data", "patches")

SCALE_FACTOR = 10000.0  # Sentinel-2 uint16 values are in [0, 10000]


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------
def extract_patches(site, year, patch_size=256, stride=128, neg_keep_ratio=0.1):
    """Extract patches from one (site, year) image/mask pair."""

    img_path = os.path.join(IMG_DIR, f"{site}_{year}.tif")
    mask_path = os.path.join(MASK_DIR, f"{site}_{year}_mask.tif")

    if not os.path.exists(img_path):
        print(f"  WARNING: {img_path} not found, skipping.")
        return
    if not os.path.exists(mask_path):
        print(f"  WARNING: {mask_path} not found, skipping.")
        return

    # Read full rasters
    with rasterio.open(img_path) as src:
        image = src.read()  # (bands, H, W) uint16
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # (H, W) uint8

    bands, height, width = image.shape
    print(f"\n{site} {year}: image {bands}x{height}x{width}, "
          f"mask {mask.shape[0]}x{mask.shape[1]}")

    # Output directory
    out_dir = os.path.join(PATCH_DIR, f"{site}_{year}")
    os.makedirs(out_dir, exist_ok=True)

    # Determine how many negatives to keep: hash threshold from ratio
    # hash((row, col)) % N == 0  where N = round(1 / neg_keep_ratio)
    neg_mod = max(1, round(1.0 / neg_keep_ratio))

    patch_idx = 0
    n_positive = 0
    n_negative_kept = 0
    n_negative_skipped = 0

    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            mask_patch = mask[row:row + patch_size, col:col + patch_size]

            has_mangrove = mask_patch.any()

            if not has_mangrove:
                # Deterministic selection: keep ~1/neg_mod of negatives
                # Use md5 instead of hash() because Python randomises
                # hash seeds across sessions (PYTHONHASHSEED).
                h = int(hashlib.md5(f"{row},{col}".encode()).hexdigest(), 16)
                if h % neg_mod != 0:
                    n_negative_skipped += 1
                    continue
                n_negative_kept += 1
            else:
                n_positive += 1

            # Extract image patch and normalise to [0, 1] float32
            img_patch = image[:, row:row + patch_size, col:col + patch_size]
            img_patch = img_patch.astype(np.float32) / SCALE_FACTOR

            # Save
            np.save(os.path.join(out_dir, f"img_{patch_idx:04d}.npy"), img_patch)
            np.save(os.path.join(out_dir, f"mask_{patch_idx:04d}.npy"),
                    mask_patch.astype(np.uint8))
            patch_idx += 1

    total = patch_idx
    print(f"  Saved {total} patches to {out_dir}")
    print(f"    Positive (has mangrove): {n_positive}")
    print(f"    Negative kept:          {n_negative_kept}")
    print(f"    Negative skipped:       {n_negative_skipped}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract 256x256 patches from Sentinel-2 / GMW mask pairs"
    )
    parser.add_argument("--site", choices=SITES, help="Single site to process")
    parser.add_argument("--year", type=int, choices=YEARS, help="Single year")
    parser.add_argument("--all", action="store_true",
                        help="Process all site/year combinations")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Patch size in pixels (default: 256)")
    parser.add_argument("--stride", type=int, default=128,
                        help="Stride in pixels (default: 128)")
    parser.add_argument("--neg-keep-ratio", type=float, default=0.1,
                        help="Fraction of all-zero-mask patches to keep (default: 0.1)")
    args = parser.parse_args()

    kwargs = dict(
        patch_size=args.patch_size,
        stride=args.stride,
        neg_keep_ratio=args.neg_keep_ratio,
    )

    if args.all:
        for site in SITES:
            for year in YEARS:
                extract_patches(site, year, **kwargs)
    elif args.site and args.year:
        extract_patches(args.site, args.year, **kwargs)
    else:
        parser.error("Provide --site and --year, or --all")


if __name__ == "__main__":
    main()
