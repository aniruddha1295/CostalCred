"""
Compute carbon flux using U-Net inference on full Sentinel-2 composites.

Runs the trained U-Net on all 6 site-year combinations (3 sites x 2 years),
then computes IPCC Tier 1 carbon stock and flux for each site.

CRITICAL: Inference is done on the FULL GeoTIFF composites using
non-overlapping 256x256 tiles (stride=256). This avoids the double-counting
problem that would occur if we used the training patches (stride=128, 50%
overlap).

Usage:
    python src/carbon/compute_carbon.py
"""

import os
import sys
import json
import math

import numpy as np
import torch
from torch.cuda.amp import autocast
import rasterio
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.models.unet.model import build_unet
from src.carbon.ipcc_tier1 import hectares_from_mask, carbon_stock, carbon_flux, full_report

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SITES = ["sundarbans", "gulf_of_kutch", "pichavaram"]
YEARS = [2020, 2024]
PATCH_SIZE = 256
STRIDE = 256  # Non-overlapping to avoid double-counting
BATCH_SIZE = 8
SCALE_FACTOR = 10000.0  # Sentinel-2 uint16 → [0, 1]

CHECKPOINT_PATH = os.path.join(REPO_ROOT, "models", "unet_best.pt")
NORM_STATS_PATH = os.path.join(REPO_ROOT, "data", "splits", "norm_stats.json")
SENTINEL2_DIR = os.path.join(REPO_ROOT, "data", "raw", "sentinel2")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: torch.device):
    """Load trained U-Net from checkpoint.

    Returns
    -------
    torch.nn.Module
        Model in eval mode on the specified device.
    """
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    model = build_unet(
        encoder_name=config.get("encoder_name", "resnet18"),
        encoder_weights=None,
        in_channels=config.get("in_channels", 6),
        classes=config.get("classes", 1),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_norm_stats():
    """Load per-band normalization statistics.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mean, std) each of shape (6, 1, 1).
    """
    if not os.path.isfile(NORM_STATS_PATH):
        raise FileNotFoundError(f"Norm stats not found: {NORM_STATS_PATH}")

    with open(NORM_STATS_PATH, "r") as f:
        stats = json.load(f)

    mean = np.array(stats["mean"], dtype=np.float32).reshape(6, 1, 1)
    std = np.array(stats["std"], dtype=np.float32).reshape(6, 1, 1)
    return mean, std


# ---------------------------------------------------------------------------
# Full-image inference
# ---------------------------------------------------------------------------

def infer_full_image(model, image: np.ndarray, mean: np.ndarray,
                     std: np.ndarray, device: torch.device) -> np.ndarray:
    """Run U-Net inference on a full Sentinel-2 composite.

    The image is tiled into non-overlapping 256x256 patches, padded at the
    edges if necessary. Predictions are stitched back into a full-size mask.

    Parameters
    ----------
    model : torch.nn.Module
        Trained U-Net in eval mode.
    image : np.ndarray
        Raw Sentinel-2 composite, shape (6, H, W), uint16.
    mean, std : np.ndarray
        Per-band normalization stats, shape (6, 1, 1).
    device : torch.device
        Compute device.

    Returns
    -------
    np.ndarray
        Binary mask (H, W), uint8, where 1 = mangrove.
    """
    bands, orig_h, orig_w = image.shape

    # Pad to make dimensions divisible by PATCH_SIZE
    pad_h = (PATCH_SIZE - orig_h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - orig_w % PATCH_SIZE) % PATCH_SIZE
    padded_h = orig_h + pad_h
    padded_w = orig_w + pad_w

    # Scale to [0, 1] and normalize
    img_float = image.astype(np.float32) / SCALE_FACTOR
    img_norm = (img_float - mean) / (std + 1e-8)

    # Pad with zeros
    if pad_h > 0 or pad_w > 0:
        img_padded = np.zeros((bands, padded_h, padded_w), dtype=np.float32)
        img_padded[:, :orig_h, :orig_w] = img_norm
    else:
        img_padded = img_norm

    # Extract non-overlapping tiles
    n_rows = padded_h // PATCH_SIZE
    n_cols = padded_w // PATCH_SIZE
    total_patches = n_rows * n_cols

    # Collect tile coordinates
    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            y = r * PATCH_SIZE
            x = c * PATCH_SIZE
            patch = img_padded[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            tiles.append((r, c, patch))

    # Run inference in batches
    pred_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)

    n_batches = math.ceil(total_patches / BATCH_SIZE)
    for batch_idx in tqdm(range(n_batches), desc="    Inference", leave=False):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, total_patches)
        batch_tiles = tiles[start:end]

        batch_tensor = torch.stack(
            [torch.from_numpy(t[2]) for t in batch_tiles]
        ).float().to(device)

        with torch.no_grad():
            with autocast():
                logits = model(batch_tensor)

        preds = (torch.sigmoid(logits).squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)

        for i, (r, c, _) in enumerate(batch_tiles):
            y = r * PATCH_SIZE
            x = c * PATCH_SIZE
            pred_mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = preds[i]

    # Remove padding
    pred_mask = pred_mask[:orig_h, :orig_w]
    return pred_mask


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and normalization stats
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    model = load_model(device)
    print("Model loaded successfully.")

    print(f"Loading norm stats: {NORM_STATS_PATH}")
    mean, std = load_norm_stats()
    print("Norm stats loaded.")

    # Run inference on all site-year combinations
    results = {}
    site_masks = {}  # {site: {year: mask}}

    print("\n" + "=" * 60)
    print("  Running U-Net inference on full composites")
    print("=" * 60)

    for site in SITES:
        site_masks[site] = {}
        for year in YEARS:
            tif_path = os.path.join(SENTINEL2_DIR, f"{site}_{year}.tif")
            print(f"\n  {site} {year}: {tif_path}")

            if not os.path.isfile(tif_path):
                print(f"    WARNING: File not found, skipping.")
                continue

            try:
                with rasterio.open(tif_path) as src:
                    image = src.read()  # (bands, H, W)
            except Exception as e:
                print(f"    ERROR reading file: {e}")
                continue

            bands, h, w = image.shape
            print(f"    Shape: {bands} bands x {h} x {w}")

            mask = infer_full_image(model, image, mean, std, device)

            mangrove_pixels = int(np.sum(mask == 1))
            total_pixels = h * w
            ha = hectares_from_mask(mask, pixel_size_m=10.0)
            print(f"    Mangrove pixels: {mangrove_pixels:,} / {total_pixels:,} "
                  f"({100 * mangrove_pixels / total_pixels:.2f}%)")
            print(f"    Mangrove area:   {ha:.2f} ha")

            site_masks[site][year] = mask

    # Compute carbon for each site
    print("\n" + "=" * 60)
    print("  IPCC Tier 1 Carbon Calculations")
    print("=" * 60)

    for site in SITES:
        masks = site_masks.get(site, {})
        if 2020 not in masks or 2024 not in masks:
            print(f"\n  {site}: Skipping (missing data for one or both years)")
            continue

        mask_2020 = masks[2020]
        mask_2024 = masks[2024]

        ha_2020 = hectares_from_mask(mask_2020, pixel_size_m=10.0)
        ha_2024 = hectares_from_mask(mask_2024, pixel_size_m=10.0)

        stock_2020 = carbon_stock(ha_2020)
        stock_2024 = carbon_stock(ha_2024)
        flux = carbon_flux(ha_2020, ha_2024, years=4)

        results[site] = {
            "baseline_year": 2020,
            "current_year": 2024,
            "baseline_stock": stock_2020,
            "current_stock": stock_2024,
            "flux": flux,
        }

        print(f"\n  {site.upper().replace('_', ' ')}")
        print(f"  {'─' * 50}")
        print(f"    2020 area:          {ha_2020:>12.2f} ha")
        print(f"    2024 area:          {ha_2024:>12.2f} ha")
        print(f"    Delta area:         {flux['delta_hectares']:>12.2f} ha")
        print(f"    2020 stock:         {stock_2020['co2e_t']:>12.2f} tCO2e")
        print(f"    2024 stock:         {stock_2024['co2e_t']:>12.2f} tCO2e")
        print(f"    Annual flux:        {flux['annual_flux_tco2e']:>12.2f} tCO2e/yr")
        print(f"    Total flux (4yr):   {flux['total_flux_tco2e']:>12.2f} tCO2e")

    # Compute aggregate totals
    if results:
        total_baseline_ha = sum(r["baseline_stock"]["hectares"] for r in results.values())
        total_current_ha = sum(r["current_stock"]["hectares"] for r in results.values())
        total_flux = sum(r["flux"]["total_flux_tco2e"] for r in results.values())
        total_annual = sum(r["flux"]["annual_flux_tco2e"] for r in results.values())

        aggregate = {
            "total_baseline_hectares": round(total_baseline_ha, 4),
            "total_current_hectares": round(total_current_ha, 4),
            "total_delta_hectares": round(total_current_ha - total_baseline_ha, 4),
            "total_annual_flux_tco2e": round(total_annual, 4),
            "total_flux_tco2e": round(total_flux, 4),
        }

        print(f"\n  {'=' * 50}")
        print(f"  AGGREGATE (all sites)")
        print(f"  {'─' * 50}")
        print(f"    Total baseline:     {total_baseline_ha:>12.2f} ha")
        print(f"    Total current:      {total_current_ha:>12.2f} ha")
        print(f"    Total delta:        {total_current_ha - total_baseline_ha:>12.2f} ha")
        print(f"    Total annual flux:  {total_annual:>12.2f} tCO2e/yr")
        print(f"    Total flux (4yr):   {total_flux:>12.2f} tCO2e")
        print(f"  {'=' * 50}")
    else:
        aggregate = {}
        print("\n  No sites processed — cannot compute aggregate.")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "description": "IPCC Tier 1 carbon report from U-Net mangrove segmentation",
        "model_checkpoint": "models/unet_best.pt",
        "pixel_size_m": 10.0,
        "tile_size": PATCH_SIZE,
        "tile_stride": STRIDE,
        "sites": results,
        "aggregate": aggregate,
    }

    out_path = os.path.join(RESULTS_DIR, "carbon_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
