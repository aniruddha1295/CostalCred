"""Pre-compute mangrove predictions and carbon estimates for all sites.

Supports three models: NDVI threshold, XGBoost, and U-Net.
NDVI and XGBoost run per-pixel on patches directly.
U-Net uses a PyTorch model checkpoint for inference.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
import xgboost as xgb
from glob import glob
from tqdm import tqdm

# IPCC constants
BIOMASS_DENSITY = 230.0
CARBON_FRACTION = 0.47
CO2_TO_C_RATIO = 44.0 / 12.0
ANNUAL_SEQUESTRATION = 7.0

SITES = ["sundarbans", "gulf_of_kutch", "pichavaram"]
YEARS = ["2020", "2024"]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def extract_features(img):
    B2, B3, B4, B8, B11, B12 = img[0], img[1], img[2], img[3], img[4], img[5]
    ndvi = (B8 - B4) / (B8 + B4 + 1e-8)
    evi = 2.5 * (B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0)
    ndwi = (B3 - B8) / (B3 + B8 + 1e-8)
    savi = 1.5 * (B8 - B4) / (B8 + B4 + 0.5)
    h, w = B2.shape
    features = np.stack([B2, B3, B4, B8, B11, B12, ndvi, evi, ndwi, savi], axis=0)
    features = features.reshape(10, h * w).T
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def compute_carbon(mangrove_pixels, pixel_size_m=10.0):
    pixel_area_ha = (pixel_size_m ** 2) / 10000.0
    hectares = float(mangrove_pixels) * pixel_area_ha
    stock = hectares * BIOMASS_DENSITY * CARBON_FRACTION * CO2_TO_C_RATIO
    return hectares, stock


def load_unet_model(device):
    """Load the trained U-Net model from checkpoint."""
    import torch
    from src.models.unet.model import build_unet

    checkpoint_path = os.path.join(REPO_ROOT, "models", "unet_best.pt")
    if not os.path.isfile(checkpoint_path):
        print(f"  U-Net checkpoint not found at {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    print("  U-Net model loaded")

    # Load normalization stats if available
    norm_stats_path = os.path.join(REPO_ROOT, "data", "splits", "norm_stats.json")
    norm_stats = None
    if os.path.isfile(norm_stats_path):
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
        print("  Normalization stats loaded")

    return model, norm_stats


def predict_unet_patch(model, img, norm_stats, device):
    """Run U-Net inference on a single patch.

    Parameters
    ----------
    model : torch.nn.Module
        Trained U-Net in eval mode.
    img : np.ndarray
        Raw image patch, shape (6, H, W).
    norm_stats : dict or None
        Normalization stats with 'mean' and 'std' per band.
    device : torch.device
        Target device.

    Returns
    -------
    np.ndarray
        Binary prediction mask, shape (H, W), dtype uint8.
    """
    import torch

    img_tensor = torch.from_numpy(img.astype(np.float32))

    # Apply normalization if stats are available
    if norm_stats is not None:
        mean = torch.tensor(norm_stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(norm_stats["std"], dtype=torch.float32).view(-1, 1, 1)
        img_tensor = (img_tensor - mean) / (std + 1e-8)

    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))
    pred = (torch.sigmoid(logits).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return pred


def run_model_predictions(model_name, results, ndvi_info=None, xgb_model=None,
                          unet_model=None, unet_norm_stats=None, unet_device=None):
    """Run predictions for a single model across all sites and years."""
    results[model_name] = {}
    print(f"\n{'='*50}")
    print(f"Model: {model_name.upper()}")
    print(f"{'='*50}")

    for site in SITES:
        results[model_name][site] = {}

        for year in YEARS:
            patch_dir = f"data/patches/{site}_{year}"
            img_files = sorted(glob(os.path.join(patch_dir, "img_*.npy")))

            if not img_files:
                print(f"  {site} {year}: no patches found, skipping")
                continue

            total_mangrove_pixels = 0
            total_gt_pixels = 0
            total_pixels = 0

            # Also save 5 sample patches for visualization
            sample_indices = np.random.RandomState(42).choice(
                len(img_files), size=min(5, len(img_files)), replace=False
            )
            sample_patches = []

            for i, img_path in enumerate(tqdm(img_files, desc=f"  {site} {year} ({model_name})")):
                img = np.load(img_path)
                mask_path = img_path.replace("img_", "mask_")
                gt_mask = np.load(mask_path) if os.path.exists(mask_path) else None

                if model_name == "ndvi":
                    ndvi = (img[3] - img[2]) / (img[3] + img[2] + 1e-8)
                    pred = (ndvi > ndvi_info["best_threshold"]).astype(np.uint8)
                elif model_name == "xgboost":
                    feats = extract_features(img)
                    pred = xgb_model.predict(feats).reshape(img.shape[1], img.shape[2]).astype(np.uint8)
                elif model_name == "unet":
                    pred = predict_unet_patch(unet_model, img, unet_norm_stats, unet_device)

                total_mangrove_pixels += int(pred.sum())
                total_pixels += pred.size
                if gt_mask is not None:
                    total_gt_pixels += int(gt_mask.sum())

                if i in sample_indices:
                    # Save RGB + pred + gt for visualization
                    rgb = np.clip(img[[2,1,0]].transpose(1,2,0) / 0.3, 0, 1)
                    sample_patches.append({
                        "rgb": rgb,
                        "pred": pred,
                        "gt": gt_mask
                    })

            pred_ha, pred_stock = compute_carbon(total_mangrove_pixels)
            gt_ha, gt_stock = compute_carbon(total_gt_pixels)

            results[model_name][site][year] = {
                "total_patches": len(img_files),
                "total_pixels": total_pixels,
                "predicted_mangrove_pixels": total_mangrove_pixels,
                "gt_mangrove_pixels": total_gt_pixels,
                "predicted_hectares": round(pred_ha, 2),
                "predicted_stock_tco2e": round(pred_stock, 2),
                "gt_hectares": round(gt_ha, 2),
                "gt_stock_tco2e": round(gt_stock, 2),
                "mangrove_fraction": round(total_mangrove_pixels / max(total_pixels, 1), 4),
            }

            print(f"    {site} {year}: {pred_ha:.1f} ha predicted, {gt_ha:.1f} ha ground truth")

            # Save sample patches as numpy arrays
            os.makedirs(f"results/samples/{model_name}/{site}_{year}", exist_ok=True)
            for idx, sp in enumerate(sample_patches):
                np.save(f"results/samples/{model_name}/{site}_{year}/rgb_{idx}.npy",
                        sp["rgb"].astype(np.float32))
                np.save(f"results/samples/{model_name}/{site}_{year}/pred_{idx}.npy",
                        sp["pred"].astype(np.uint8))
                if sp["gt"] is not None:
                    np.save(f"results/samples/{model_name}/{site}_{year}/gt_{idx}.npy",
                            sp["gt"].astype(np.uint8))

        # Compute carbon flux for this site
        if "2020" in results[model_name][site] and "2024" in results[model_name][site]:
            baseline_ha = results[model_name][site]["2020"]["predicted_hectares"]
            current_ha = results[model_name][site]["2024"]["predicted_hectares"]
            delta_ha = current_ha - baseline_ha
            annual_flux = delta_ha * ANNUAL_SEQUESTRATION
            total_flux = annual_flux * 4

            results[model_name][site]["carbon_flux"] = {
                "baseline_ha_2020": baseline_ha,
                "current_ha_2024": current_ha,
                "delta_ha": round(delta_ha, 2),
                "annual_flux_tco2e": round(annual_flux, 2),
                "total_flux_tco2e_4yr": round(total_flux, 2),
                "years": 4,
            }
            print(f"    FLUX: {delta_ha:+.1f} ha -> {total_flux:+,.0f} tCO2e (4yr)")


def add_unet_from_carbon_report(results):
    """Add U-Net predictions from carbon_report.json when patches/model unavailable.

    Falls back to deriving pixel counts from the hectare values in the
    pre-existing carbon_report.json (produced by the U-Net carbon pipeline).
    Ground truth values are taken from the NDVI entry (same underlying data).
    """
    report_path = os.path.join(REPO_ROOT, "results", "carbon_report.json")
    if not os.path.isfile(report_path):
        print("  carbon_report.json not found, cannot add U-Net predictions")
        return

    with open(report_path) as f:
        report = json.load(f)

    # Use NDVI ground truth as reference (same patches, same ground truth)
    ndvi_data = results.get("ndvi", {})

    results["unet"] = {}
    print(f"\n{'='*50}")
    print(f"Model: UNET (from carbon_report.json)")
    print(f"{'='*50}")

    for site in SITES:
        if site not in report.get("sites", {}):
            print(f"  {site}: not in carbon_report.json, skipping")
            continue

        site_report = report["sites"][site]
        results["unet"][site] = {}

        for year in YEARS:
            year_key = "baseline_stock" if year == "2020" else "current_stock"
            flux_key = "baseline_hectares" if year == "2020" else "current_hectares"

            if year_key not in site_report:
                continue

            pred_ha = site_report[year_key]["hectares"]
            pred_stock = site_report[year_key]["co2e_t"]
            # Reverse-engineer pixel count: hectares / 0.01 (each 10m pixel = 0.01 ha)
            pred_pixels = int(round(pred_ha / 0.01))

            # Ground truth from NDVI entry (same data source)
            ndvi_year = ndvi_data.get(site, {}).get(year, {})
            gt_pixels = ndvi_year.get("gt_mangrove_pixels", 0)
            gt_ha = ndvi_year.get("gt_hectares", 0.0)
            gt_stock = ndvi_year.get("gt_stock_tco2e", 0.0)
            total_patches = ndvi_year.get("total_patches", 0)
            total_pixels = ndvi_year.get("total_pixels", 0)

            results["unet"][site][year] = {
                "total_patches": total_patches,
                "total_pixels": total_pixels,
                "predicted_mangrove_pixels": pred_pixels,
                "gt_mangrove_pixels": gt_pixels,
                "predicted_hectares": round(pred_ha, 2),
                "predicted_stock_tco2e": round(pred_stock, 2),
                "gt_hectares": round(gt_ha, 2),
                "gt_stock_tco2e": round(gt_stock, 2),
                "mangrove_fraction": round(pred_pixels / max(total_pixels, 1), 4),
            }

            print(f"    {site} {year}: {pred_ha:.1f} ha predicted, {gt_ha:.1f} ha ground truth")

        # Carbon flux from the report
        if "flux" in site_report:
            flux = site_report["flux"]
            results["unet"][site]["carbon_flux"] = {
                "baseline_ha_2020": flux["baseline_hectares"],
                "current_ha_2024": flux["current_hectares"],
                "delta_ha": round(flux["delta_hectares"], 2),
                "annual_flux_tco2e": round(flux["annual_flux_tco2e"], 2),
                "total_flux_tco2e_4yr": round(flux["total_flux_tco2e"], 2),
                "years": flux["years"],
            }
            print(f"    FLUX: {flux['delta_hectares']:+.1f} ha -> {flux['total_flux_tco2e']:+,.0f} tCO2e (4yr)")


def main():
    import torch

    # Load NDVI threshold
    with open("results/ndvi.json") as f:
        ndvi_info = json.load(f)
    best_threshold = ndvi_info["best_threshold"]
    print(f"NDVI threshold: {best_threshold}")

    # Load XGBoost model
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgboost_model.json")
    print("XGBoost model loaded")

    results = {}

    # --- NDVI and XGBoost (per-pixel, no model loading needed beyond XGBoost) ---
    for model_name in ["ndvi", "xgboost"]:
        run_model_predictions(model_name, results,
                              ndvi_info=ndvi_info, xgb_model=xgb_model)

    # --- U-Net ---
    # Check if patches exist for at least one site
    has_patches = any(
        glob(f"data/patches/{site}_{year}/img_*.npy")
        for site in SITES for year in YEARS
    )

    if has_patches:
        # Full U-Net inference from patches
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nU-Net device: {device}")
        unet_result = load_unet_model(device)
        if unet_result is not None:
            unet_model, unet_norm_stats = unet_result
            run_model_predictions("unet", results,
                                  unet_model=unet_model,
                                  unet_norm_stats=unet_norm_stats,
                                  unet_device=device)
        else:
            print("\n  U-Net checkpoint missing; falling back to carbon_report.json")
            add_unet_from_carbon_report(results)
    else:
        # No patches on disk — derive U-Net predictions from carbon_report.json
        print("\n  No patches found; deriving U-Net predictions from carbon_report.json")
        add_unet_from_carbon_report(results)

    # Save all results
    os.makedirs("results", exist_ok=True)
    with open("results/carbon_predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to results/carbon_predictions.json")

if __name__ == "__main__":
    main()
