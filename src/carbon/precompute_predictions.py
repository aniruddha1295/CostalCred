"""Pre-compute mangrove predictions and carbon estimates for all sites."""
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

def main():
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

    for model_name in ["ndvi", "xgboost"]:
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
                        pred = (ndvi > best_threshold).astype(np.uint8)
                    else:
                        feats = extract_features(img)
                        pred = xgb_model.predict(feats).reshape(img.shape[1], img.shape[2]).astype(np.uint8)

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

    # Save all results
    os.makedirs("results", exist_ok=True)
    with open("results/carbon_predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to results/carbon_predictions.json")

if __name__ == "__main__":
    main()
