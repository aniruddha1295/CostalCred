"""Add U-Net predictions to carbon_predictions.json.

Derives U-Net carbon prediction data from the existing carbon_report.json
(produced by the U-Net carbon pipeline) and merges it into
carbon_predictions.json alongside the existing NDVI and XGBoost entries.

Ground truth pixel counts and hectares are taken from the NDVI entry since
all three models operate on the same underlying data patches.

Pixel counts are reverse-engineered from hectare values:
    pixels = hectares / 0.01  (each 10m Sentinel-2 pixel = 0.01 ha)
"""

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SITES = ["sundarbans", "gulf_of_kutch", "pichavaram"]
YEARS = ["2020", "2024"]


def main():
    # Load existing carbon_predictions.json
    predictions_path = os.path.join(REPO_ROOT, "results", "carbon_predictions.json")
    with open(predictions_path) as f:
        predictions = json.load(f)

    # Load carbon_report.json (U-Net carbon pipeline output)
    report_path = os.path.join(REPO_ROOT, "results", "carbon_report.json")
    with open(report_path) as f:
        report = json.load(f)

    # Use NDVI as reference for ground truth values (same data, same GT)
    ndvi_data = predictions["ndvi"]

    unet = {}

    for site in SITES:
        site_report = report["sites"][site]
        ndvi_site = ndvi_data[site]
        unet[site] = {}

        for year in YEARS:
            year_key = "baseline_stock" if year == "2020" else "current_stock"

            pred_ha = site_report[year_key]["hectares"]
            pred_stock = site_report[year_key]["co2e_t"]
            pred_pixels = int(round(pred_ha / 0.01))

            # Ground truth from NDVI entry (same data source for all models)
            ndvi_year = ndvi_site[year]
            gt_pixels = ndvi_year["gt_mangrove_pixels"]
            gt_ha = ndvi_year["gt_hectares"]
            gt_stock = ndvi_year["gt_stock_tco2e"]
            total_patches = ndvi_year["total_patches"]
            total_pixels = ndvi_year["total_pixels"]

            unet[site][year] = {
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

            print(f"  {site} {year}: {pred_ha:.1f} ha predicted, {gt_ha:.1f} ha ground truth")

        # Carbon flux
        flux = site_report["flux"]
        unet[site]["carbon_flux"] = {
            "baseline_ha_2020": flux["baseline_hectares"],
            "current_ha_2024": flux["current_hectares"],
            "delta_ha": round(flux["delta_hectares"], 2),
            "annual_flux_tco2e": round(flux["annual_flux_tco2e"], 2),
            "total_flux_tco2e_4yr": round(flux["total_flux_tco2e"], 2),
            "years": flux["years"],
        }
        print(f"  {site} FLUX: {flux['delta_hectares']:+.1f} ha -> {flux['total_flux_tco2e']:+,.0f} tCO2e (4yr)")

    # Merge into predictions
    predictions["unet"] = unet

    # Write back
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nU-Net predictions added to {predictions_path}")
    print(f"Keys now present: {list(predictions.keys())}")


if __name__ == "__main__":
    main()
