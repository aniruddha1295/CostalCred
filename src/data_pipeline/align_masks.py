"""
Rasterize Global Mangrove Watch (GMW) v3 polygons to binary masks
aligned to Sentinel-2 composites.

Produces one single-band uint8 GeoTIFF per (site, year) pair:
  1 = mangrove, 0 = not mangrove

The output mask has identical CRS, transform, width, and height as the
corresponding Sentinel-2 composite so they can be stacked directly.

Usage:
  # Single site + year
  python src/data_pipeline/align_masks.py --site sundarbans --year 2024

  # All 6 composites
  python src/data_pipeline/align_masks.py --all

  # Override GMW data path
  python src/data_pipeline/align_masks.py --all --gmw data/raw/gmw/custom.gpkg
"""

import argparse
import glob
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Constants (must match fetch_sentinel2.py)
# ---------------------------------------------------------------------------
SITES = ["sundarbans", "gulf_of_kutch", "pichavaram"]
YEARS = [2020, 2024]

SENTINEL2_DIR = os.path.join("data", "raw", "sentinel2")
GMW_DIR = os.path.join("data", "raw", "gmw")
RESULTS_DIR = "results"

# Default filenames to search for GMW data (in priority order)
GMW_CANDIDATES = ["gmw_v3.gpkg", "gmw_v3.shp"]


# ---------------------------------------------------------------------------
# Locate GMW vector file
# ---------------------------------------------------------------------------
def find_gmw_file(gmw_override=None):
    """Find the GMW vector file, checking override then defaults."""
    if gmw_override:
        if not os.path.exists(gmw_override):
            raise FileNotFoundError(f"GMW file not found: {gmw_override}")
        return gmw_override

    # Check named candidates first
    for name in GMW_CANDIDATES:
        path = os.path.join(GMW_DIR, name)
        if os.path.exists(path):
            return path

    # Fall back to any .gpkg or .shp in the directory
    for ext in ("*.gpkg", "*.shp"):
        matches = glob.glob(os.path.join(GMW_DIR, ext))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No GMW vector file found in {GMW_DIR}. "
        "Download GMW v3 from https://data.unep-wcmc.org/datasets/45 "
        "or pass --gmw <path>."
    )


# ---------------------------------------------------------------------------
# Core alignment logic
# ---------------------------------------------------------------------------
def align_mask(site, year, gmw_path):
    """
    Rasterize GMW polygons to match a Sentinel-2 composite exactly.

    Returns the path to the output mask GeoTIFF.
    """
    sentinel_path = os.path.join(SENTINEL2_DIR, f"{site}_{year}.tif")
    if not os.path.exists(sentinel_path):
        raise FileNotFoundError(
            f"Sentinel-2 composite not found: {sentinel_path}. "
            "Run fetch_sentinel2.py first."
        )

    out_path = os.path.join(GMW_DIR, f"{site}_{year}_mask.tif")
    if os.path.exists(out_path):
        print(f"  {out_path} already exists, skipping.")
        return out_path

    # 1. Read reference raster metadata
    with rasterio.open(sentinel_path) as src:
        ref_crs = src.crs
        ref_transform = src.transform
        ref_width = src.width
        ref_height = src.height
        ref_bounds = src.bounds

    print(f"\n  Reference: {sentinel_path}")
    print(f"    CRS={ref_crs}  Size={ref_width}x{ref_height}")
    print(f"    Bounds={ref_bounds}")

    # 2. Read GMW vectors, clipped to the Sentinel-2 bounding box
    #    Use bbox filter to avoid loading the entire global dataset
    ref_box = box(ref_bounds.left, ref_bounds.bottom,
                  ref_bounds.right, ref_bounds.top)

    print(f"  Reading GMW from {gmw_path} ...")
    gmw = gpd.read_file(gmw_path, bbox=ref_box)
    print(f"    {len(gmw)} features intersect the bounding box")

    if len(gmw) == 0:
        print("    WARNING: No GMW polygons found in this extent.")
        print("    Writing all-zero mask.")
        mask = np.zeros((ref_height, ref_width), dtype=np.uint8)
    else:
        # 3. Reproject GMW to match Sentinel-2 CRS if needed
        if gmw.crs and not gmw.crs.equals(ref_crs):
            print(f"    Reprojecting GMW from {gmw.crs} to {ref_crs} ...")
            gmw = gmw.to_crs(ref_crs)

        # 4. Clip geometries to the reference bounds
        gmw_clipped = gmw.clip(ref_box)
        valid = gmw_clipped[~gmw_clipped.is_empty & gmw_clipped.is_valid]
        print(f"    {len(valid)} features after clipping")

        if len(valid) == 0:
            mask = np.zeros((ref_height, ref_width), dtype=np.uint8)
        else:
            # 5. Rasterize
            shapes = [(geom, 1) for geom in valid.geometry]
            mask = rasterize(
                shapes,
                out_shape=(ref_height, ref_width),
                transform=ref_transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True,  # include edge pixels
            )

    mangrove_px = int(mask.sum())
    total_px = ref_height * ref_width
    pct = 100.0 * mangrove_px / total_px if total_px > 0 else 0
    print(f"    Mangrove pixels: {mangrove_px:,} / {total_px:,} ({pct:.2f}%)")

    # 6. Write output mask GeoTIFF
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": ref_width,
        "height": ref_height,
        "count": 1,
        "crs": ref_crs,
        "transform": ref_transform,
        "compress": "deflate",
        "nodata": None,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved: {out_path} ({size_kb:.1f} KB)")

    # 7. Visual verification
    save_alignment_preview(sentinel_path, mask, site, year)

    return out_path


# ---------------------------------------------------------------------------
# Visual sanity check
# ---------------------------------------------------------------------------
def save_alignment_preview(sentinel_path, mask, site, year):
    """Save an RGB + mask overlay image for visual verification."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_png = os.path.join(RESULTS_DIR, f"alignment_check_{site}_{year}.png")

    with rasterio.open(sentinel_path) as src:
        # Read RGB bands (B4=Red, B3=Green, B2=Blue → bands 3, 2, 1)
        red = src.read(3).astype(np.float32)   # B4
        green = src.read(2).astype(np.float32)  # B3
        blue = src.read(1).astype(np.float32)   # B2

    # Stack and normalize for display
    rgb = np.stack([red, green, blue], axis=-1)

    # Handle uint16 scaled data (values in 0-10000 range)
    p2 = np.percentile(rgb[rgb > 0], 2) if (rgb > 0).any() else 0
    p98 = np.percentile(rgb[rgb > 0], 98) if (rgb > 0).any() else 1
    if p98 <= p2:
        p98 = p2 + 1
    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: RGB
    axes[0].imshow(rgb)
    axes[0].set_title(f"Sentinel-2 RGB — {site} {year}")
    axes[0].axis("off")

    # Panel 2: Mask
    axes[1].imshow(mask, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title(f"GMW Mask — {site} {year}")
    axes[1].axis("off")

    # Panel 3: RGB + mask contour overlay
    axes[2].imshow(rgb)
    if mask.any():
        axes[2].contour(mask, levels=[0.5], colors=["lime"], linewidths=0.8)
    axes[2].set_title(f"Overlay — {site} {year}")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Preview: {out_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Rasterize GMW v3 polygons to match Sentinel-2 composites"
    )
    parser.add_argument(
        "--site", choices=SITES, help="Single site to process"
    )
    parser.add_argument(
        "--year", type=int, choices=YEARS, help="Single year to process"
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all 6 site-year combos"
    )
    parser.add_argument(
        "--gmw", type=str, default=None,
        help="Path to GMW vector file (default: auto-detect in data/raw/gmw/)"
    )
    args = parser.parse_args()

    gmw_path = find_gmw_file(args.gmw)
    print(f"Using GMW data: {gmw_path}")

    if args.all:
        for site in SITES:
            for year in YEARS:
                align_mask(site, year, gmw_path)
    elif args.site and args.year:
        align_mask(args.site, args.year, gmw_path)
    else:
        parser.error("Provide --site and --year, or --all")

    print("\nDone.")


if __name__ == "__main__":
    main()
