"""
Fetch Sentinel-2 L2A median composites from Google Earth Engine.

Produces one 6-band GeoTIFF per (site, year) pair:
  Bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)

Downloads in tiles (to stay under GEE's 50 MB limit) then merges with rasterio.

Usage:
  # Smoke test -- single site, single year
  python src/data_pipeline/fetch_sentinel2.py --site sundarbans --year 2024

  # All 6 composites
  python src/data_pipeline/fetch_sentinel2.py --all
"""

import argparse
import json
import math
import os
import tempfile
import shutil

import ee
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
KEY_FILE = "ee-key.json"
PROJECT = "costal-492719"


def authenticate():
    with open(KEY_FILE) as f:
        sa_email = json.load(f)["client_email"]
    credentials = ee.ServiceAccountCredentials(sa_email, KEY_FILE)
    ee.Initialize(credentials, project=PROJECT)
    print(f"Authenticated as {sa_email}")


# ---------------------------------------------------------------------------
# Site bounding boxes  (lon_min, lat_min, lon_max, lat_max)
# ---------------------------------------------------------------------------
SITES = {
    "sundarbans": {
        "bbox": [88.0, 21.5, 89.5, 22.5],
        "description": "Sundarbans, West Bengal",
    },
    "gulf_of_kutch": {
        "bbox": [69.5, 22.2, 70.5, 23.0],
        "description": "Gulf of Kutch, Gujarat",
    },
    "pichavaram": {
        "bbox": [79.7, 11.35, 79.9, 11.5],
        "description": "Pichavaram, Tamil Nadu",
    },
}

YEARS = [2020, 2024]
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
CLOUD_THRESH = 20  # max cloud cover % per scene
SCALE = 10  # metres per pixel
TILE_DEG = 0.15  # tile size in degrees (keeps each tile under ~50 MB)

OUT_DIR = os.path.join("data", "raw", "sentinel2")


# ---------------------------------------------------------------------------
# Cloud masking (QA60 bitmask)
# ---------------------------------------------------------------------------
def mask_clouds(image):
    """Mask opaque and cirrus clouds using Sentinel-2 QA60 band."""
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000)  # scale to [0, 1]


# ---------------------------------------------------------------------------
# Build composite
# ---------------------------------------------------------------------------
def build_composite(site_name, year):
    """Return a cloud-free median composite ee.Image for a site and year."""
    site = SITES[site_name]
    bbox = ee.Geometry.Rectangle(site["bbox"])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bbox)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THRESH))
        .map(mask_clouds)
        .select(BANDS)
    )

    count = collection.size().getInfo()
    print(f"  {site_name} {year}: {count} scenes after cloud filter")

    composite = collection.median().clip(bbox)
    return composite


# ---------------------------------------------------------------------------
# Tiled download helpers
# ---------------------------------------------------------------------------
def make_tiles(bbox, tile_deg):
    """Split a bbox into a grid of smaller bboxes."""
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = []
    lon = lon_min
    while lon < lon_max:
        lat = lat_min
        while lat < lat_max:
            t = [
                lon,
                lat,
                min(lon + tile_deg, lon_max),
                min(lat + tile_deg, lat_max),
            ]
            tiles.append(t)
            lat += tile_deg
        lon += tile_deg
    return tiles


def download_tile(composite, tile_bbox, out_path):
    """Download a single tile via ee.Image.getDownloadURL."""
    region = ee.Geometry.Rectangle(tile_bbox)

    # Convert to uint16 [0, 10000] for smaller file size
    image = composite.multiply(10000).toUint16()

    url = image.getDownloadURL({
        "region": region,
        "scale": SCALE,
        "format": "GEO_TIFF",
        "bands": BANDS,
    })

    import urllib.request
    urllib.request.urlretrieve(url, out_path)


def download_composite(site_name, year):
    """Download a full composite as tiled GeoTIFFs then merge."""
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{site_name}_{year}.tif")

    if os.path.exists(out_path):
        print(f"  {out_path} already exists, skipping.")
        return out_path

    site = SITES[site_name]
    tiles = make_tiles(site["bbox"], TILE_DEG)
    print(f"\nBuilding composite: {site_name} {year} ...")
    composite = build_composite(site_name, year)

    print(f"  Downloading {len(tiles)} tiles ...")
    tmp_dir = tempfile.mkdtemp(prefix=f"ee_{site_name}_{year}_")
    tile_paths = []

    for i, tile_bbox in enumerate(tiles):
        tile_path = os.path.join(tmp_dir, f"tile_{i:04d}.tif")
        try:
            download_tile(composite, tile_bbox, tile_path)
            tile_paths.append(tile_path)
            print(f"    Tile {i+1}/{len(tiles)} OK")
        except Exception as e:
            print(f"    Tile {i+1}/{len(tiles)} FAILED: {e}")

    # Merge tiles
    print(f"  Merging {len(tile_paths)} tiles ...")
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, out_transform = merge(datasets)
    for ds in datasets:
        ds.close()

    # Write merged file
    profile = rasterio.open(tile_paths[0]).profile.copy()
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=out_transform,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Done: {out_path} ({size_mb:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch Sentinel-2 composites from GEE")
    parser.add_argument("--site", choices=list(SITES.keys()), help="Single site to fetch")
    parser.add_argument("--year", type=int, choices=YEARS, help="Single year to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch all 6 composites")
    args = parser.parse_args()

    authenticate()

    if args.all:
        for site in SITES:
            for year in YEARS:
                download_composite(site, year)
    elif args.site and args.year:
        download_composite(args.site, args.year)
    else:
        parser.error("Provide --site and --year, or --all")


if __name__ == "__main__":
    main()
