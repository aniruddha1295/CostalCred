# CoastalCred

Blockchain-based blue carbon registry and MRV system for mangrove ecosystems in India.

Sem VI mini project — RCOEM Dept. of Data Science, Nagpur.

## Team

| Member | Role |
|--------|------|
| Ansh Chopada (lead) | Architecture + Documentation Lead |
| Aniruddha Lahoti | Data Pipeline + Infra Lead |
| TBD | Deep Learning Lead |
| TBD | Classical ML + Evaluation + Carbon Lead |

Guide: Dr. Aarti Karandikar | Industry Mentor: Rishikesh Kale (Filecoin/Protocol Labs)

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU with CUDA (for U-Net training)

### 1. Clone and install dependencies

```bash
git clone https://github.com/aniruddha1295/CostalCred.git
cd CostalCred
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download raw data

Data is hosted on GitHub Releases (~3.9 GB total):

```bash
python src/data_pipeline/download_data.py
```

This pulls Sentinel-2 GeoTIFFs and GMW shapefiles into `data/raw/`.

### 3. Generate patches and splits

```bash
python src/data_pipeline/align_masks.py --all
python src/data_pipeline/extract_patches.py --all
python src/data_pipeline/make_splits.py
```

This produces:
- `data/patches/` — 256x256 image/mask patch pairs (.npy)
- `data/splits/` — train.txt, val.txt, test.txt manifests + norm_stats.json

### 4. Start the database (optional, for PostGIS work)

```bash
docker compose up -d
```

PostgreSQL 16 + PostGIS 3.4. Schema in `sql/` auto-applies on first run.

## Data

Data artifacts are **not** checked into Git (see `.gitignore`). They are distributed via GitHub Releases:

- [`v0.1-raw-data`](https://github.com/aniruddha1295/CostalCred/releases/tag/v0.1-raw-data) — Sentinel-2 composites + GMW shapefiles
- `v0.2-trained-models` — trained model weights (after Phase F)

### Sites and data summary

| Site | Location | Bbox size | Composite size |
|------|----------|-----------|---------------|
| Sundarbans | West Bengal | 1.5 x 1.0 deg | ~1 GB |
| Gulf of Kutch | Gujarat | 1.0 x 0.8 deg | ~460 MB |
| Pichavaram | Tamil Nadu | 0.2 x 0.15 deg | ~27 MB |

**Years:** 2020 (baseline) and 2024 (current) -- two-point flux measurement.

**Bands:** B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2) at 10m resolutions.

**Labels:** Global Mangrove Watch v3 (2020) polygons rasterized to match Sentinel-2 grids.

### Splits (site-level, no leakage)

| Split | Patches | Source |
|-------|---------|--------|
| Train | 6,374 | Sundarbans 2024 + Gulf of Kutch 2024 |
| Val | 708 | 10% held out from train |
| Test | 71 | Pichavaram 2024 (unseen site) |

2020 patches are kept separately for carbon flux calculation.

## Model Training (Phase D/E/F)

All models use `src/evaluation/metrics.py` for evaluation (Precision, Recall, IoU, F1). Save results as JSON to `results/`.

**NDVI Baseline** (Phase D):
- `ndvi = (B8 - B4) / (B8 + B4 + 1e-8)`, tune threshold on val set
- Output: `results/ndvi.json`

**XGBoost** (Phase E):
- 10 features per pixel: 6 bands + NDVI, EVI, NDWI, SAVI
- `scale_pos_weight` for class imbalance, early stopping on val
- Output: `results/xgboost.json` + `results/feature_importance.png`

**U-Net** (Phase F):
- `segmentation_models_pytorch.Unet`, ResNet-18 encoder, 6-channel input
- `BCEWithLogitsLoss` with `pos_weight`, mixed precision, batch size 8
- Output: `results/unet.json` + best checkpoint to GitHub Release `v0.2-trained-models`

**Comparison** (Phase G):
```bash
python src/evaluation/comparison.py
python src/carbon/ipcc_tier1.py
```

## Project Structure

```
src/
  data_pipeline/       — GEE fetch, mask alignment, patch extraction, splits
    fetch_sentinel2.py — Download Sentinel-2 composites from GEE
    align_masks.py     — Rasterize GMW polygons to match Sentinel-2 grids
    extract_patches.py — Cut 256x256 patches with stride 128
    make_splits.py     — Site-level train/val/test split + normalization stats
    download_data.py   — Pull raw data from GitHub Releases
  models/
    ndvi/              — NDVI threshold baseline
    xgboost/           — XGBoost pixel classifier
    unet/              — U-Net segmentation (smp + ResNet-18)
  evaluation/
    metrics.py         — Shared Precision/Recall/IoU/F1 (used by all 3 models)
    comparison.py      — Builds 3-model comparison table
  carbon/
    ipcc_tier1.py      — IPCC Tier 1 carbon stock + flux calculation
sql/                   — PostgreSQL + PostGIS schema (~23 tables)
api/                   — OpenAPI spec
docs/                  — PRD, architecture docs, viva prep
reports/               — Phase 1 & Phase 2 report PDFs
results/               — Metric JSONs + comparison plots (committed)
```

## Citations

- Copernicus Sentinel data [2020, 2024], processed by ESA
- Bunting et al. 2018, "The Global Mangrove Watch", Remote Sensing 10(10):1669
- IPCC 2013 Wetlands Supplement (carbon constants)
