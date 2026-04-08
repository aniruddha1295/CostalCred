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
- Google Earth Engine account (sign up at https://earthengine.google.com/)
- NVIDIA GPU with CUDA (for U-Net training)

### 1. Clone and install dependencies

```bash
git clone https://github.com/<org>/coastalcred.git
cd coastalcred
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the database

```bash
docker compose up -d
```

This starts PostgreSQL 16 with PostGIS 3.4. The schema in `sql/` is auto-applied on first run.

### 3. Authenticate with Google Earth Engine

```bash
earthengine authenticate
```

### 4. Download raw data

After the team publishes raw data to the GitHub Release `v0.1-raw-data`:

```bash
python src/data_pipeline/download_data.py
```

This pulls Sentinel-2 GeoTIFFs and GMW shapefiles into `data/raw/`.

### 5. Prepare patches

```bash
python src/data_pipeline/align_masks.py
python src/data_pipeline/extract_patches.py
python src/data_pipeline/make_splits.py
```

Patches land in `data/patches/`, splits in `data/splits/`.

## Data

Data artifacts are **not** checked into Git (see `.gitignore`). They are distributed via GitHub Releases:

- `v0.1-raw-data` — Sentinel-2 composites + GMW shapefiles
- `v0.2-trained-models` — trained model weights

## Project Structure

```
src/
  data_pipeline/   — GEE fetch, mask alignment, patch extraction, splits
  models/
    ndvi/          — NDVI threshold baseline
    xgboost/       — XGBoost pixel classifier
    unet/          — U-Net segmentation (smp + ResNet-18)
  evaluation/      — Shared metrics (Precision, Recall, IoU, F1)
  carbon/          — IPCC Tier 1 carbon calculation
sql/               — PostgreSQL + PostGIS schema
api/               — OpenAPI spec
docs/              — Architecture docs, PRD, viva prep
reports/           — Phase 1 & Phase 2 report PDFs
notebooks/         — Exploration only (not production code)
results/           — Metric JSONs + comparison plots (committed)
```
