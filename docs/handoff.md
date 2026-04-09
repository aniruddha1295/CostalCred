# CoastalCred — Team Handoff Document

**Date:** 2026-04-09
**Author:** Aniruddha Lahoti (Data Pipeline + Infra Lead)
**Status:** Phase B+C complete. Data pipeline delivered. ML team unblocked.

---

## What's been done

### Phase A — Environment & Access
- [x] GCP project `costal-492719` with Earth Engine API enabled
- [x] Service account `coastalcred-ee@costal-492719.iam.gserviceaccount.com`
- [x] GitHub repo with all teammates having write access
- [x] Repo scaffolding: directories, .gitignore, docker-compose, requirements.txt

### Phase B — Data Acquisition
- [x] 6 Sentinel-2 L2A median composites fetched via Google Earth Engine
- [x] All composites published to GitHub Release [`v0.1-raw-data`](https://github.com/aniruddha1295/CostalCred/releases/tag/v0.1-raw-data)

| Composite | Size | Scenes used |
|-----------|------|-------------|
| sundarbans_2024.tif | 1.1 GB | 280 |
| sundarbans_2020.tif | 958 MB | — |
| gulf_of_kutch_2024.tif | 472 MB | — |
| gulf_of_kutch_2020.tif | 458 MB | — |
| pichavaram_2024.tif | 27 MB | — |
| pichavaram_2020.tif | 26 MB | — |

**Image spec:** 6 bands (B2, B3, B4, B8, B11, B12), 10m resolution, EPSG:4326, uint16 scaled [0-10000].

### Phase C — Data Preparation
- [x] GMW v3 (2020) polygons rasterized to match each Sentinel-2 grid
- [x] 256x256 patches extracted with stride 128
- [x] Site-level train/val/test split (no data leakage)
- [x] Per-band normalization stats computed on training data only

**Patch counts:**

| Site | Total patches | Positive (has mangrove) | Negative kept |
|------|--------------|------------------------|---------------|
| Sundarbans (per year) | 5,815 | 5,283 | 532 |
| Gulf of Kutch (per year) | 1,267 | 813 | 454 |
| Pichavaram (per year) | 71 | 59 | 12 |

**Split strategy (site-level, prevents leakage):**

| Split | Patches | Source sites |
|-------|---------|-------------|
| Train | 6,374 | Sundarbans 2024 + Gulf of Kutch 2024 |
| Val | 708 | 10% random holdout from train pool (seed=42) |
| Test | 71 | Pichavaram 2024 (completely unseen site) |

2020 patches exist but are NOT in the splits — they are reserved for carbon flux calculation in Phase G.

**Normalization stats** (from `data/splits/norm_stats.json`, computed on train set only):

| Band | Mean | Std |
|------|------|-----|
| B2 (Blue) | 0.0518 | 0.0378 |
| B3 (Green) | 0.0686 | 0.0448 |
| B4 (Red) | 0.0605 | 0.0467 |
| B8 (NIR) | 0.1400 | 0.1125 |
| B11 (SWIR1) | 0.0818 | 0.0813 |
| B12 (SWIR2) | 0.0509 | 0.0594 |

### Shared code committed
- [x] `src/evaluation/metrics.py` — Precision, Recall, IoU, F1 (all models MUST use this)
- [x] `src/evaluation/comparison.py` — builds 3-model comparison table from result JSONs
- [x] `src/carbon/ipcc_tier1.py` — IPCC Tier 1 carbon stock + flux calculation
- [x] `sql/schema.sql` — PostgreSQL + PostGIS schema (~23 tables)
- [x] Alignment check PNGs in `results/` — visual proof masks align with satellite imagery

---

## How to get started (new teammate setup)

```bash
# 1. Clone
git clone https://github.com/aniruddha1295/CostalCred.git
cd CostalCred

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download raw data (~3.9 GB)
python src/data_pipeline/download_data.py

# 5. Generate aligned masks
python src/data_pipeline/align_masks.py --all

# 6. Extract patches
python src/data_pipeline/extract_patches.py --all

# 7. Create splits + normalization stats
python src/data_pipeline/make_splits.py
```

After step 7, you should have:
- `data/patches/{site}_{year}/img_NNNN.npy` — shape `(6, 256, 256)` float32, range [0, 1]
- `data/patches/{site}_{year}/mask_NNNN.npy` — shape `(256, 256)` uint8, values {0, 1}
- `data/splits/train.txt`, `val.txt`, `test.txt` — one patch path per line
- `data/splits/norm_stats.json` — per-band mean and std

---

## What needs to happen next

### Phase D — NDVI Baseline (owner: Classical ML Lead)

**Goal:** Fast win. Establishes a floor for model performance. ~1 hour of work.

**File:** `src/models/ndvi/baseline.py`

**Steps:**
1. Compute NDVI per patch: `ndvi = (B8 - B4) / (B8 + B4 + 1e-8)` where B8 = band index 3, B4 = band index 2
2. Try thresholds: 0.2, 0.3, 0.4, 0.5, 0.6 on **val set**
3. Pick the best threshold by IoU
4. Evaluate on **test set** using `src/evaluation/metrics.py`
5. Save results to `results/ndvi.json`

**Band index mapping** (in the .npy patches):
| Index | Band |
|-------|------|
| 0 | B2 (Blue) |
| 1 | B3 (Green) |
| 2 | B4 (Red) |
| 3 | B8 (NIR) |
| 4 | B11 (SWIR1) |
| 5 | B12 (SWIR2) |

**Result JSON format** (all models should follow this):
```json
{
  "model": "ndvi_threshold",
  "threshold": 0.4,
  "precision": 0.85,
  "recall": 0.72,
  "iou": 0.64,
  "f1": 0.78,
  "training_time_seconds": 0
}
```

### Phase E — XGBoost (owner: Classical ML Lead)

**Goal:** Classical ML pixel classifier. ~5-10 min training on CPU.

**Files:** `src/models/xgboost/features.py`, `train.py`, `evaluate.py`

**Steps:**
1. Per-pixel feature extraction (10 features per pixel):
   - 6 raw bands: B2, B3, B4, B8, B11, B12
   - NDVI = (B8 - B4) / (B8 + B4 + 1e-8)
   - EVI = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1)
   - NDWI = (B3 - B8) / (B3 + B8 + 1e-8)
   - SAVI = 1.5 * (B8 - B4) / (B8 + B4 + 0.5)
2. Subsample ~200k pixels from training patches (balanced classes)
3. Train XGBoost: `n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=<neg/pos ratio>`
4. Early stopping on val logloss
5. Feature importance plot -> `results/feature_importance.png`
6. Evaluate on test set -> `results/xgboost.json`

### Phase F — U-Net (owner: Deep Learning Lead, needs RTX 3050)

**Goal:** Deep learning segmentation. ~60-90 min training.

**Files:** `src/models/unet/dataset.py`, `model.py`, `train.py`, `evaluate.py`

**Architecture:**
- `segmentation_models_pytorch.Unet`, encoder=`resnet18`, encoder_weights=`imagenet`
- Input: 6 channels, Output: 1 channel (binary mask)
- Loss: `BCEWithLogitsLoss(pos_weight=<class_imbalance_ratio>)`
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-5
- Mixed precision: `torch.cuda.amp` (required for 4 GB VRAM)
- Batch size: 8 (drop to 4 + gradient accumulation if OOM)
- Epochs: 30-50, early stopping on val IoU
- Normalize inputs using `data/splits/norm_stats.json`

**VRAM tips (4 GB RTX 3050):**
- Patch size 256x256, base filters 32 (not 64)
- If OOM: batch_size=4 + gradient accumulation (effective batch=8)
- Colab free tier T4 (15 GB VRAM) is the fallback

**Output:**
- Best checkpoint -> upload to GitHub Release `v0.2-trained-models`
- Metrics -> `results/unet.json`
- 5 prediction visualizations -> `results/unet_pred_*.png` (RGB | GT mask | predicted mask)

### Phase G — Comparison + Carbon (owner: team)

**Goal:** Final deliverable tables.

**Steps:**
1. Run `python src/evaluation/comparison.py` to build the 3-row comparison table from `results/ndvi.json`, `results/xgboost.json`, `results/unet.json`
2. Run `python src/carbon/ipcc_tier1.py` to compute carbon flux per site using U-Net predictions on 2020 vs 2024

**Carbon math (IPCC Tier 1):**
```
stock_tCO2e = hectares * 230 * 0.47 * 3.67
flux_tCO2e  = (current_hectares - baseline_hectares) * 7.0 * years_elapsed
```

### Phase H — Documentation

- Phase 1 report PDF (architecture, schema, smart contracts, API specs)
- Phase 2 report PDF (data pipeline, model comparison, carbon results)
- Viva prep document

---

## Important conventions

1. **All models MUST use `src/evaluation/metrics.py`** for evaluation — no custom metric code. This ensures apples-to-apples comparison (FR-2.6).

2. **Results go in `results/` as JSON** following the format shown above. The comparison script reads from there.

3. **No plain accuracy.** Mangroves are ~2% of pixels. A model predicting all-zero gets 98% accuracy. We report Precision, Recall, IoU, F1. IoU is the primary headline metric.

4. **Normalize U-Net inputs** using the stats in `data/splits/norm_stats.json`. XGBoost and NDVI work on raw [0,1] values.

5. **Don't retrain on test data.** Pichavaram is held out. Only evaluate on it. Never tune hyperparameters on it.

6. **Commit result JSONs and plots to `results/`.** Don't commit model weights — upload those to GitHub Releases.

---

## Key files reference

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Full project context, architecture decisions, viva answers |
| `docs/coastalcred-prd.md` | Product requirements document |
| `src/evaluation/metrics.py` | Shared eval code — Precision, Recall, IoU, F1 |
| `src/evaluation/comparison.py` | Builds 3-model comparison table |
| `src/carbon/ipcc_tier1.py` | IPCC carbon stock + flux calculator |
| `data/splits/norm_stats.json` | Per-band mean/std for normalization |
| `data/splits/train.txt` | Training patch paths |
| `data/splits/val.txt` | Validation patch paths |
| `data/splits/test.txt` | Test patch paths |

---

## Questions?

Reach out to Aniruddha if the data pipeline or setup isn't working. Read `CLAUDE.md` for architecture decisions and viva prep.
