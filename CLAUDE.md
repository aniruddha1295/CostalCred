# CoastalCred — Claude Code Handoff Context

> Drop this file into your repo as `CLAUDE.md` at the root. Claude Code reads it automatically and will have full context of what we decided and why, without you re-explaining anything.

---

## What this project is

**CoastalCred** is a Sem VI mini project at RCOEM Dept. of Data Science (Nagpur) — a blockchain-based blue carbon registry and MRV system focused on mangrove ecosystems in India. Guide: Dr. Aarti Karandikar. Industry mentor: Rishikesh Kale (Filecoin/Protocol Labs).

The project runs across Sem VI (design + ML) and Sem VII (full stack + deployment). **This repo is the Sem VI scope.**

Team (4 members):
- **Aniruddha Lahoti** — Data Pipeline + Infra Lead (repo owner, primary user of Claude Code)
- **Ansh Chopada** (team lead) — Architecture + Documentation Lead
- **[TBD]** — Deep Learning Lead (U-Net, owns RTX 3050)
- **[TBD]** — Classical ML + Evaluation + Carbon Lead

## What we're building this sprint (Sem VI)

**Phase 1 — Design & Architecture (docs + POCs)**
- System architecture doc for 7-microservice design
- PostgreSQL schema with ~23 tables (PostGIS for geometry)
- Smart contract architecture spec (ERC-721 TerritoryNFT, ERC-1155 CarbonCredit, Marketplace) — spec only, not deployed
- OpenAPI spec for 7 services
- Repo skeleton + Docker Compose + CI/CD stub
- 3 POC spikes: PostGIS spatial query, Hardhat contract skeleton compilation, GEE Sentinel-2 fetch + NDVI

**Phase 2 — Data & ML (the main deliverable)**
- Data pipeline: Google Earth Engine → Sentinel-2 composites (6 bands: B2, B3, B4, B8, B11, B12) for 3 sites × 2 years
- Ground truth: Global Mangrove Watch (GMW) v3 polygons rasterized to match Sentinel-2
- Sites: **Sundarbans (WB), Gulf of Kutch (Gujarat), Pichavaram (TN)**
- Years: **2020 and 2024** (two-point flux, not stock — critical design decision)
- **Three models compared on the same evaluation code:**
  1. **NDVI threshold** (rule-based baseline)
  2. **XGBoost** (classical ML pixel classifier, with `scale_pos_weight` for class imbalance)
  3. **U-Net** (deep learning segmentation via `segmentation_models_pytorch`, ResNet-18 encoder, ImageNet pretrained, 6-channel input)
- Comparison table: Precision, Recall, IoU, F1, training time
- IPCC Tier 1 carbon calculation: hectares → stock tCO₂e → 4-year flux tCO₂e
- Phase 1 + Phase 2 report PDFs

## Explicitly out of scope for this sprint

- Smart contract deployment (design only)
- Microservices as separate deployed containers — use **modular monolith** during sprint; architecture *docs* honor the microservices design
- Frontend portals (Sem VII)
- Mobile app, Aadhaar, UPI integration (Sem VII)
- IPFS/Filecoin implementation (design only; Sem VII consultation with mentor)
- TimescaleDB implementation (design only)
- Kafka (design only; in-process events for sprint)
- DeepLabV3+/SegFormer (U-Net is sufficient)
- Real field verification data (GMW is proxy)

## Key architectural decisions (with reasoning — for the viva)

### 1. Flux, not stock
Carbon credits represent *change over time*, not accumulated existing carbon. A single satellite image can only measure stock; we use two time points (2020 and 2024) to measure flux. This mirrors real MRV methodology (Verra, Gold Standard).

**Viva answer:** *"We measure change in mangrove cover between a 2020 baseline and 2024 current state, then compute additional sequestration following IPCC flux methodology."*

### 2. Additionality framework
Credits should only be issued for change caused by human action, not natural growth. Real MRV combines three evidence streams:
1. Remote sensing (our ML model) — confirms change
2. Provenance documentation (Community Portal — Sem VII) — proves who caused it
3. Field verification (mobile app — Sem VII) — confirms ground truth

The ML model alone doesn't issue credits — the *architecture around it* (Aadhaar, IPFS, blockchain) is what makes additionality claims defensible.

### 3. Three models walking up complexity ladder
- NDVI = rules
- XGBoost = classical ML (per-pixel features: 6 bands + NDVI + EVI + NDWI + SAVI)
- U-Net = deep learning (spatial context)

Same data pipeline feeds all three. Same evaluation code scores all three. Comparison table is the Phase 2 highlight.

### 4. XGBoost over Random Forest
XGBoost handles class imbalance better via `scale_pos_weight` (mangroves are ~2% of pixels), typically 1-2% better IoU than RF on tabular features with correlated inputs like spectral bands, and gives feature importances for the report.

### 5. U-Net over DeepLabV3+/SegFormer
U-Net is the standard remote sensing baseline, small enough for 4GB VRAM, abundant reference code, defensible at viva. Heavier architectures are a Sem VII upgrade.

### 6. Site-level train/test split (not random patch split)
Train: Sundarbans 2024 + Gulf of Kutch 2024. Test: Pichavaram 2024 (unseen).
Random splitting causes leakage — patches from the same forest in both train and test. Site-level splitting honestly measures generalization.

### 7. Polygon Amoy, not Mumbai
Original synopsis says "Polygon Mumbai" but Mumbai was deprecated April 2024. All new docs should reference **Polygon Amoy** (current testnet). Flag this correction to Dr. Karandikar proactively.

### 8. IPCC Tier 1 carbon math (the 4-step chain)
```
hectares × 230 t/ha (biomass density)
        × 0.47 (carbon fraction)
        × 44/12 ≈ 3.67 (CO₂:C molecular ratio)
        = tCO₂e stock

flux_tCO₂e = (current_hectares − baseline_hectares) × 7.0 × years
```
All constants from IPCC Wetlands Supplement (2013). These are published defaults — we do NOT invent numbers.

### 9. Metrics — no plain accuracy
Mangroves are ~2% of pixels → a model predicting all-zero scores 98% accuracy. We report **Precision, Recall, IoU, F1** instead. IoU is the primary headline metric.

### 10. Data sources (both free)
- **Images:** Sentinel-2 L2A via Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED`)
- **Labels:** Global Mangrove Watch v3 (https://data.unep-wcmc.org/datasets/45)

Cite as: *Copernicus Sentinel data [2020, 2024], processed by ESA* and *Bunting et al. 2018, Remote Sensing 10(10):1669*.

## Repo structure (create this)

```
coastalcred/
├── CLAUDE.md                    ← this file
├── README.md                    ← team onboarding instructions
├── .gitignore                   ← see below
├── docker-compose.yml           ← Postgres + PostGIS
├── requirements.txt
├── src/
│   ├── data_pipeline/
│   │   ├── fetch_sentinel2.py   ← GEE fetch
│   │   ├── align_masks.py       ← GMW rasterization
│   │   ├── extract_patches.py   ← Patch cutting
│   │   ├── make_splits.py       ← Site-level train/val/test
│   │   └── download_data.py     ← Fetch raw data from GitHub Release
│   ├── models/
│   │   ├── ndvi/
│   │   │   └── baseline.py
│   │   ├── xgboost/
│   │   │   ├── features.py
│   │   │   ├── train.py
│   │   │   └── evaluate.py
│   │   └── unet/
│   │       ├── dataset.py
│   │       ├── model.py
│   │       ├── train.py
│   │       └── evaluate.py
│   ├── evaluation/
│   │   ├── metrics.py           ← Shared Precision/Recall/IoU/F1
│   │   └── comparison.py        ← Builds the 3-model comparison table
│   └── carbon/
│       └── ipcc_tier1.py        ← Carbon calculation
├── sql/
│   └── schema.sql               ← ~23 tables
├── docs/
│   ├── prd.md                   ← the full PRD
│   ├── architecture.md
│   ├── erd.png
│   ├── smart_contracts.md
│   ├── tech_stack_justifications.md
│   └── viva_prep.md
├── api/
│   └── openapi.yaml
├── reports/
│   ├── phase1_report.pdf
│   └── phase2_report.pdf
├── notebooks/                   ← exploration only
├── data/                        ← gitignored
│   ├── raw/{sentinel2,gmw}/
│   ├── patches/
│   └── splits/
├── models/                      ← gitignored (use GitHub Releases for trained weights)
└── results/                     ← small JSON metric files + comparison PNGs, committed
```

## .gitignore (must have from day one)

```
# Data artifacts
data/
models/
*.tif
*.tiff
*.npy
*.pt
*.pth
*.h5
*.gpkg
*.shp
*.shx
*.dbf
*.prj

# Python
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/

# Secrets
.env
*.key
credentials.json

# IDE
.vscode/
.idea/
.DS_Store
```

## Data sharing strategy

**GitHub Releases, not Git LFS, not Google Drive.**

- Raw Sentinel-2 GeoTIFFs + GMW files → uploaded as release assets to tag `v0.1-raw-data`
- Trained model weights → uploaded to `v0.2-trained-models`
- Teammates run `python src/data_pipeline/download_data.py` after clone → pulls from the release
- Teammates regenerate patches locally by running `extract_patches.py` → patches are NOT shared, they're derived

Why: versioned with code, no external service, no LFS quota issues, one-command setup for teammates.

## Execution plan (follow this order — phases have dependencies)

### Phase A — Environment & Access
1. Sign up for Google Earth Engine (https://earthengine.google.com/) — **do this immediately, approval can take hours**
2. Verify CUDA on RTX 3050: `python -c "import torch; print(torch.cuda.is_available())"` → must print `True`
3. Create GitHub repo, invite 3 teammates with Write access
4. Install Python stack: `numpy pandas scikit-learn xgboost rasterio geopandas shapely earthengine-api geemap torch torchvision segmentation_models_pytorch matplotlib seaborn tqdm`
5. `earthengine authenticate` on Aniruddha's machine

### Phase B — Data Acquisition (critical path starts here)
1. **Smoke test first:** fetch ONE site, ONE year (Sundarbans 2024). Verify GeoTIFF opens with rasterio and RGB preview looks like Sundarbans.
2. Scale to all 6 composites (3 sites × 2 years)
3. Download GMW v3 shapefiles, clip to 3 site bounding boxes
4. Publish release `v0.1-raw-data` with all 9 files
5. Eye-check: overlay GMW polygons on Sentinel-2 RGB — do they match where mangroves actually are?

### Phase C — Data Preparation (critical path bottleneck)
1. Rasterize GMW polygons to match Sentinel-2 CRS/resolution/bounds exactly
2. Verify alignment with visual overlay
3. Extract 256×256 patches (stride 128 for overlap), skip all-zero masks but keep some negatives
4. Site-level split: Train=Sundarbans+Kutch, Val=10% of train, Test=Pichavaram
5. Compute normalization stats (per-band mean/std) on training data only

### Phase D — NDVI Baseline (fast win, unblocks evaluation code)
1. NDVI = (B8 - B4) / (B8 + B4 + 1e-8)
2. Tune threshold on validation set (try 0.2, 0.3, 0.4, 0.5, 0.6)
3. Evaluate on test set, save `results/ndvi.json`

### Phase E — XGBoost
1. Per-pixel features: 6 bands + NDVI + EVI + NDWI + SAVI = 10 features
2. Subsample ~200k pixels with class balance
3. Train with `scale_pos_weight`, early stopping on val
4. Feature importance plot → `results/feature_importance.png`
5. Evaluate on test set, save `results/xgboost.json`

### Phase F — U-Net (the main event)
1. Use `segmentation_models_pytorch` Unet, ResNet-18 encoder, ImageNet weights, 6 input channels, 1 output class
2. `BCEWithLogitsLoss` with `pos_weight`
3. Batch size 8 (drop to 4 + grad accumulation if OOM on 4GB VRAM)
4. Mixed precision (`torch.cuda.amp`)
5. 30-50 epochs, early stopping on val IoU
6. Save best checkpoint, upload to release `v0.2-trained-models`
7. Evaluate on test set, save `results/unet.json`
8. Save 5 prediction visualizations (RGB | GT mask | predicted mask)

### Phase G — Comparison + Carbon
1. Build 3-row comparison table from result JSONs
2. Per-site breakdown
3. Implement IPCC Tier 1 carbon calculation
4. Run U-Net on 2020 and 2024 composites per site → compute flux credits

### Phase H — Documentation
1. Phase 1 report PDF
2. Phase 2 report PDF
3. Viva prep document with answers to 7 standard questions

## VRAM / training notes

- 4 GB VRAM is tight
- Patch size 256×256, base filters 32 (not 64), batch size 8
- If OOM: batch size 4 + `torch.cuda.amp` mixed precision + gradient accumulation
- Colab free tier T4 (15 GB VRAM) is the fallback — set up an account before you need it

## The viva questions (team must answer all 7 confidently)

1. How does CoastalCred prevent fraud and ensure additionality?
2. Why U-Net and not DeepLabV3+ or SegFormer?
3. Why XGBoost and not Random Forest?
4. Why measure flux instead of stock?
5. How do you convert mangrove hectares to tCO₂e?
6. What's the class imbalance problem and how did you handle it?
7. Why three models and not one?

## The critical path

```
GEE auth → Sentinel-2 fetch → GMW alignment → patch extraction → splits → U-Net training → evaluation → comparison table → Phase 2 report
```

If this chain stalls at any step, the whole sprint stalls. **Aniruddha owns most of this chain** (everything up through splits). If Aniruddha is blocked, the team drops what they're doing and unblocks him.

## The one rule that matters most

**Do not write U-Net training code before data exists on disk.** Everyone will want to. Resist. The U-Net is ~200 lines. The data pipeline is what makes or breaks the project. Ship Phase B and C first.

---

*This handoff doc was written by Claude (chat) before execution moved to Claude Code. Everything above is locked and agreed. Execute in the order described, respect the dependency chain, and don't second-guess decisions already made — they were made for reasons captured here.*
