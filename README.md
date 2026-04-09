# CoastalCred — Blockchain-Based Blue Carbon Registry & MRV System

> Sem VI Mini Project | RCOEM Dept. of Data Science, Nagpur
> Guide: Dr. Aarti Karandikar | Industry Mentor: Rishikesh Kale (Filecoin/Protocol Labs)

## Team
| Name | Role |
|------|------|
| Ansh Chopada (Lead) | Architecture + Documentation Lead |
| Aniruddha Lahoti | Data Pipeline + Infra Lead |
| TBD | Deep Learning Lead |
| TBD | Classical ML + Evaluation + Carbon Lead |

## Project Overview

CoastalCred is a blockchain-based blue carbon registry and MRV (Monitoring, Reporting, and Verification) system focused on mangrove ecosystems in India. It uses satellite imagery (Sentinel-2) and machine learning to detect mangrove cover change, then converts that change into verifiable carbon credits using IPCC Tier 1 methodology.

The system compares three models of increasing complexity:
1. **NDVI Threshold** — rule-based vegetation index baseline
2. **XGBoost** — classical ML pixel classifier with 10 spectral features  
3. **U-Net** — deep learning semantic segmentation (spatial context)

## Study Sites

| Site | State | Ecosystem | Role |
|------|-------|-----------|------|
| Sundarbans | West Bengal | Gangetic delta | Training |
| Gulf of Kutch | Gujarat | Arid coastal | Training |
| Pichavaram | Tamil Nadu | Backwater estuary | Test (unseen) |

**Years:** 2020 (baseline) → 2024 (current) — 4-year flux measurement

## Data Pipeline

**Source:** Sentinel-2 L2A surface reflectance via Google Earth Engine
**Labels:** Global Mangrove Watch v3 (2020) polygons rasterized to match Sentinel-2

| Band | Sentinel-2 | Wavelength | Use |
|------|-----------|------------|-----|
| B2 | Blue | 490 nm | Water penetration, soil |
| B3 | Green | 560 nm | Vegetation vigor, NDWI |
| B4 | Red | 665 nm | Chlorophyll absorption, NDVI |
| B8 | NIR | 842 nm | Vegetation reflectance, NDVI |
| B11 | SWIR-1 | 1610 nm | Moisture content |
| B12 | SWIR-2 | 2190 nm | Mineral/soil discrimination |

### Patch Extraction
- Patch size: 256×256 pixels (2.56 km × 2.56 km at 10m resolution)
- Stride: 128 (50% overlap for training augmentation)
- Format: `.npy` arrays, float32, scaled [0, 1]

### Site-Level Train/Test Split (Prevents Data Leakage)

| Split | Patches | Source | Purpose |
|-------|---------|--------|---------|
| Train | 6,374 | Sundarbans + Gulf of Kutch (2024) | Model training |
| Val | 708 | 10% holdout from train (seed=42) | Hyperparameter tuning |
| Test | 71 | Pichavaram 2024 (unseen site) | Generalization measurement |

**Why site-level split?** Random patch splitting causes data leakage — adjacent patches from the same forest share spectral patterns. Site-level splitting honestly measures whether the model generalizes to ecosystems it has never seen.

## Model Results

### Comparison Table

| Metric | NDVI Threshold | XGBoost | U-Net (pending) |
|--------|---------------|---------|-----------------|
| **Val Precision** | 0.866 | 0.693 | — |
| **Val Recall** | 0.627 | 0.966 | — |
| **Val IoU** | 0.572 | 0.676 | — |
| **Val F1** | 0.728 | 0.807 | — |
| **Test Precision** | 0.428 | 0.438 | — |
| **Test Recall** | 0.610 | 0.580 | — |
| **Test IoU** | **0.336** | **0.332** | — |
| **Test F1** | 0.503 | 0.499 | — |
| Training Time | 0 sec | 2.79 sec | ~60-90 min |

**Primary metric:** IoU (Intersection over Union) — robust under class imbalance.

### Model 1: NDVI Threshold Baseline
- **Algorithm:** NDVI = (NIR − Red) / (NIR + Red + ε), predict mangrove if NDVI > threshold
- **Best threshold:** 0.55 (tuned on validation set by IoU)
- **Strengths:** Physics-based, zero training time, generalizable (universal vegetation property)
- **Weaknesses:** No spectral or spatial context, single-feature decision boundary

### Model 2: XGBoost Pixel Classifier
- **Algorithm:** Gradient boosted trees on 10 per-pixel spectral features
- **Features (10):** B2, B3, B4, B8, B11, B12, NDVI, EVI, NDWI, SAVI
- **Hyperparameters:** 300 trees (287 used, early stopping), max_depth=6, lr=0.1, scale_pos_weight=1.47
- **Top 3 features:** NDVI (0.456), B2 Blue (0.214), B11 SWIR (0.099)
- **Strengths:** Multi-feature decision, feature importance interpretability, fast training
- **Weaknesses:** No spatial context (pixel-independent), overfits to training site spectral signatures

### Model 3: U-Net (Pending — Teammate Assignment)
- **Architecture:** segmentation_models_pytorch Unet, ResNet-18 encoder, ImageNet pretrained
- **Input:** 6 channels, 256×256 patches | **Output:** binary mask
- **Expected advantage:** Spatial context (convolutional receptive field captures neighborhood patterns)

### Key Finding: The Generalization Gap

| Model | Val IoU | Test IoU | Gap |
|-------|---------|----------|-----|
| NDVI | 0.572 | 0.336 | 0.236 |
| XGBoost | 0.676 | 0.332 | **0.344** |

XGBoost achieves higher validation IoU but a **larger generalization gap**. It memorized Sundarbans/Kutch spectral signatures that don't transfer to Pichavaram. NDVI's physics-based approach generalizes more naturally. This demonstrates why spatial context (U-Net) is needed for cross-site generalization.

## Carbon Credit Estimation (IPCC Tier 1)

### Constants (IPCC 2013 Wetlands Supplement)

| Constant | Value | Source |
|----------|-------|--------|
| Biomass density | 230 t/ha (dry matter) | IPCC default for mangroves |
| Carbon fraction | 0.47 | IPCC default |
| CO₂:C ratio | 44/12 ≈ 3.667 | Molecular weight |
| Annual sequestration | 7.0 tCO₂e/ha/year | IPCC default |

### Stock Formula
```
stock_tCO₂e = hectares × 230 × 0.47 × 3.667
```

### Flux Formula (Carbon Credits)
```
flux_tCO₂e = (current_hectares − baseline_hectares) × 7.0 × years
```

**Positive flux** = mangrove area grew = carbon credits earned
**Negative flux** = mangrove area lost = carbon debt

## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (for U-Net only)
- Google Earth Engine account (for data acquisition only)

### Setup
```bash
git clone https://github.com/aniruddha1295/CostalCred.git
cd CostalCred
pip install -r requirements.txt

# Download raw data (~3.9 GB)
python src/data_pipeline/download_data.py

# Prepare patches and splits
python src/data_pipeline/align_masks.py --all
python src/data_pipeline/extract_patches.py --all
python src/data_pipeline/make_splits.py
```

### Run Models
```bash
# NDVI baseline
python src/models/ndvi/baseline.py

# XGBoost
python src/models/xgboost/train.py
python src/models/xgboost/evaluate.py

# U-Net (when implemented)
python src/models/unet/train.py
python src/models/unet/evaluate.py

# Comparison table
python src/evaluation/comparison.py
```

### Dashboard
```bash
streamlit run app.py
# Open http://localhost:8501
```

## Repository Structure
```
coastalcred/
├── app.py                        ← Streamlit dashboard (6 pages)
├── CLAUDE.md                     ← Project context & architecture decisions
├── README.md                     ← This file
├── requirements.txt
├── docker-compose.yml            ← PostgreSQL + PostGIS
├── src/
│   ├── data_pipeline/
│   │   ├── fetch_sentinel2.py    ← GEE → Sentinel-2 composites
│   │   ├── align_masks.py        ← GMW rasterization
│   │   ├── extract_patches.py    ← 256×256 patch cutting
│   │   ├── make_splits.py        ← Site-level train/val/test
│   │   └── download_data.py      ← Fetch from GitHub Releases
│   ├── models/
│   │   ├── ndvi/baseline.py      ← NDVI threshold model
│   │   └── xgboost/
│   │       ├── features.py       ← 10-feature extraction
│   │       ├── train.py          ← Memory-efficient training
│   │       └── evaluate.py       ← Test-set evaluation
│   ├── evaluation/
│   │   ├── metrics.py            ← Shared Precision/Recall/IoU/F1
│   │   └── comparison.py         ← 3-model comparison table
│   └── carbon/
│       ├── ipcc_tier1.py          ← Carbon stock & flux calculation
│       └── precompute_predictions.py ← Pre-compute all predictions
├── sql/schema.sql                ← PostgreSQL + PostGIS schema (~23 tables)
├── docs/
│   ├── coastalcred-prd.md        ← Product Requirements Document
│   └── handoff.md                ← Phase B+C handoff
├── results/                      ← Metric JSONs, plots, sample predictions
├── data/                         ← gitignored (via GitHub Releases)
└── models/                       ← gitignored (trained weights)
```

## Data Citations

- **Sentinel-2:** Copernicus Sentinel data [2020, 2024], processed by ESA. Dataset: `COPERNICUS/S2_SR_HARMONIZED`
- **Ground Truth:** Bunting et al. 2018, "The Global Mangrove Watch — A New 2010 Global Baseline of Mangrove Extent", Remote Sensing 10(10):1669. Dataset: Global Mangrove Watch v3.

---

## Viva Preparation — Comprehensive Q&A

### Q1: How does CoastalCred prevent fraud and ensure additionality?

**Answer:**
Additionality means carbon credits should only be issued for change *caused by human action*, not natural growth. CoastalCred addresses this through three evidence streams:

1. **Remote Sensing (this sprint):** Our ML models detect *change* in mangrove cover between 2020 and 2024. We measure flux (change), not stock (existing), specifically because credits represent new sequestration, not pre-existing carbon.

2. **Provenance Documentation (Sem VII):** A Community Portal with Aadhaar-linked identity verification ties each mangrove restoration project to a specific community or organization. This creates a chain of custody proving *who* caused the change.

3. **Field Verification (Sem VII):** A mobile app enables ground-truthing with GPS-tagged photos, species identification, and health assessments. This physical evidence prevents satellite-only fraud (e.g., a model misclassifying agricultural land as mangroves).

4. **Blockchain Immutability:** All three evidence streams are hashed and stored on Polygon (Amoy testnet). Once a verification is recorded, it cannot be altered. Smart contracts (ERC-721 TerritoryNFT, ERC-1155 CarbonCredit) enforce rules programmatically.

**Key insight:** The ML model alone doesn't issue credits — the *architecture around it* makes additionality claims defensible. A satellite can confirm change happened, but can't prove *why* it happened. That's why the full system (ML + identity + field verification + blockchain) exists.

### Q2: Why U-Net and not DeepLabV3+ or SegFormer?

**Answer:**
Three reasons:

1. **Hardware constraint:** Our training GPU is an RTX 3050 with 4 GB VRAM. U-Net with ResNet-18 encoder and 32 base filters fits comfortably. DeepLabV3+ with ResNet-50+ or SegFormer (transformer-based) would require 8-16 GB VRAM — impossible without Colab or cloud GPUs.

2. **Standard remote sensing baseline:** U-Net (Ronneberger et al., 2015) is the most widely cited architecture in remote sensing segmentation literature. It has abundant reference implementations, making our results directly comparable to published work. Reviewers and examiners are familiar with it.

3. **Sufficient for our task:** Binary segmentation of mangroves from 6-band Sentinel-2 imagery at 10m resolution doesn't require the complexity of ASPP (atrous spatial pyramid pooling in DeepLabV3+) or multi-head self-attention (SegFormer). U-Net's encoder-decoder with skip connections provides adequate spatial context for this task.

4. **Upgrade path:** If U-Net proves insufficient, DeepLabV3+ or SegFormer are Sem VII upgrades with the same data pipeline and evaluation code. The modular architecture supports this swap.

**Viva one-liner:** *"U-Net is the standard baseline in remote sensing, fits our 4GB VRAM, and provides sufficient spatial context. Heavier architectures are Sem VII upgrades."*

### Q3: Why XGBoost and not Random Forest?

**Answer:**

1. **Better class imbalance handling:** XGBoost supports `scale_pos_weight` natively, which adjusts the loss function gradient to account for rare positive samples. Random Forest handles imbalance through `class_weight='balanced'` or resampling, but XGBoost's gradient-based approach is more principled for extreme imbalance (~2% mangrove pixels globally).

2. **Typically higher IoU:** In benchmarks on tabular features with correlated inputs (like spectral bands), XGBoost usually achieves 1-2% better IoU than Random Forest. This is because XGBoost builds trees sequentially to correct previous errors (boosting), while Random Forest builds independent trees (bagging).

3. **Feature importance:** XGBoost provides gain-based feature importance that directly shows which spectral features drive decisions. Our results show NDVI dominates (0.456), followed by B2 Blue (0.214) and B11 SWIR (0.099). This is scientifically interpretable.

4. **Regularization:** XGBoost has built-in L1/L2 regularization (`reg_alpha`, `reg_lambda`), learning rate shrinkage, and early stopping — reducing overfitting risk compared to Random Forest.

5. **Speed:** XGBoost with `tree_method='hist'` uses histogram-based splitting, training in ~3 seconds on 200k samples vs. ~10+ seconds for comparable Random Forest.

**Our actual results:** XGBoost val IoU = 0.676, with NDVI as the dominant feature (0.456 importance). The model learned that NDVI is the strongest signal, confirming the vegetation index approach, while adding value from SWIR and Blue bands for water/soil discrimination.

### Q4: Why measure flux instead of stock?

**Answer:**
This is fundamental to how carbon credits work in real MRV systems (Verra VCS, Gold Standard):

1. **Credits = additional sequestration, not existing carbon.** An old-growth mangrove forest has enormous carbon stock, but it's already been there for decades. No one "caused" that carbon to be there. Credits reward *new* sequestration from restoration or protection.

2. **A single image can only measure stock.** One Sentinel-2 composite tells you where mangroves are *right now*. You need two time points to measure *change*:
   - 2020 baseline: X hectares of mangroves
   - 2024 current: Y hectares of mangroves
   - Flux = (Y − X) × sequestration rate × years

3. **IPCC methodology requires temporal comparison.** The IPCC Guidelines for National Greenhouse Gas Inventories (2006, 2013 Wetlands Supplement) define carbon accounting as change over time. We follow this standard:
   ```
   flux_tCO₂e = Δhectares × 7.0 tCO₂e/ha/year × 4 years
   ```

4. **Prevents gaming:** If credits were based on stock, anyone could claim credits for existing forests they didn't create. Flux-based accounting requires demonstrating that mangrove area *increased* during the monitoring period.

**Our implementation:** We predict mangrove masks for both 2020 and 2024 composites, compute area in hectares, and apply the IPCC flux formula. The 7.0 tCO₂e/ha/year rate is the IPCC default annual sequestration rate for mangrove ecosystems.

### Q5: How do you convert mangrove hectares to tCO₂e?

**Answer:**
Four-step IPCC Tier 1 chain, all constants from published sources:

**Step 1 — Pixels to hectares:**
```
Each Sentinel-2 pixel = 10m × 10m = 100 m² = 0.01 ha
mangrove_hectares = count(mangrove_pixels) × 0.01
```

**Step 2 — Hectares to biomass:**
```
biomass = hectares × 230 t/ha
```
The 230 t/ha is the IPCC default above-ground dry matter density for mangroves (IPCC 2013 Wetlands Supplement, Table 4.4).

**Step 3 — Biomass to carbon to CO₂:**
```
carbon = biomass × 0.47 (carbon fraction of dry biomass)
tCO₂e = carbon × (44/12) = carbon × 3.667
```
The 0.47 carbon fraction is the IPCC default. The 44/12 ratio converts carbon mass to CO₂ mass (molecular weight: CO₂ = 44, C = 12).

**Step 4 — Stock to flux (credits):**
```
Δhectares = current_ha − baseline_ha
annual_flux = Δhectares × 7.0 tCO₂e/ha/year
total_credits = annual_flux × 4 years (2020→2024)
```
The 7.0 tCO₂e/ha/year is the IPCC default net annual sequestration rate for mangroves.

**Example calculation:**
If model detects +100 hectares of new mangroves:
```
annual flux = 100 × 7.0 = 700 tCO₂e/year
4-year credits = 700 × 4 = 2,800 tCO₂e
```

**Important caveat:** These are Tier 1 estimates using global defaults. Tier 2 (country-specific) and Tier 3 (site-specific measurements) would be more accurate but require field data we don't have in this sprint.

### Q6: What's the class imbalance problem and how did you handle it?

**Answer:**
**The problem:** In our satellite imagery, mangrove pixels represent roughly 2% of the total landscape (globally across the full Sentinel-2 tiles). Even after patch extraction (which filters out many all-negative patches), positive pixels are about 40% within kept patches. A naive model predicting "not mangrove" everywhere achieves 60-98% accuracy depending on the subset — but is completely useless.

**Why accuracy is misleading:**
```
98% of pixels are non-mangrove
→ "Predict all zeros" model gets 98% accuracy
→ But 0% recall, 0% IoU — useless for mangrove detection
```

**How we handle it:**

1. **Metrics (all models):** We report **Precision, Recall, IoU (Jaccard), and F1** — never plain accuracy. IoU is the primary metric because it penalizes both false positives and false negatives equally:
   ```
   IoU = TP / (TP + FP + FN)
   ```

2. **NDVI baseline:** Threshold tuning on validation IoU (not accuracy). Best threshold 0.55 was selected by highest val IoU, not highest accuracy.

3. **XGBoost:** `scale_pos_weight = 1.47` (ratio of negative to positive samples in training data). This makes the loss function penalize misclassifying a mangrove pixel 1.47× more than misclassifying a non-mangrove pixel. Effectively, the model "cares more" about finding mangroves.

4. **U-Net (planned):** `BCEWithLogitsLoss(pos_weight=ratio)` — same principle as XGBoost, applied to the neural network's binary cross-entropy loss.

5. **Patch selection:** `extract_patches.py` keeps all patches containing mangroves but only 10% of purely negative patches. This enriches the training distribution without fully removing negatives (model still needs to learn what "not mangrove" looks like).

### Q7: Why three models and not just one?

**Answer:**
The three-model ladder serves three purposes:

**1. Scientific comparison:**
Each model represents a level of complexity:
| Model | Type | Context | Features |
|-------|------|---------|----------|
| NDVI | Rule-based | None | 1 (vegetation index) |
| XGBoost | Classical ML | None (pixel-level) | 10 (spectral + indices) |
| U-Net | Deep Learning | Spatial (256×256 patch) | 6 raw bands (learned features) |

This progression answers: *"Does adding complexity actually improve results?"*

**2. Ablation study:**
- NDVI → XGBoost: Does adding more spectral features help? (Yes on val: 0.572 → 0.676 IoU, but no on test: 0.336 → 0.332)
- XGBoost → U-Net: Does spatial context help? (Expected yes — pending results)
- The comparison table quantifies exactly how much value each architectural decision adds.

**3. Practical engineering baseline:**
- NDVI runs in <1 second, requires no training, no GPU, no ML library
- If the complex model fails or is too slow for production, NDVI is a reliable fallback
- This is standard practice in ML engineering: always compare against a simple baseline

**Our key finding:** XGBoost matches NDVI on the unseen test site (both ~0.33 IoU) despite much higher val performance (0.676 vs 0.572). This proves that **spectral features alone aren't enough for cross-site generalization** — spatial context (U-Net) is needed. This is the central insight of our Phase 2 comparison.

### Q8: Why is the test IoU so much lower than validation IoU?

**Answer:**
This is the **site-level generalization gap**, and it's intentional:

| Model | Val IoU | Test IoU | Gap |
|-------|---------|----------|-----|
| NDVI | 0.572 | 0.336 | 0.236 |
| XGBoost | 0.676 | 0.332 | 0.344 |

**Why it happens:**
- Val data = patches from Sundarbans and Gulf of Kutch (same sites as training)
- Test data = Pichavaram (completely unseen site, different coast, different ecosystem)
- Models learn site-specific patterns (water color, soil type, atmospheric conditions) that don't transfer

**Why XGBoost's gap is larger:**
XGBoost memorized the exact spectral fingerprints of Sundarbans + Kutch mangroves. These patterns are highly specific — Sundarbans has muddy deltaic water, Pichavaram has clear lagoon water. The spectral signatures differ enough that XGBoost's 10-feature classifier gets confused.

NDVI is more robust because it captures a **universal physical property** (chlorophyll reflects NIR, absorbs Red) that works regardless of site. This is physics-based generalization.

**Why this is a feature, not a bug:**
If we used random patch splitting, both models would show ~0.65+ IoU on "test" because patches from the same forest leak between train and test. Our site-level split **honestly measures real-world performance** — can this model work on a forest it's never seen?

**For the viva:** *"The generalization gap demonstrates why spatial context matters. Pixel-level methods overfit to site-specific spectral signatures, while physics-based indices (NDVI) generalize better. U-Net should close this gap by learning transferable spatial patterns."*

### Q9: How does the data pipeline work end-to-end?

**Answer:**
```
Google Earth Engine → fetch_sentinel2.py → 6-band GeoTIFF (10m resolution)
        ↓
Global Mangrove Watch v3 → align_masks.py → binary mask GeoTIFF (matched CRS/resolution)
        ↓
extract_patches.py → 256×256 .npy patches (image + mask pairs)
        ↓
make_splits.py → train.txt / val.txt / test.txt + norm_stats.json
        ↓
Model training → results/*.json
        ↓
comparison.py → comparison table + chart
        ↓
ipcc_tier1.py → carbon stock + flux report
```

**Key design decisions in the pipeline:**
1. **Sentinel-2 L2A (not L1C):** L2A is atmospherically corrected (surface reflectance), not top-of-atmosphere. This removes atmospheric effects that would vary between sites/dates.
2. **Median composite:** Instead of a single date, we composite the median pixel value across a year's worth of cloud-free observations. This reduces noise from individual scenes.
3. **6 bands (not all 13):** B2, B3, B4, B8 at 10m native resolution + B11, B12 resampled to 10m. We skip B1 (aerosol), B5-B7 (red edge, 20m), B8A (20m NIR), B9 (water vapor), B10 (cirrus).
4. **GMW v3 as labels:** Free, peer-reviewed, global coverage, 2020 epoch matches our baseline year.
5. **Normalization stats on train only:** Prevents information leakage from val/test into training.

### Q10: What are the spectral indices and why did you choose these four?

**Answer:**
We compute 4 vegetation/water indices from the 6 Sentinel-2 bands:

| Index | Formula | What it measures | Why included |
|-------|---------|-----------------|-------------|
| **NDVI** | (NIR−Red)/(NIR+Red) | Vegetation greenness/health | Primary mangrove indicator; our #1 feature (importance 0.456) |
| **EVI** | 2.5×(NIR−Red)/(NIR+6×Red−7.5×Blue+1) | Enhanced vegetation, corrects for soil/atmosphere | Better than NDVI in dense canopy (saturation correction) |
| **NDWI** | (Green−NIR)/(Green+NIR) | Water content / water body detection | Distinguishes mangroves from open water |
| **SAVI** | 1.5×(NIR−Red)/(NIR+Red+0.5) | Soil-adjusted vegetation | Reduces soil brightness effects in sparse canopy areas |

**Why these four and not others?**
- These are the standard indices used in mangrove remote sensing literature (Alongi 2014, Giri et al. 2011)
- Each captures a different biophysical property
- Together with 6 raw bands, they give XGBoost 10 interpretable features
- NDVI dominates (0.456 importance), confirming vegetation reflectance is the primary signal
- B2 (Blue) is second (0.214), useful for water depth and turbidity discrimination
- B11 (SWIR) is third (0.099), sensitive to moisture content — helps separate mangroves from dry vegetation

### Q11: What is the blockchain architecture and how does it relate to the ML component?

**Answer:**
The blockchain architecture (Sem VII implementation, this sprint is design only):

**Smart Contracts (Polygon Amoy testnet):**
- **TerritoryNFT (ERC-721):** Each verified mangrove restoration site becomes a unique NFT with GPS boundaries, community ownership, and verification history
- **CarbonCredit (ERC-1155):** Fungible tokens representing tCO₂e credits, minted when verification passes
- **Marketplace:** Peer-to-peer trading of carbon credits with transparent pricing

**How ML connects to blockchain:**
```
ML model predicts mangrove change (this sprint)
        ↓
Verification service validates change against 3 evidence streams
        ↓
If validated: smart contract mints CarbonCredit tokens (Sem VII)
        ↓
Tokens are tradeable on the marketplace
```

The ML model is the *first evidence stream* — it provides the satellite-based proof that mangrove area changed. The blockchain ensures this evidence is immutable and the resulting credits are unique (no double-counting).

**Why Polygon Amoy (not Mumbai)?** Mumbai testnet was deprecated in April 2024. Amoy is the current Polygon testnet. This is a correction from the original synopsis.

### Q12: What are the limitations of the current system?

**Answer:**
Be honest about limitations (examiners respect this):

1. **IPCC Tier 1 constants are global averages.** Biomass density (230 t/ha) and sequestration rate (7.0 tCO₂e/ha/yr) are defaults for all mangroves worldwide. Actual values vary by species, age, soil type, and climate. Tier 2/3 would use India-specific or site-specific measurements.

2. **GMW labels are from a single epoch (2020).** We use the same ground truth for both 2020 and 2024, meaning we can't validate temporal change in the labels — only in our predictions. True validation requires multi-temporal ground truth.

3. **Site-level generalization is poor.** Both NDVI (0.336) and XGBoost (0.332) show low test IoU on the unseen Pichavaram site. This means deploying to new sites requires either retraining or a model with better spatial generalization (U-Net).

4. **Overlap double-counting in carbon estimation.** Patches are extracted with 50% overlap (stride 128), so summing mangrove pixels across all patches inflates hectares by ~4x. The correct approach is stitching predictions back into a full raster. (Known bug, fix in progress.)

5. **No field validation.** GMW is our proxy for ground truth, but it's itself derived from satellite data. True MRV requires field visits with GPS measurements, species identification, and health assessments (Sem VII mobile app).

6. **Binary classification only.** We classify mangrove vs. non-mangrove. We don't distinguish mangrove species, canopy density, or health status — all of which affect actual carbon sequestration rates.

### Q13: How would you improve the system if you had more time?

**Answer:**

1. **Multi-temporal analysis:** Instead of just 2020 and 2024, use annual composites (2020-2024) to detect gradual change and distinguish seasonal variation from real change.

2. **Ensemble prediction:** Combine NDVI + XGBoost + U-Net predictions via majority voting or weighted average to reduce individual model errors.

3. **Transfer learning from larger datasets:** Pre-train U-Net on the global SEN12MS dataset (Schmitt et al. 2019), then fine-tune on Indian mangroves.

4. **Tier 2/3 carbon constants:** Replace IPCC defaults with India-specific biomass density values from published forestry surveys.

5. **Change detection models:** Instead of predicting masks independently per year, use Siamese networks or temporal attention to directly detect change between image pairs.

6. **Active learning:** Use model uncertainty to select the most informative patches for human labeling, improving the training set efficiently.

7. **Edge deployment:** Optimize U-Net with ONNX or TensorRT for faster inference, enabling near-real-time monitoring.

---

## License

This project is developed for academic purposes as part of the Sem VI Mini Project at RCOEM, Nagpur.
