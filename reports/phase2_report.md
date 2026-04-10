# CoastalCred Phase 2 Report: Data Pipeline & Machine Learning for Mangrove Classification

---

**Project:** CoastalCred — Blockchain-Based Blue Carbon Registry & MRV System

**Institution:** Shri Ramdeobaba College of Engineering and Management (RCOEM), Nagpur
Department of Data Science

**Semester:** VI (2025--2026)

**Guide:** Dr. Aarti Karandikar

**Industry Mentor:** Rishikesh Kale (Filecoin / Protocol Labs)

**Team Members:**

| Name | Role |
|------|------|
| Ansh Chopada (Lead) | Architecture + Documentation Lead |
| Aniruddha Lahoti | Data Pipeline + Infra Lead |
| TBD | Deep Learning Lead |
| TBD | Classical ML + Evaluation + Carbon Lead |

**Date:** April 2026

---

## Abstract

This report presents Phase 2 of CoastalCred, a blockchain-based blue carbon registry and MRV (Monitoring, Reporting, and Verification) system for Indian mangrove ecosystems. Phase 2 delivers an end-to-end machine learning pipeline for mangrove cover classification using Sentinel-2 satellite imagery across three ecologically distinct coastal sites: Sundarbans (West Bengal), Gulf of Kutch (Gujarat), and Pichavaram (Tamil Nadu). We constructed a data pipeline that ingests 6-band multispectral composites from Google Earth Engine, aligns ground truth labels from Global Mangrove Watch v3, extracts 256x256 patches, and implements a site-level train/test split to honestly measure cross-site generalization. Three models of increasing complexity were trained and compared: an NDVI threshold baseline (rule-based), an XGBoost pixel classifier (classical ML with 10 spectral features), and a U-Net deep learning segmentation model (ResNet-18 encoder). On the unseen Pichavaram test site, the models achieved IoU scores of 0.336 (NDVI), 0.332 (XGBoost), and 0.268 (U-Net), revealing a substantial generalization gap from validation performance (0.572, 0.676, and 0.768 respectively). Carbon credit estimates were computed using IPCC Tier 1 methodology, converting predicted mangrove area change (2020 to 2024) into tonnes of CO2 equivalent. This work establishes the remote sensing foundation for the full CoastalCred platform to be completed in Semester VII.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Study Area](#2-study-area)
3. [Data Sources and Acquisition](#3-data-sources-and-acquisition)
4. [Data Pipeline](#4-data-pipeline)
5. [Methodology](#5-methodology)
   - 5.1 [Evaluation Metrics](#51-evaluation-metrics)
   - 5.2 [NDVI Threshold Baseline](#52-ndvi-threshold-baseline)
   - 5.3 [XGBoost Pixel Classifier](#53-xgboost-pixel-classifier)
   - 5.4 [U-Net Semantic Segmentation](#54-u-net-semantic-segmentation)
6. [Results and Discussion](#6-results-and-discussion)
   - 6.1 [Model Comparison Table](#61-model-comparison-table)
   - 6.2 [Per-Site Breakdown](#62-per-site-breakdown)
   - 6.3 [Feature Importance Analysis](#63-feature-importance-analysis-xgboost)
   - 6.4 [U-Net Prediction Visualizations](#64-u-net-prediction-visualizations)
   - 6.5 [Generalization Gap Analysis](#65-generalization-gap-analysis)
7. [Carbon Credit Estimation](#7-carbon-credit-estimation)
   - 7.1 [IPCC Tier 1 Methodology](#71-ipcc-tier-1-methodology)
   - 7.2 [Per-Site Carbon Flux Results](#72-per-site-carbon-flux-results)
   - 7.3 [Cross-Model Carbon Agreement](#73-cross-model-carbon-agreement)
8. [Limitations](#8-limitations)
9. [Conclusion](#9-conclusion)
10. [Future Work](#10-future-work)
11. [References](#11-references)

---

## 1. Introduction

Mangrove ecosystems are among the most carbon-dense habitats on Earth, storing up to four times more carbon per unit area than terrestrial forests (Donato et al., 2011). Despite covering less than 1% of tropical forest area, mangroves provide critical ecosystem services including coastal protection, biodiversity habitat, and blue carbon sequestration. India's mangrove cover spans approximately 4,975 km^2 across its coastline, with significant concentrations in the Sundarbans, Gujarat, and Tamil Nadu (Forest Survey of India, 2021).

Accurate and scalable monitoring of mangrove cover change is essential for carbon credit verification under frameworks such as Verra VCS and Gold Standard. Traditional field surveys are costly, time-consuming, and impractical for the spatial scales required by national and international carbon markets. Satellite-based remote sensing, particularly using freely available Sentinel-2 multispectral imagery at 10-meter resolution, offers a viable alternative for continuous monitoring.

This report presents Phase 2 of CoastalCred, which focuses on building the machine learning component of a blockchain-based MRV system. The specific objectives of Phase 2 are:

1. **Data pipeline construction:** Ingest Sentinel-2 L2A surface reflectance composites and Global Mangrove Watch v3 ground truth labels for three Indian mangrove sites across two time points (2020 and 2024).
2. **Model development and comparison:** Train and evaluate three models of increasing complexity -- NDVI threshold (rule-based), XGBoost (classical ML), and U-Net (deep learning) -- using identical data splits and evaluation code.
3. **Cross-site generalization assessment:** Use site-level train/test splitting to honestly measure whether models trained on Sundarbans and Gulf of Kutch can generalize to the unseen Pichavaram site.
4. **Carbon credit estimation:** Apply IPCC Tier 1 methodology to convert predicted mangrove area change into tonnes of CO2 equivalent (tCO2e).

The three-model comparison ladder serves as both a scientific ablation study (does complexity improve results?) and an engineering baseline (what is the simplest model that works?).

---

## 2. Study Area

Three ecologically distinct mangrove sites along India's coastline were selected to maximize ecosystem diversity and test cross-site generalization.

### 2.1 Sundarbans, West Bengal

| Parameter | Value |
|-----------|-------|
| Approximate centre | 21.95 N, 88.90 E |
| Ecosystem type | Gangetic delta, tidal mangrove forest |
| Mangrove area | Largest contiguous mangrove in the world (~4,260 km^2 across India and Bangladesh) |
| Dominant species | *Heritiera fomes*, *Excoecaria agallocha*, *Avicennia* spp. |
| Role in study | Training |

The Sundarbans is characterized by dense, mature mangrove canopy with high biomass density, muddy deltaic water, and strong tidal influence. Its large spatial extent (5,815 patches at 256x256) provides the bulk of training data.

### 2.2 Gulf of Kutch, Gujarat

| Parameter | Value |
|-----------|-------|
| Approximate centre | 22.85 N, 69.60 E |
| Ecosystem type | Arid coastal mangrove, high salinity |
| Mangrove area | ~914 km^2 |
| Dominant species | *Avicennia marina* (monospecific stands) |
| Role in study | Training |

Gulf of Kutch mangroves grow in an arid, hyper-saline environment with lower canopy density and different spectral signatures compared to the humid Sundarbans. Including this site forces models to learn mangrove features robust to environmental variation.

### 2.3 Pichavaram, Tamil Nadu

| Parameter | Value |
|-----------|-------|
| Approximate centre | 11.42 N, 79.78 E |
| Ecosystem type | Backwater estuary, brackish lagoon |
| Mangrove area | ~11 km^2 |
| Dominant species | *Rhizophora* spp., *Avicennia marina* |
| Role in study | Test (unseen) |

Pichavaram was deliberately chosen as the test site because it is (a) geographically distant from the training sites, (b) ecologically distinct (backwater estuary versus delta or arid coast), and (c) small enough (71 patches) to represent a realistic monitoring deployment scenario.

### Site Selection Rationale

These three sites were chosen to span the major mangrove ecosystem types found along India's coastline:
- **Deltaic** (Sundarbans) -- high rainfall, freshwater influx, dense mixed-species canopy
- **Arid coastal** (Gulf of Kutch) -- low rainfall, high salinity, sparse monospecific stands
- **Estuarine** (Pichavaram) -- moderate conditions, backwater mixing, small-scale ecosystem

This diversity tests whether models learn generalizable mangrove features or overfit to site-specific spectral signatures.

---

## 3. Data Sources and Acquisition

### 3.1 Satellite Imagery: Sentinel-2 L2A

**Source:** European Space Agency (ESA) Copernicus Sentinel-2 L2A surface reflectance, accessed via Google Earth Engine (dataset: `COPERNICUS/S2_SR_HARMONIZED`).

**Temporal coverage:** Annual median composites for 2020 (baseline) and 2024 (current), yielding 6 composites (3 sites x 2 years).

**Spectral bands selected:**

| Band | Name | Wavelength (nm) | Native Resolution | Purpose |
|------|------|-----------------|-------------------|---------|
| B2 | Blue | 490 | 10 m | Water penetration, turbidity, soil discrimination |
| B3 | Green | 560 | 10 m | Vegetation vigor, NDWI computation |
| B4 | Red | 665 | 10 m | Chlorophyll absorption, NDVI computation |
| B8 | NIR | 842 | 10 m | Vegetation reflectance, NDVI/EVI computation |
| B11 | SWIR-1 | 1610 | 20 m (resampled to 10 m) | Moisture content, canopy water |
| B12 | SWIR-2 | 2190 | 20 m (resampled to 10 m) | Mineral/soil discrimination |

**Band selection rationale:** Six bands were chosen from the full Sentinel-2 13-band suite. B2, B3, B4, B8 are available at native 10 m resolution. B11 and B12 (20 m native) were resampled to 10 m during GEE export and provide critical moisture and soil information for distinguishing mangroves from other vegetation. Bands B1 (aerosol, 60 m), B5-B7 (red edge, 20 m), B8A (narrow NIR, 20 m), B9 (water vapour), and B10 (cirrus) were excluded due to coarser resolution or limited discriminative value for this task.

**Pre-processing in GEE:**
- Cloud masking using the `QA60` bitmask (opaque and cirrus cloud flags)
- Annual median composite (reduces noise from individual scenes, fills gaps from cloud masking)
- Scale factor: raw values divided by 10,000 to convert to surface reflectance [0, 1]
- Export as GeoTIFF, CRS EPSG:4326

**Composite sizes:**

| Composite | File Size | Approx. Scenes in Composite |
|-----------|-----------|------------------------------|
| sundarbans_2024.tif | 1.1 GB | ~280 |
| sundarbans_2020.tif | 958 MB | -- |
| gulf_of_kutch_2024.tif | 472 MB | -- |
| gulf_of_kutch_2020.tif | 458 MB | -- |
| pichavaram_2024.tif | 27 MB | -- |
| pichavaram_2020.tif | 26 MB | -- |

### 3.2 Ground Truth: Global Mangrove Watch v3

**Source:** Global Mangrove Watch (GMW) v3 (2020 epoch), hosted by UN Environment Programme World Conservation Monitoring Centre (UNEP-WCMC).

**Format:** Vector polygons (GeoPackage) delineating mangrove extent globally, derived from a combination of ALOS PALSAR radar and Landsat optical imagery with machine learning classification.

**Processing:** GMW polygons were clipped to each site's bounding box and rasterized to match the corresponding Sentinel-2 composite's CRS, resolution (10 m), and spatial extent using `rasterio` and `geopandas`.

**Limitation:** GMW v3 provides a single 2020 epoch. The same labels are used for both the 2020 and 2024 composites, which means temporal change in ground truth labels cannot be validated -- only temporal change in model predictions can be assessed.

**Citation:** Bunting et al. (2018). "The Global Mangrove Watch -- A New 2010 Global Baseline of Mangrove Extent." *Remote Sensing*, 10(10), 1669.

---

## 4. Data Pipeline

The data pipeline transforms raw satellite imagery and vector labels into model-ready training data through four stages.

### 4.1 Pipeline Architecture

```
Google Earth Engine
    |
    v
fetch_sentinel2.py --> 6-band GeoTIFF per site per year (10 m resolution)
    |
    v
Global Mangrove Watch v3 --> align_masks.py --> binary mask GeoTIFF (matched CRS/resolution)
    |
    v
extract_patches.py --> 256 x 256 .npy patches (image + mask pairs)
    |
    v
make_splits.py --> train.txt / val.txt / test.txt + norm_stats.json
    |
    v
Model training --> results/*.json
    |
    v
comparison.py --> comparison table + charts
    |
    v
ipcc_tier1.py --> carbon stock + flux report
```

### 4.2 Patch Extraction

- **Patch size:** 256 x 256 pixels (2.56 km x 2.56 km at 10 m resolution)
- **Stride:** 128 pixels (50% overlap) -- increases training samples and provides augmentation through shifted context
- **Format:** NumPy `.npy` arrays, `float32`, values scaled to [0, 1]
- **Patch filtering:** All patches containing at least one mangrove pixel are retained. Of purely negative (no-mangrove) patches, only 10% are kept to enrich the training distribution without completely removing negative examples.

**Patch counts per site:**

| Site | Total Patches (per year) | Positive (contains mangrove) | Negative Kept |
|------|--------------------------|------------------------------|---------------|
| Sundarbans | 5,815 | 5,283 | 532 |
| Gulf of Kutch | 1,267 | 813 | 454 |
| Pichavaram | 71 | 59 | 12 |

### 4.3 Site-Level Train/Val/Test Split

A site-level splitting strategy was adopted to prevent data leakage. In random patch splitting, adjacent patches from the same forest would appear in both train and test sets, sharing spectral patterns and inflating test metrics. Site-level splitting ensures the test site (Pichavaram) has never been seen during training.

| Split | Patches | Source Sites | Purpose |
|-------|---------|--------------|---------|
| Train | 6,374 | Sundarbans 2024 + Gulf of Kutch 2024 | Model training |
| Val | 708 | 10% random holdout from train pool (seed=42) | Hyperparameter tuning, early stopping |
| Test | 71 | Pichavaram 2024 (completely unseen site) | Generalization measurement |

**Note:** 2020 patches for all sites exist but are excluded from the train/val/test splits. They are reserved exclusively for carbon flux calculation (comparing 2020 predictions against 2024 predictions).

### 4.4 Normalization Statistics

Per-band mean and standard deviation were computed on the training set only to prevent information leakage from validation and test data.

| Band | Mean | Std |
|------|------|-----|
| B2 (Blue) | 0.0518 | 0.0378 |
| B3 (Green) | 0.0686 | 0.0448 |
| B4 (Red) | 0.0605 | 0.0467 |
| B8 (NIR) | 0.1400 | 0.1125 |
| B11 (SWIR-1) | 0.0818 | 0.0813 |
| B12 (SWIR-2) | 0.0509 | 0.0594 |

These statistics are used by the U-Net model for input normalization. NDVI and XGBoost operate on raw [0, 1] reflectance values.

---

## 5. Methodology

### 5.1 Evaluation Metrics

All three models are evaluated using a shared evaluation module (`src/evaluation/metrics.py`) to ensure apples-to-apples comparison.

**Why not accuracy?** Mangrove pixels constitute approximately 2% of the total landscape globally. A naive model predicting "not mangrove" for every pixel would achieve 98% accuracy while being completely useless for mangrove detection. We therefore report four metrics robust to class imbalance:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Precision | TP / (TP + FP) | Of predicted mangrove pixels, what fraction is correct? |
| Recall | TP / (TP + FN) | Of actual mangrove pixels, what fraction was detected? |
| IoU (Jaccard) | TP / (TP + FP + FN) | Overlap between predicted and actual mangrove area |
| F1 Score | 2 * Precision * Recall / (Precision + Recall) | Harmonic mean of precision and recall |

**Primary metric:** IoU (Intersection over Union) is the primary headline metric because it penalizes both false positives and false negatives equally and is the standard in semantic segmentation tasks.

### 5.2 NDVI Threshold Baseline

**Algorithm:** The Normalized Difference Vegetation Index (NDVI) is a physics-based vegetation index that exploits the spectral signature of chlorophyll -- high reflectance in NIR, strong absorption in Red:

```
NDVI = (NIR - Red) / (NIR + Red + epsilon)
     = (B8 - B4) / (B8 + B4 + 1e-8)
```

Pixels with NDVI above a threshold are classified as mangrove.

**Threshold search:** Thresholds from 0.10 to 0.60 (step 0.05) were evaluated on the validation set by IoU:

| Threshold | Val IoU |
|-----------|---------|
| 0.10 | 0.465 |
| 0.15 | 0.470 |
| 0.20 | 0.478 |
| 0.25 | 0.488 |
| 0.30 | 0.500 |
| 0.35 | 0.515 |
| 0.40 | 0.533 |
| 0.45 | 0.551 |
| 0.50 | 0.566 |
| **0.55** | **0.572** |
| 0.60 | 0.566 |

**Best threshold:** 0.55 (highest validation IoU = 0.572)

**Strengths:** Zero training time, physics-based (captures universal chlorophyll property), no hyperparameters beyond threshold, interpretable.

**Weaknesses:** Single-feature decision boundary, no spectral context (ignores SWIR, Blue), no spatial context (pixel-independent classification).

### 5.3 XGBoost Pixel Classifier

**Algorithm:** Gradient boosted decision trees (XGBoost) trained on per-pixel spectral features.

**Feature engineering:** 10 features per pixel, combining 6 raw spectral bands with 4 derived vegetation/water indices:

| Feature | Formula / Source | Physical Meaning |
|---------|------------------|------------------|
| B2 | Raw band | Water penetration, turbidity |
| B3 | Raw band | Vegetation vigor |
| B4 | Raw band | Chlorophyll absorption |
| B8 | Raw band | Vegetation NIR reflectance |
| B11 | Raw band | Canopy moisture (SWIR-1) |
| B12 | Raw band | Soil/mineral discrimination (SWIR-2) |
| NDVI | (B8 - B4) / (B8 + B4 + 1e-8) | Vegetation greenness |
| EVI | 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1) | Enhanced vegetation (soil/atmosphere corrected) |
| NDWI | (B3 - B8) / (B3 + B8 + 1e-8) | Water content / water bodies |
| SAVI | 1.5 * (B8 - B4) / (B8 + B4 + 0.5) | Soil-adjusted vegetation |

**Training configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 300 (287 used) | Early stopping before full 300 |
| max_depth | 6 | Moderate depth prevents overfitting |
| learning_rate | 0.1 | Standard default |
| scale_pos_weight | 1.47 | Ratio of negative to positive samples; compensates class imbalance |
| tree_method | hist | Histogram-based splitting for speed |
| early_stopping_rounds | 20 | Stop if val logloss does not improve for 20 rounds |
| Training sample | ~200k pixels | Subsampled with class balance from training patches |

**Training time:** 2.79 seconds on CPU.

**Strengths:** Multi-feature decision boundary, feature importance interpretability, fast training, handles class imbalance via `scale_pos_weight`.

**Weaknesses:** No spatial context (each pixel classified independently), prone to overfitting site-specific spectral signatures.

### 5.4 U-Net Semantic Segmentation

**Architecture:** U-Net (Ronneberger et al., 2015) implemented via the `segmentation_models_pytorch` library with a ResNet-18 encoder pre-trained on ImageNet.

| Parameter | Value |
|-----------|-------|
| Encoder | ResNet-18 (ImageNet pre-trained) |
| Input channels | 6 (B2, B3, B4, B8, B11, B12) |
| Output channels | 1 (binary mangrove mask) |
| Loss function | BCEWithLogitsLoss with pos_weight |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| Batch size | 8 |
| Epochs | 50 (best at epoch 50) |
| Mixed precision | torch.cuda.amp (required for 4 GB VRAM) |
| Input normalization | Per-band z-score using training set statistics |
| GPU | NVIDIA RTX 3050 (4 GB VRAM) |

**Training time:** 7,298.7 seconds (~121.6 minutes / ~2 hours).

**Key design decisions:**
- **ResNet-18 over deeper encoders:** Fits within 4 GB VRAM constraint. ResNet-50+ or transformer encoders would require 8--16 GB.
- **ImageNet pre-training:** Even though ImageNet contains natural photographs (not satellite imagery), transfer learning from ImageNet provides useful low-level feature extractors (edges, textures) that improve convergence.
- **6-channel input:** The first convolutional layer is modified to accept 6 channels instead of the standard 3 RGB channels. ImageNet weights for the first layer are adapted by averaging the RGB weights for the additional channels.
- **Mixed precision (AMP):** Reduces memory footprint by approximately 50%, enabling batch size 8 on 4 GB VRAM.

**Strengths:** Captures spatial context through convolutional receptive fields (256x256 patch neighbourhood), learns features end-to-end, encoder-decoder with skip connections preserves fine-grained spatial details.

**Weaknesses:** Longer training time, requires GPU, more complex hyperparameter space, risk of overfitting on small datasets.

---

## 6. Results and Discussion

### 6.1 Model Comparison Table

The following table presents evaluation metrics for all three models on both validation (Sundarbans + Gulf of Kutch holdout) and test (Pichavaram, unseen) sets.

**Validation Set Performance (Sundarbans + Gulf of Kutch holdout):**

| Metric | NDVI Threshold | XGBoost | U-Net |
|--------|---------------|---------|-------|
| Precision | 0.866 | 0.693 | 0.778 |
| Recall | 0.627 | 0.966 | 0.984 |
| **IoU** | **0.572** | **0.676** | **0.768** |
| F1 | 0.728 | 0.807 | 0.869 |

**Test Set Performance (Pichavaram -- unseen site):**

| Metric | NDVI Threshold | XGBoost | U-Net |
|--------|---------------|---------|-------|
| Precision | 0.428 | 0.438 | 0.908 |
| Recall | 0.610 | 0.580 | 0.276 |
| **IoU** | **0.336** | **0.332** | **0.268** |
| F1 | 0.503 | 0.499 | 0.423 |
| Training Time | 0 sec | 2.79 sec | 7,298.7 sec (~2 hr) |

**Key observations:**

1. **U-Net achieves the highest validation IoU (0.768)** but the lowest test IoU (0.268), exhibiting the largest generalization gap of all three models.
2. **NDVI threshold achieves the highest test IoU (0.336)** despite being the simplest model, demonstrating the robustness of physics-based approaches for cross-site generalization.
3. **XGBoost provides the best precision-recall balance on validation** (0.807 F1) but overfits to training site spectral signatures, performing comparably to NDVI on the unseen test site.
4. **U-Net on the test site shows extremely high precision (0.908) but very low recall (0.276)**, meaning it is highly conservative -- when it predicts mangrove, it is almost always correct, but it misses approximately 72% of actual mangrove pixels at Pichavaram.

### 6.2 Per-Site Breakdown

| Model | Site | Precision | Recall | IoU | F1 |
|-------|------|-----------|--------|-----|-----|
| NDVI Threshold | Sundarbans + Gulf of Kutch (val) | 0.866 | 0.627 | 0.572 | 0.728 |
| NDVI Threshold | Pichavaram (test) | 0.428 | 0.610 | 0.336 | 0.503 |
| XGBoost | Sundarbans + Gulf of Kutch (val) | 0.693 | 0.966 | 0.676 | 0.807 |
| XGBoost | Pichavaram (test) | 0.438 | 0.580 | 0.332 | 0.499 |
| U-Net | Sundarbans + Gulf of Kutch (val) | 0.778 | 0.984 | 0.768 | 0.869 |
| U-Net | Pichavaram (test) | 0.908 | 0.276 | 0.268 | 0.423 |

**Generalization gap summary:**

| Model | Val IoU | Test IoU | Gap | Gap (%) |
|-------|---------|----------|-----|---------|
| NDVI Threshold | 0.572 | 0.336 | 0.236 | 41.3% |
| XGBoost | 0.676 | 0.332 | 0.344 | 50.9% |
| U-Net | 0.768 | 0.268 | 0.500 | 65.1% |

### 6.3 Feature Importance Analysis (XGBoost)

XGBoost provides gain-based feature importance, revealing which spectral features drive classification decisions.

| Rank | Feature | Importance (Gain) | Interpretation |
|------|---------|-------------------|----------------|
| 1 | NDVI | 0.456 | Dominant signal: chlorophyll-based vegetation detection |
| 2 | B2 (Blue) | 0.214 | Water depth, turbidity, coastal discrimination |
| 3 | B11 (SWIR-1) | 0.099 | Canopy moisture content |
| 4 | NDWI | 0.066 | Water body vs. vegetation discrimination |
| 5 | B8 (NIR) | 0.047 | Raw NIR vegetation reflectance |
| 6 | B12 (SWIR-2) | 0.042 | Soil/mineral discrimination |
| 7 | B4 (Red) | 0.023 | Chlorophyll absorption (redundant with NDVI) |
| 8 | B3 (Green) | 0.020 | Vegetation vigor |
| 9 | EVI | 0.018 | Largely redundant with NDVI for this task |
| 10 | SAVI | 0.017 | Largely redundant with NDVI for this task |

**Analysis:**
- **NDVI dominates** with 45.6% of total feature importance, confirming that the vegetation index is the primary signal for mangrove detection. This validates the NDVI threshold baseline as a reasonable approach.
- **B2 (Blue) is second** at 21.4%, likely because it captures water turbidity and depth variations that help discriminate between mangrove-adjacent water bodies and actual vegetation.
- **B11 (SWIR-1) is third** at 9.9%, providing moisture content information that separates mangroves (high moisture) from dry vegetation.
- **EVI and SAVI contribute minimally** (<2% each), suggesting they are largely redundant with NDVI for this classification task.

The feature importance results are saved as `results/feature_importance.png`.

### 6.4 U-Net Prediction Visualizations

Five sample prediction visualizations are saved in the results directory, each showing a triptych of RGB composite, ground truth mask, and predicted mask:

- `results/unet_pred_0.png`
- `results/unet_pred_1.png`
- `results/unet_pred_2.png`
- `results/unet_pred_3.png`
- `results/unet_pred_4.png`

These visualizations provide qualitative assessment of U-Net's performance, illustrating:
- Where the model correctly segments mangrove extent
- False positive regions (non-mangrove predicted as mangrove)
- False negative regions (mangrove missed by the model)
- Edge delineation quality compared to ground truth boundaries

### 6.5 Generalization Gap Analysis

The most significant finding of Phase 2 is the substantial generalization gap between validation and test performance across all three models.

**Why does the gap exist?**

1. **Site-specific spectral signatures:** The Sundarbans (muddy deltaic water, dense canopy) and Gulf of Kutch (arid, hyper-saline, sparse canopy) have distinct spectral profiles from Pichavaram (brackish lagoon, estuarine conditions). Models learn these site-specific patterns during training.

2. **Environmental variation:** Water colour, soil type, atmospheric conditions, and vegetation density differ substantially between the west coast (Gujarat), the eastern delta (Sundarbans), and the south-eastern coast (Pichavaram).

3. **Scale mismatch:** Pichavaram is much smaller (71 patches vs. 5,815 for Sundarbans), meaning the test set represents a concentrated spatial area with less internal diversity.

**Model-specific observations:**

- **NDVI (gap = 0.236):** Smallest gap because the NDVI ratio captures a universal physical property -- chlorophyll reflects NIR and absorbs Red regardless of location. However, the optimal threshold (0.55) may be suboptimal for Pichavaram's specific vegetation mix.

- **XGBoost (gap = 0.344):** Larger gap because the 10-feature model memorized the exact spectral fingerprints of Sundarbans + Kutch mangroves. The high val recall (0.966) drops to 0.580 on the test set, indicating the model learned decision boundaries too specific to the training sites.

- **U-Net (gap = 0.500):** Largest gap despite being the most complex model. The extremely high test precision (0.908) but very low recall (0.276) suggests U-Net learned to be very conservative on unfamiliar data -- it only predicts mangrove when spatial patterns strongly match what it saw during training. When the spatial texture of Pichavaram mangroves differs from Sundarbans/Kutch patterns, the model defaults to predicting "not mangrove."

**Conclusion from the gap:** Adding model complexity (NDVI -> XGBoost -> U-Net) consistently improves validation performance but does not proportionally improve test performance. Spatial context (U-Net) helps on familiar sites but may overfit to site-specific spatial textures. Cross-site generalization requires either (a) training on more diverse sites, (b) domain adaptation techniques, or (c) multi-temporal analysis.

---

## 7. Carbon Credit Estimation

### 7.1 IPCC Tier 1 Methodology

Carbon credits are computed using the IPCC Tier 1 methodology from the *2013 Supplement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories: Wetlands* (Hiraishi et al., 2013).

**Constants used (all from IPCC published defaults):**

| Constant | Value | Source |
|----------|-------|--------|
| Biomass density | 230 t/ha (dry matter) | IPCC 2013 Wetlands Supplement, Table 4.4 |
| Carbon fraction | 0.47 | IPCC default for mangrove biomass |
| CO2:C molecular ratio | 44/12 = 3.667 | Molecular weight ratio |
| Annual sequestration rate | 7.0 tCO2e/ha/year | IPCC default for mangroves |

**Step 1 -- Pixels to hectares:**
```
Each Sentinel-2 pixel = 10 m x 10 m = 100 m^2 = 0.01 ha
mangrove_hectares = count(mangrove_pixels) x 0.01
```

**Step 2 -- Hectares to biomass:**
```
biomass_t = hectares x 230 t/ha
```

**Step 3 -- Biomass to CO2 equivalent (stock):**
```
carbon_t = biomass_t x 0.47
co2e_t = carbon_t x 3.667

Equivalently:
stock_tCO2e = hectares x 230 x 0.47 x 3.667
            = hectares x 396.49 tCO2e/ha
```

**Step 4 -- Stock to flux (carbon credits):**
```
delta_hectares = current_hectares_2024 - baseline_hectares_2020
annual_flux_tCO2e = delta_hectares x 7.0 tCO2e/ha/year
total_flux_tCO2e = annual_flux_tCO2e x 4 years
```

Positive flux indicates mangrove area gain (carbon credits earned). Negative flux indicates mangrove area loss (carbon debt).

### 7.2 Per-Site Carbon Flux Results

Carbon flux was computed for each model by running predictions on both the 2020 and 2024 composites for all three sites and applying the IPCC Tier 1 formula.

#### 7.2.1 Sundarbans

| Metric | NDVI | XGBoost | U-Net |
|--------|------|---------|-------|
| Predicted ha (2020) | 1,222,571.91 | 2,457,279.86 | 725,736.30 |
| Predicted ha (2024) | 1,357,643.66 | 2,337,287.92 | 656,833.74 |
| Delta ha | +135,071.75 | -119,991.94 | -68,902.56 |
| Annual flux (tCO2e/yr) | +945,502.25 | -839,943.58 | -482,317.92 |
| **4-year flux (tCO2e)** | **+3,782,009.00** | **-3,359,774.32** | **-1,929,271.68** |

#### 7.2.2 Gulf of Kutch

| Metric | NDVI | XGBoost | U-Net |
|--------|------|---------|-------|
| Predicted ha (2020) | 33,745.40 | 415,415.23 | 278,263.30 |
| Predicted ha (2024) | 32,530.47 | 377,767.94 | 251,946.28 |
| Delta ha | -1,214.93 | -37,647.29 | -26,317.02 |
| Annual flux (tCO2e/yr) | -8,504.51 | -263,531.03 | -184,219.14 |
| **4-year flux (tCO2e)** | **-34,018.04** | **-1,054,124.12** | **-736,876.56** |

#### 7.2.3 Pichavaram

| Metric | NDVI | XGBoost | U-Net |
|--------|------|---------|-------|
| Predicted ha (2020) | 6,619.43 | 10,271.05 | 654.95 |
| Predicted ha (2024) | 8,265.54 | 7,689.83 | 489.72 |
| Delta ha | +1,646.11 | -2,581.22 | -165.23 |
| Annual flux (tCO2e/yr) | +11,522.77 | -18,068.54 | -1,156.61 |
| **4-year flux (tCO2e)** | **+46,091.08** | **-72,274.16** | **-4,626.44** |

#### 7.2.4 Ground Truth Reference (GMW v3)

For comparison, the GMW ground truth labels (available only for 2020) provide reference hectares:

| Site | GMW Ground Truth ha (2020) |
|------|---------------------------|
| Sundarbans | 1,795,698.80 |
| Gulf of Kutch | 91,940.90 |
| Pichavaram | 5,805.10 |

### 7.3 Cross-Model Carbon Agreement

The three models show significant disagreement in carbon flux predictions, both in magnitude and direction:

**Direction of change (sign of flux):**

| Site | NDVI | XGBoost | U-Net |
|------|------|---------|-------|
| Sundarbans | Gain (+) | Loss (-) | Loss (-) |
| Gulf of Kutch | Loss (-) | Loss (-) | Loss (-) |
| Pichavaram | Gain (+) | Loss (-) | Loss (-) |

**Analysis:**

1. **Models disagree on the direction of change** for Sundarbans and Pichavaram. NDVI predicts mangrove area gain while XGBoost and U-Net predict loss. This fundamental disagreement means carbon credit estimates from any single model are unreliable without validation.

2. **Magnitude varies by orders of magnitude.** XGBoost predicts 4.5x more Sundarbans mangrove area than NDVI (2,457,279 ha vs. 1,222,571 ha in 2020), reflecting its tendency to over-predict (high recall, low precision). U-Net predicts much less area for Pichavaram (654 ha vs. GMW ground truth of 5,805 ha), consistent with its conservative test-set behaviour.

3. **Overlap-based inflation.** Patch extraction with 50% overlap (stride 128) means mangrove pixels near patch boundaries are counted in multiple overlapping patches. Summing predicted pixels across all patches inflates total hectares. The correct approach requires stitching predictions back into a single full-resolution raster before computing area. This is a known limitation and is flagged for correction.

4. **Key takeaway for MRV:** No single model's carbon estimate should be trusted in isolation. The disagreement reinforces the need for (a) ensemble methods that combine multiple model predictions, (b) field validation to calibrate remote sensing estimates, and (c) conservative crediting that accounts for model uncertainty.

---

## 8. Limitations

1. **GMW proxy labels from a single epoch (2020).** The same ground truth masks are used for both 2020 and 2024 composites, preventing validation of temporal change in the labels. True temporal validation would require multi-epoch ground truth or field surveys.

2. **Single unseen test site.** Generalization is measured on Pichavaram alone (71 patches). A more robust evaluation would use multiple held-out sites spanning different ecosystem types and geographic regions.

3. **Severe class imbalance.** Even after patch filtering, mangrove pixels are a minority class. While we mitigate this through loss weighting (`scale_pos_weight`, `pos_weight`) and IoU-based evaluation, the imbalance still biases models toward the dominant non-mangrove class, especially on unfamiliar sites.

4. **No temporal analysis.** Models predict each time point independently. We do not use temporal features (multi-date composites, change detection, Siamese networks) that could directly learn patterns of change rather than inferring them from independent snapshots.

5. **Overlap double-counting in carbon estimation.** Patches are extracted with 50% overlap, so summing predicted mangrove pixels across all patches inflates the total area by approximately 4x. A proper full-raster stitching approach is needed for accurate carbon quantification.

6. **IPCC Tier 1 constants are global averages.** The biomass density (230 t/ha) and sequestration rate (7.0 tCO2e/ha/yr) are IPCC default values for all mangroves worldwide. Actual values vary significantly by species, age, soil type, and climate. India-specific Tier 2 constants would improve accuracy.

7. **Binary classification only.** We classify mangrove vs. non-mangrove without distinguishing species, canopy density, age, or health status -- all of which affect actual carbon sequestration rates.

8. **No field validation.** GMW is itself derived from satellite data (ALOS PALSAR + Landsat). True MRV requires field visits with GPS measurements, species identification, and health assessments.

9. **Site-level generalization gap.** All three models show substantial performance drops on the unseen Pichavaram site (test IoU 0.268--0.336 vs. val IoU 0.572--0.768), indicating that current models are not ready for deployment to arbitrary new sites without retraining or adaptation.

---

## 9. Conclusion

Phase 2 of CoastalCred successfully delivered an end-to-end machine learning pipeline for mangrove classification from Sentinel-2 satellite imagery, along with IPCC Tier 1 carbon credit estimation. The key findings and deliverables are:

### Key Findings

1. **The complexity ladder yields diminishing returns on generalization.** While U-Net (IoU 0.768) substantially outperforms XGBoost (0.676) and NDVI (0.572) on the validation set, the ranking reverses on the unseen test site: NDVI (0.336) > XGBoost (0.332) > U-Net (0.268). More complex models overfit more severely to training-site spectral and spatial patterns.

2. **NDVI is the most robust baseline for cross-site deployment.** Its physics-based approach (chlorophyll reflects NIR, absorbs Red) captures a universal vegetation property that transfers across ecosystems, even though the absolute performance ceiling is lower.

3. **XGBoost confirms NDVI as the dominant feature.** With 45.6% of total feature importance attributed to NDVI, XGBoost validates that the vegetation index is the primary signal. The additional spectral features (Blue, SWIR) provide marginal improvement on familiar sites but do not help generalization.

4. **U-Net is highly conservative on unseen sites.** Its test precision of 0.908 (when it predicts mangrove, it is almost always correct) but recall of 0.276 (it misses 72% of actual mangroves) reveals a model that defaults to "not mangrove" when spatial patterns are unfamiliar.

5. **Carbon estimates from different models disagree in both magnitude and direction**, highlighting the need for ensemble methods, field validation, and conservative crediting approaches.

### Phase 2 Deliverables Summary

| Deliverable | Status |
|-------------|--------|
| Data pipeline (GEE fetch, alignment, patches, splits) | Complete |
| NDVI threshold baseline | Complete (best threshold = 0.55, test IoU = 0.336) |
| XGBoost pixel classifier | Complete (287 trees, test IoU = 0.332) |
| U-Net segmentation model | Complete (ResNet-18, 50 epochs, test IoU = 0.268) |
| 3-model comparison table | Complete (val + test, per-site breakdown) |
| Feature importance analysis | Complete (NDVI dominant at 0.456) |
| Carbon credit estimation (IPCC Tier 1) | Complete (per-site flux for all 3 models) |
| Streamlit dashboard | Complete (6 pages: overview, models, carbon, comparison) |
| Result JSONs and visualizations | Complete (committed to results/) |

---

## 10. Future Work

### Semester VII Extensions

1. **U-Net improvements:**
   - Train on more diverse sites to reduce the generalization gap
   - Experiment with data augmentation (random flips, rotations, colour jitter) specific to remote sensing
   - Try DeepLabV3+ or SegFormer if higher-VRAM hardware becomes available
   - Pre-train on SEN12MS (Schmitt et al., 2019) or similar global datasets before fine-tuning on Indian mangroves

2. **Multi-temporal analysis:**
   - Use annual composites (2020--2024) instead of just two time points
   - Implement Siamese networks or temporal attention for direct change detection between image pairs
   - Distinguish seasonal variation from real mangrove cover change

3. **Ensemble methods:**
   - Combine NDVI + XGBoost + U-Net predictions via majority voting or weighted average
   - Use model disagreement as an uncertainty signal for active learning

4. **Field validation (mobile app):**
   - GPS-tagged ground truth collection at all three sites
   - Species identification and canopy health assessment
   - Replace GMW proxy labels with field-verified labels for validation

5. **Tier 2/3 carbon constants:**
   - Replace IPCC global defaults with India-specific biomass density values from published forestry surveys
   - Incorporate species-specific sequestration rates

6. **Full-raster stitching:**
   - Implement proper prediction stitching to eliminate overlap double-counting in carbon estimation
   - Weighted blending at patch boundaries for smoother predictions

7. **Blockchain integration:**
   - Deploy smart contracts (TerritoryNFT, CarbonCredit, Marketplace) on Polygon Amoy testnet
   - Integrate ML predictions with the verification service
   - Mint carbon credit tokens based on validated flux estimates

8. **Edge deployment:**
   - Optimize U-Net with ONNX or TensorRT for faster inference
   - Enable near-real-time monitoring for the MRV pipeline

---

## 11. References

1. Bunting, P., Rosenqvist, A., Lucas, R. M., Rebelo, L.-M., Hilarides, L., Thomas, N., Hardy, A., Itoh, T., Shimada, M., and Finlayson, C. M. (2018). "The Global Mangrove Watch -- A New 2010 Global Baseline of Mangrove Extent." *Remote Sensing*, 10(10), 1669. https://doi.org/10.3390/rs10101669

2. Chen, T. and Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785--794. https://doi.org/10.1145/2939672.2939785

3. Copernicus Sentinel data [2020, 2024], processed by ESA. Dataset: `COPERNICUS/S2_SR_HARMONIZED`. https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED

4. Donato, D. C., Kauffman, J. B., Murdiyarso, D., Kurnianto, S., Stidham, M., and Kanninen, M. (2011). "Mangroves among the most carbon-rich forests in the tropics." *Nature Geoscience*, 4(5), 293--297. https://doi.org/10.1038/ngeo1123

5. Forest Survey of India (2021). *India State of Forest Report 2021*. Ministry of Environment, Forest and Climate Change, Government of India.

6. Giri, C., Ochieng, E., Tieszen, L. L., Zhu, Z., Singh, A., Loveland, T., Masek, J., and Duke, N. (2011). "Status and distribution of mangrove forests of the world using earth observation satellite data." *Global Ecology and Biogeography*, 20(1), 154--159. https://doi.org/10.1111/j.1466-8238.2010.00584.x

7. Hiraishi, T., Krug, T., Tanabe, K., Srivastava, N., Baasansuren, J., Fukuda, M., and Troxler, T. G. (eds.) (2013). *2013 Supplement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories: Wetlands*. IPCC, Switzerland.

8. IPCC (2006). *2006 IPCC Guidelines for National Greenhouse Gas Inventories*. Eggleston, H. S., Buendia, L., Miwa, K., Ngara, T., and Tanabe, K. (eds.). Published: IGES, Japan.

9. Ronneberger, O., Fischer, P., and Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Springer, LNCS, Vol. 9351, 234--241. https://doi.org/10.1007/978-3-319-24574-4_28

10. Schmitt, M., Hughes, L. H., Qiu, C., and Zhu, X. X. (2019). "SEN12MS -- A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion." *ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, IV-2/W7, 153--160. https://doi.org/10.5194/isprs-annals-IV-2-W7-153-2019

11. Yakubov, S., et al. (2022). `segmentation_models_pytorch` -- Python library for image segmentation with pre-trained encoders. https://github.com/qubvel/segmentation_models.pytorch

---

*Report generated as part of the CoastalCred Sem VI Mini Project at RCOEM, Nagpur.*
