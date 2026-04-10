# CoastalCred -- Viva Preparation Document

> Sem VI Mini Project | RCOEM Dept. of Data Science, Nagpur
> Guide: Dr. Aarti Karandikar | Industry Mentor: Rishikesh Kale (Filecoin/Protocol Labs)
> Team: Ansh Chopada (Lead), Aniruddha Lahoti, TBD, TBD

---

## How to Use This Document

This document covers the **13 questions** the team must be prepared to answer confidently during the viva examination. Questions 1--7 are the **core mandatory questions** identified during project planning. Questions 8--13 are additional questions that examiners commonly ask about systems of this nature.

Each answer includes:
- A concise explanation (2--4 paragraphs)
- Key numbers and facts to memorize (marked with bullet points)
- A **one-liner** suitable for opening your answer before elaborating

Read through this document at least twice before the viva. Practice saying the one-liners aloud -- they set the tone for a confident answer.

---

## Core Questions (1--7)

---

### Q1: How does CoastalCred prevent fraud and ensure additionality?

> **One-liner:** *"Additionality requires proving that carbon credits represent new sequestration caused by human action, not pre-existing natural carbon. CoastalCred enforces this through three independent evidence streams plus blockchain immutability."*

**Answer:**

Additionality means carbon credits should only be issued for change *caused by human action*, not natural growth. A forest that has existed for decades has enormous carbon stock, but no one "caused" that carbon to be there. Real MRV systems (Verra VCS, Gold Standard) require demonstrating that the credited sequestration would not have occurred without the project intervention. CoastalCred addresses this through three evidence streams:

1. **Remote Sensing (this sprint):** Our ML models detect *change* in mangrove cover between 2020 and 2024. We measure flux (change over time), not stock (existing carbon), specifically because credits represent new sequestration.

2. **Provenance Documentation (Sem VII):** A Community Portal with Aadhaar-linked identity verification ties each mangrove restoration project to a specific community or organization. This creates a chain of custody proving *who* caused the change.

3. **Field Verification (Sem VII):** A mobile app enables ground-truthing with GPS-tagged photos, species identification, and health assessments. This physical evidence prevents satellite-only fraud (e.g., a model misclassifying agricultural land as mangroves).

4. **Blockchain Immutability:** All three evidence streams are hashed and stored on Polygon (Amoy testnet). Once a verification is recorded, it cannot be altered. Smart contracts (ERC-721 TerritoryNFT, ERC-1155 CarbonCredit) enforce rules programmatically -- no human can bypass the verification logic.

**Key insight to emphasize:** The ML model alone does not issue credits. The *architecture around it* (ML + identity + field verification + blockchain) is what makes additionality claims defensible. A satellite can confirm change happened, but cannot prove *why* it happened. That is why the full system exists.

---

### Q2: Why U-Net and not DeepLabV3+ or SegFormer?

> **One-liner:** *"U-Net is the standard baseline in remote sensing, fits our 4GB VRAM, and provides sufficient spatial context. Heavier architectures are Sem VII upgrades."*

**Answer:**

Three primary reasons drove this decision:

1. **Hardware constraint:** Our training GPU is an RTX 3050 with 4 GB VRAM. U-Net with ResNet-18 encoder and 32 base filters fits comfortably. DeepLabV3+ with ResNet-50+ requires 8+ GB VRAM. SegFormer (transformer-based) requires 12--16 GB. Neither is feasible without cloud GPUs.

2. **Standard remote sensing baseline:** U-Net (Ronneberger et al., 2015) is the most widely cited architecture in remote sensing segmentation literature. It has abundant reference implementations, making our results directly comparable to published work. Reviewers and examiners are familiar with it.

3. **Sufficient for our task:** Binary segmentation of mangroves from 6-band Sentinel-2 imagery at 10m resolution does not require the complexity of ASPP (atrous spatial pyramid pooling in DeepLabV3+) or multi-head self-attention (SegFormer). U-Net's encoder-decoder with skip connections provides adequate spatial context for this task.

4. **Upgrade path:** If U-Net proves insufficient, DeepLabV3+ or SegFormer are Sem VII upgrades. The modular architecture (same data pipeline, same evaluation code) supports swapping the model without changing anything else.

**Numbers to remember:**
- U-Net VRAM: ~2--3 GB with batch size 8 at 256x256
- RTX 3050 VRAM: 4 GB
- DeepLabV3+ minimum VRAM: ~8 GB
- SegFormer minimum VRAM: ~12 GB
- U-Net reference: Ronneberger et al., 2015

---

### Q3: Why XGBoost and not Random Forest?

> **One-liner:** *"XGBoost handles class imbalance better via scale_pos_weight, typically achieves 1--2% higher IoU on correlated spectral features, and provides interpretable feature importances."*

**Answer:**

1. **Better class imbalance handling:** XGBoost supports `scale_pos_weight` natively, which adjusts the loss function gradient to account for rare positive samples. Random Forest handles imbalance through `class_weight='balanced'` or resampling, but XGBoost's gradient-based approach is more principled for extreme imbalance (~2% mangrove pixels globally, ~40% within kept patches).

2. **Typically higher IoU:** In benchmarks on tabular features with correlated inputs (like spectral bands), XGBoost usually achieves 1--2% better IoU than Random Forest. This is because XGBoost builds trees sequentially to correct previous errors (boosting), while Random Forest builds independent trees (bagging).

3. **Feature importance:** XGBoost provides gain-based feature importance that directly shows which spectral features drive decisions. Our results confirm: NDVI dominates (0.456 importance), followed by B2 Blue (0.214) and B11 SWIR (0.099). This is scientifically interpretable and makes for a strong report figure.

4. **Regularization:** XGBoost has built-in L1/L2 regularization (`reg_alpha`, `reg_lambda`), learning rate shrinkage, and early stopping -- reducing overfitting risk compared to Random Forest.

5. **Speed:** XGBoost with `tree_method='hist'` uses histogram-based splitting, training in ~2.79 seconds on 200k samples vs. ~10+ seconds for comparable Random Forest.

**Numbers to remember:**
- XGBoost val IoU: 0.676
- Top 3 features: NDVI (0.456), B2 Blue (0.214), B11 SWIR (0.099)
- Training time: 2.79 seconds
- Hyperparameters: 300 trees (287 used), max_depth=6, lr=0.1, scale_pos_weight=1.47

---

### Q4: Why measure flux instead of stock?

> **One-liner:** *"We measure change in mangrove cover between a 2020 baseline and 2024 current state, then compute additional sequestration following IPCC flux methodology."*

**Answer:**

This is fundamental to how carbon credits work in real MRV systems (Verra VCS, Gold Standard):

1. **Credits = additional sequestration, not existing carbon.** An old-growth mangrove forest has enormous carbon stock, but it has been there for decades. No one "caused" that carbon to be there. Credits reward *new* sequestration from restoration or protection activities.

2. **A single image can only measure stock.** One Sentinel-2 composite tells you where mangroves are *right now*. You need two time points to measure *change*:
   - 2020 baseline: X hectares of mangroves
   - 2024 current: Y hectares of mangroves
   - Flux = (Y - X) x sequestration rate x years

3. **IPCC methodology requires temporal comparison.** The IPCC Guidelines for National Greenhouse Gas Inventories (2006, 2013 Wetlands Supplement) define carbon accounting as change over time. Our formula: `flux_tCO2e = delta_hectares x 7.0 tCO2e/ha/year x 4 years`.

4. **Prevents gaming:** If credits were based on stock, anyone could claim credits for existing forests they did not create. Flux-based accounting requires demonstrating that mangrove area *increased* during the monitoring period.

**Our implementation:** We predict mangrove masks for both 2020 and 2024 composites, compute area in hectares, and apply the IPCC flux formula. Positive flux = mangrove area grew = carbon credits earned. Negative flux = mangrove area lost = carbon debt.

---

### Q5: How do you convert mangrove hectares to tCO2e?

> **One-liner:** *"We follow the IPCC Tier 1 four-step chain: pixels to hectares, hectares to biomass, biomass to carbon to CO2, then stock to flux credits -- all using published IPCC constants."*

**Answer:**

**Step 1 -- Pixels to hectares:**
```
Each Sentinel-2 pixel = 10m x 10m = 100 m2 = 0.01 ha
mangrove_hectares = count(mangrove_pixels) x 0.01
```

**Step 2 -- Hectares to biomass:**
```
biomass = hectares x 230 t/ha
```
The 230 t/ha is the IPCC default above-ground dry matter density for mangroves (IPCC 2013 Wetlands Supplement, Table 4.4).

**Step 3 -- Biomass to carbon to CO2:**
```
carbon = biomass x 0.47 (carbon fraction of dry biomass)
tCO2e = carbon x (44/12) = carbon x 3.667
```
The 0.47 carbon fraction is the IPCC default. The 44/12 ratio converts carbon mass to CO2 mass (molecular weight: CO2 = 44, C = 12).

**Step 4 -- Stock to flux (credits):**
```
delta_hectares = current_ha - baseline_ha
annual_flux = delta_hectares x 7.0 tCO2e/ha/year
total_credits = annual_flux x 4 years (2020 to 2024)
```
The 7.0 tCO2e/ha/year is the IPCC default net annual sequestration rate for mangroves.

**Worked example:** If the model detects +100 hectares of new mangroves:
```
annual flux = 100 x 7.0 = 700 tCO2e/year
4-year credits = 700 x 4 = 2,800 tCO2e
```

**Important caveat to mention if asked:** These are Tier 1 estimates using global defaults. Tier 2 (country-specific) and Tier 3 (site-specific measurements) would be more accurate but require field data we do not have in this sprint.

**Constants to memorize:**
- Biomass density: 230 t/ha
- Carbon fraction: 0.47
- CO2:C ratio: 44/12 = 3.667
- Annual sequestration: 7.0 tCO2e/ha/year
- Pixel area: 0.01 ha (10m x 10m)

---

### Q6: What is the class imbalance problem and how did you handle it?

> **One-liner:** *"Mangroves represent roughly 2% of pixels globally in our satellite tiles. A naive all-zero model gets 98% accuracy but 0% recall. We use IoU as the primary metric and weighted loss functions to address this."*

**Answer:**

**The problem:** In our satellite imagery, mangrove pixels represent roughly 2% of the total landscape across the full Sentinel-2 tiles. Even after patch extraction (which filters out many all-negative patches), positive pixels are about 40% within kept patches. A naive model predicting "not mangrove" everywhere achieves 60--98% accuracy depending on the subset -- but is completely useless for mangrove detection.

**Why accuracy is misleading:**
```
98% of pixels are non-mangrove
"Predict all zeros" model gets 98% accuracy
But: 0% recall, 0% IoU -- completely useless
```

**How we handle it across all three models:**

1. **Metrics (all models):** We report Precision, Recall, IoU (Jaccard), and F1 -- never plain accuracy. IoU is the primary metric because it penalizes both false positives and false negatives: `IoU = TP / (TP + FP + FN)`.

2. **NDVI baseline:** Threshold tuning on validation IoU (not accuracy). Best threshold 0.55 was selected by highest val IoU.

3. **XGBoost:** `scale_pos_weight = 1.47` (ratio of negative to positive samples in training data). This makes the loss function penalize misclassifying a mangrove pixel 1.47x more than misclassifying a non-mangrove pixel.

4. **U-Net (planned):** `BCEWithLogitsLoss(pos_weight=ratio)` -- same principle as XGBoost, applied to the neural network's binary cross-entropy loss.

5. **Patch selection:** `extract_patches.py` keeps all patches containing mangroves but only 10% of purely negative patches. This enriches the training distribution without fully removing negatives (the model still needs to learn what "not mangrove" looks like).

---

### Q7: Why three models and not just one?

> **One-liner:** *"The three-model ladder -- rules, classical ML, deep learning -- serves as a scientific ablation study, quantifying exactly how much value each level of complexity adds."*

**Answer:**

The three-model ladder serves three purposes:

**1. Scientific comparison -- complexity vs. performance:**

| Model | Type | Context | Features |
|-------|------|---------|----------|
| NDVI | Rule-based | None | 1 (vegetation index) |
| XGBoost | Classical ML | None (pixel-level) | 10 (spectral + indices) |
| U-Net | Deep Learning | Spatial (256x256 patch) | 6 raw bands (learned) |

This progression answers: *"Does adding complexity actually improve results?"*

**2. Ablation study with quantified results:**
- NDVI to XGBoost: Does adding more spectral features help? Yes on val (0.572 to 0.676 IoU), but no on test (0.336 to 0.332 IoU).
- XGBoost to U-Net: Does spatial context help? (Expected yes -- pending results.)
- The comparison table quantifies exactly how much value each architectural decision adds.

**3. Practical engineering baseline:**
- NDVI runs in less than 1 second, requires no training, no GPU, no ML library.
- If the complex model fails or is too slow for production, NDVI is a reliable fallback.
- This is standard practice in ML engineering: always compare against a simple baseline.

**Key finding to emphasize:** XGBoost matches NDVI on the unseen test site (both ~0.33 IoU) despite much higher val performance (0.676 vs. 0.572). This proves that spectral features alone are not enough for cross-site generalization -- spatial context (U-Net) is needed. This is the central insight of our Phase 2 comparison.

---

## Additional Questions (8--13)

---

### Q8: Why is the test IoU so much lower than validation IoU?

> **One-liner:** *"This is the site-level generalization gap -- intentional by design. Validation data comes from the same sites as training, but the test site (Pichavaram) is completely unseen."*

**Answer:**

| Model | Val IoU | Test IoU | Gap |
|-------|---------|----------|-----|
| NDVI | 0.572 | 0.336 | 0.236 |
| XGBoost | 0.676 | 0.332 | 0.344 |

**Why it happens:**
- Validation data = patches from Sundarbans and Gulf of Kutch (same sites as training).
- Test data = Pichavaram (completely unseen site, different coast, different ecosystem type).
- Models learn site-specific patterns (water color, soil type, atmospheric conditions) that do not transfer.

**Why XGBoost's gap is larger:** XGBoost memorized the exact spectral fingerprints of Sundarbans + Kutch mangroves. These patterns are highly site-specific -- Sundarbans has muddy deltaic water, Pichavaram has clear lagoon water. The spectral signatures differ enough that XGBoost's 10-feature classifier gets confused.

**Why NDVI generalizes better:** NDVI captures a universal physical property -- chlorophyll reflects NIR and absorbs Red. This works regardless of site. This is physics-based generalization.

**Why this is a feature, not a bug:** If we used random patch splitting, both models would show ~0.65+ IoU on "test" because patches from the same forest would leak between train and test. Our site-level split honestly measures real-world performance: can this model work on a forest it has never seen?

---

### Q9: How does the data pipeline work end-to-end?

> **One-liner:** *"Six-stage pipeline: GEE fetch, mask alignment, patch extraction, site-level splitting, model training, then carbon estimation -- each stage feeds the next."*

**Answer:**

```
Google Earth Engine --> fetch_sentinel2.py --> 6-band GeoTIFF (10m resolution)
        |
Global Mangrove Watch v3 --> align_masks.py --> binary mask GeoTIFF (matched CRS/resolution)
        |
extract_patches.py --> 256x256 .npy patches (image + mask pairs)
        |
make_splits.py --> train.txt / val.txt / test.txt + norm_stats.json
        |
Model training --> results/*.json
        |
comparison.py --> comparison table + chart
        |
ipcc_tier1.py --> carbon stock + flux report
```

**Key design decisions in the pipeline:**

1. **Sentinel-2 L2A (not L1C):** L2A is atmospherically corrected (surface reflectance), not top-of-atmosphere. This removes atmospheric effects that vary between sites and dates.

2. **Median composite:** Instead of a single date, we composite the median pixel value across a year's worth of cloud-free observations. This reduces noise from individual scenes.

3. **6 bands (not all 13):** B2, B3, B4, B8 at 10m native resolution + B11, B12 resampled to 10m. We skip B1 (aerosol), B5--B7 (red edge, 20m), B8A (20m NIR), B9 (water vapor), B10 (cirrus).

4. **GMW v3 as labels:** Free, peer-reviewed, global coverage, 2020 epoch matches our baseline year.

5. **Normalization stats on train only:** Prevents information leakage from val/test into training.

---

### Q10: What are the spectral indices and why did you choose these four?

> **One-liner:** *"NDVI, EVI, NDWI, and SAVI -- each captures a different biophysical property. Together with 6 raw bands, they give XGBoost 10 interpretable features."*

**Answer:**

| Index | Formula | What It Measures | Why Included |
|-------|---------|-----------------|-------------|
| **NDVI** | (NIR-Red)/(NIR+Red) | Vegetation greenness/health | Primary mangrove indicator; #1 feature (importance 0.456) |
| **EVI** | 2.5x(NIR-Red)/(NIR+6xRed-7.5xBlue+1) | Enhanced vegetation, corrects for soil/atmosphere | Better than NDVI in dense canopy (saturation correction) |
| **NDWI** | (Green-NIR)/(Green+NIR) | Water content / water body detection | Distinguishes mangroves from open water |
| **SAVI** | 1.5x(NIR-Red)/(NIR+Red+0.5) | Soil-adjusted vegetation | Reduces soil brightness effects in sparse canopy areas |

**Why these four and not others:**
- These are the standard indices used in mangrove remote sensing literature (Alongi 2014, Giri et al. 2011).
- Each captures a different biophysical property (greenness, canopy density, water, soil correction).
- NDVI dominates feature importance (0.456), confirming vegetation reflectance is the primary signal.
- B2 (Blue) is second (0.214), useful for water depth and turbidity discrimination.
- B11 (SWIR) is third (0.099), sensitive to moisture content -- helps separate mangroves from dry vegetation.

---

### Q11: What is the blockchain architecture and how does it relate to the ML component?

> **One-liner:** *"The ML model provides the first evidence stream -- satellite-based proof of mangrove change. The blockchain ensures this evidence is immutable and the resulting carbon credits cannot be double-counted."*

**Answer:**

**Smart Contracts (Polygon Amoy testnet -- design only this sprint):**
- **TerritoryNFT (ERC-721):** Each verified mangrove restoration site becomes a unique NFT with GPS boundaries, community ownership, and verification history.
- **CarbonCredit (ERC-1155):** Fungible tokens representing tCO2e credits, minted when verification passes all three evidence streams.
- **Marketplace:** Peer-to-peer trading of carbon credits with transparent pricing.

**How ML connects to blockchain:**
```
ML model predicts mangrove change (this sprint)
        |
Verification service validates change against 3 evidence streams
        |
If validated: smart contract mints CarbonCredit tokens (Sem VII)
        |
Tokens are tradeable on the marketplace
```

The ML model is the *first evidence stream* -- it provides the satellite-based proof that mangrove area changed. The blockchain ensures this evidence is immutable and the resulting credits are unique (no double-counting).

**Why Polygon Amoy (not Mumbai)?** The original synopsis references "Polygon Mumbai" but Mumbai testnet was deprecated in April 2024. Amoy is the current Polygon testnet. This is a correction from the original synopsis -- flag this proactively to the examiner if asked about the testnet.

---

### Q12: What are the limitations of the current system?

> **One-liner:** *"We are transparent about six key limitations -- IPCC Tier 1 global averages, single-epoch ground truth, poor cross-site generalization, patch overlap, no field validation, and binary-only classification."*

**Answer:**

Be honest about limitations during the viva -- examiners respect intellectual honesty:

1. **IPCC Tier 1 constants are global averages.** Biomass density (230 t/ha) and sequestration rate (7.0 tCO2e/ha/yr) are defaults for all mangroves worldwide. Actual values vary by species, age, soil type, and climate. Tier 2/3 would use India-specific or site-specific measurements.

2. **GMW labels are from a single epoch (2020).** We use the same ground truth for both 2020 and 2024, meaning we cannot validate temporal change in the labels -- only in our predictions. True validation requires multi-temporal ground truth.

3. **Site-level generalization is poor.** Both NDVI (0.336) and XGBoost (0.332) show low test IoU on the unseen Pichavaram site. Deploying to new sites requires either retraining or a model with better spatial generalization (U-Net).

4. **Overlap double-counting in carbon estimation.** Patches are extracted with 50% overlap (stride 128), so summing mangrove pixels across all patches inflates hectares. The correct approach is stitching predictions back into a full raster. (Known issue.)

5. **No field validation.** GMW is our proxy for ground truth, but it is itself derived from satellite data. True MRV requires field visits with GPS measurements, species identification, and health assessments (Sem VII mobile app).

6. **Binary classification only.** We classify mangrove vs. non-mangrove. We do not distinguish mangrove species, canopy density, or health status -- all of which affect actual carbon sequestration rates.

---

### Q13: How would you improve the system if you had more time?

> **One-liner:** *"Seven concrete improvements spanning temporal resolution, ensemble methods, transfer learning, localized carbon constants, change detection, active learning, and edge deployment."*

**Answer:**

1. **Multi-temporal analysis:** Instead of just 2020 and 2024, use annual composites (2020--2024) to detect gradual change and distinguish seasonal variation from real change.

2. **Ensemble prediction:** Combine NDVI + XGBoost + U-Net predictions via majority voting or weighted average to reduce individual model errors.

3. **Transfer learning from larger datasets:** Pre-train U-Net on the global SEN12MS dataset (Schmitt et al. 2019), then fine-tune on Indian mangroves.

4. **Tier 2/3 carbon constants:** Replace IPCC defaults with India-specific biomass density values from published forestry surveys (e.g., Forest Survey of India reports).

5. **Change detection models:** Instead of predicting masks independently per year, use Siamese networks or temporal attention to directly detect change between image pairs.

6. **Active learning:** Use model uncertainty to select the most informative patches for human labeling, improving the training set efficiently.

7. **Edge deployment:** Optimize U-Net with ONNX or TensorRT for faster inference, enabling near-real-time monitoring.

---

## Quick Reference

This section contains the key numbers, tables, and facts the team should have at their fingertips during the viva.

### IPCC Tier 1 Constants

| Constant | Value | Source |
|----------|-------|--------|
| Biomass density | 230 t/ha (dry matter) | IPCC 2013 Wetlands Supplement, Table 4.4 |
| Carbon fraction | 0.47 | IPCC default |
| CO2:C molecular ratio | 44/12 = 3.667 | Molecular weight |
| Annual sequestration | 7.0 tCO2e/ha/year | IPCC default for mangroves |
| Sentinel-2 pixel area | 0.01 ha (10m x 10m) | Sentinel-2 specification |

**Stock formula:** `stock_tCO2e = hectares x 230 x 0.47 x 3.667`

**Flux formula:** `flux_tCO2e = (current_ha - baseline_ha) x 7.0 x years`

### Model Comparison Table

| Metric | NDVI Threshold | XGBoost | U-Net (pending) |
|--------|---------------|---------|-----------------|
| Val Precision | 0.866 | 0.693 | -- |
| Val Recall | 0.627 | 0.966 | -- |
| Val IoU | 0.572 | 0.676 | -- |
| Val F1 | 0.728 | 0.807 | -- |
| Test Precision | 0.428 | 0.438 | -- |
| Test Recall | 0.610 | 0.580 | -- |
| **Test IoU** | **0.336** | **0.332** | -- |
| Test F1 | 0.503 | 0.499 | -- |
| Training Time | 0 sec | 2.79 sec | ~60--90 min |

**Primary metric:** IoU (Intersection over Union) -- robust under class imbalance.

### Generalization Gap

| Model | Val IoU | Test IoU | Gap |
|-------|---------|----------|-----|
| NDVI | 0.572 | 0.336 | 0.236 |
| XGBoost | 0.676 | 0.332 | 0.344 |

XGBoost has a larger gap because it memorized site-specific spectral signatures. NDVI generalizes better due to physics-based reasoning.

### Study Sites

| Site | State | Ecosystem | Role |
|------|-------|-----------|------|
| Sundarbans | West Bengal | Gangetic delta | Training |
| Gulf of Kutch | Gujarat | Arid coastal | Training |
| Pichavaram | Tamil Nadu | Backwater estuary | Test (unseen) |

**Years:** 2020 (baseline) and 2024 (current) -- 4-year flux measurement.

### Data Sources

| Source | Dataset | Citation |
|--------|---------|----------|
| Satellite imagery | Sentinel-2 L2A via Google Earth Engine | Copernicus Sentinel data [2020, 2024], processed by ESA. Dataset: `COPERNICUS/S2_SR_HARMONIZED` |
| Ground truth labels | Global Mangrove Watch v3 | Bunting et al. 2018, "The Global Mangrove Watch -- A New 2010 Global Baseline of Mangrove Extent", Remote Sensing 10(10):1669 |

### Sentinel-2 Bands Used

| Band | Wavelength | Use |
|------|------------|-----|
| B2 (Blue) | 490 nm | Water penetration, soil |
| B3 (Green) | 560 nm | Vegetation vigor, NDWI |
| B4 (Red) | 665 nm | Chlorophyll absorption, NDVI |
| B8 (NIR) | 842 nm | Vegetation reflectance, NDVI |
| B11 (SWIR-1) | 1610 nm | Moisture content |
| B12 (SWIR-2) | 2190 nm | Mineral/soil discrimination |

### XGBoost Feature Importance (Top 5)

| Feature | Importance |
|---------|-----------|
| NDVI | 0.456 |
| B2 (Blue) | 0.214 |
| B11 (SWIR-1) | 0.099 |
| Other 7 features | remaining ~0.231 |

### XGBoost Hyperparameters

| Parameter | Value |
|-----------|-------|
| Number of trees | 300 (287 used, early stopping) |
| Max depth | 6 |
| Learning rate | 0.1 |
| scale_pos_weight | 1.47 |
| Tree method | hist |

### Data Split

| Split | Patches | Source | Purpose |
|-------|---------|--------|---------|
| Train | 6,374 | Sundarbans + Gulf of Kutch (2024) | Model training |
| Val | 708 | 10% holdout from train (seed=42) | Hyperparameter tuning |
| Test | 71 | Pichavaram 2024 (unseen site) | Generalization measurement |

### Key Terminology

| Term | Definition |
|------|-----------|
| **MRV** | Monitoring, Reporting, and Verification -- the process of measuring and validating carbon credits |
| **Additionality** | Proof that credited sequestration would not have occurred without project intervention |
| **Flux** | Change in carbon over time (as opposed to stock, which is a snapshot) |
| **IoU** | Intersection over Union = TP / (TP + FP + FN) -- primary evaluation metric |
| **tCO2e** | Tonnes of CO2 equivalent -- standard unit for carbon credits |
| **GMW** | Global Mangrove Watch -- satellite-derived global mangrove extent dataset |
| **L2A** | Level-2A -- atmospherically corrected surface reflectance (not top-of-atmosphere) |
| **Polygon Amoy** | Current Polygon testnet (Mumbai was deprecated April 2024) |

---

*Prepared for Sem VI viva examination. All constants are from published IPCC sources. All metrics are from actual model runs on CoastalCred data.*
