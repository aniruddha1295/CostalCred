# CoastalCred — Product Requirements Document (PRD)

**Project:** CoastalCred — Blockchain-Based Blue Carbon Registry and MRV System
**Course:** Sem VI Mini Project 2025-26, Dept. of Data Science, RCOEM Nagpur
**Team:** Ansh Chopada (Lead), Aniruddha Lahoti, + 2 teammates
**Guide:** Dr. Aarti Karandikar (HOD, Data Science)
**Industry Mentor:** Rishikesh Kale (DevRel, Filecoin / Protocol Labs)
**Document Owner:** Ansh Chopada
**Status:** Draft v1 — sprint target

---

## 1. Problem Statement

India has 4,600+ km of coastline with significant blue carbon sequestration potential via mangroves, seagrass, and salt marshes. The current carbon credit market fails coastal communities and climate action for four structural reasons:

1. **Slow verification** — carbon credit verification takes 18+ months of manual expert review
2. **Opaque provenance** — credit authenticity cannot be independently verified, enabling double-counting and greenwashing
3. **Intermediary extraction** — 40–60% of credit value is captured by brokers and verifiers, not the communities doing restoration
4. **No direct marketplace** — small-scale restoration providers (NGOs, village cooperatives) cannot reach corporate buyers without intermediaries

CoastalCred addresses these by combining **AI-accelerated satellite verification** (60 days instead of 18 months), **blockchain-based immutable credit records** (no double-counting), and a **direct marketplace with automated escrow** (no intermediaries). The system follows IPCC methodology and is designed to align with India's Carbon Credit Trading Scheme (CCTS 2023).

---

## 2. Goals and Non-Goals

### In Scope (Sem VI — this project)

**Phase 1 (Design & Architecture):**
- Complete system architecture design and documentation
- Database schema (PostgreSQL + PostGIS) for ~23 core tables
- Smart contract architecture specification (ERC-721 Territory NFTs, ERC-1155 Carbon Credits, Marketplace)
- REST API specification (OpenAPI/Swagger) for 7 microservices
- Repository skeleton with Docker Compose, CI/CD stub
- Proof-of-concept spikes: PostGIS spatial query, Hardhat smart contract compilation, Sentinel-2 fetch via Google Earth Engine

**Phase 2 (Data + ML):**
- End-to-end data pipeline from Sentinel-2 imagery to ML-ready patches
- Three trained/implemented models: NDVI threshold baseline, XGBoost pixel classifier, U-Net semantic segmentation
- Evaluation on held-out test site with Precision, Recall, IoU, F1
- IPCC Tier 1 carbon calculation module computing tCO₂e per territory
- Flux-based credit calculation using 2020 vs 2024 mangrove cover change
- Phase 2 report with model comparison, methodology, and results

### Out of Scope (deferred to Sem VII or beyond)

The following are *intentionally* excluded from this sprint. Calling them out prevents scope creep and is itself a design decision:

- Deployed smart contracts on testnet (architecture only in this phase)
- Frontend portals (Community Portal, Corporate Portal) — design only
- Live backend microservices (modular monolith code instead of deployed services)
- Mobile app with offline sync for field data collection
- Apache Kafka event bus (in-process events for sprint; Kafka design documented)
- TimescaleDB (design-only; justification: "Sem VII when ingesting IoT sensor streams and monthly composites")
- IPFS document storage (design-only; Rishikesh Kale's expertise leveraged in Sem VII)
- Aadhaar KYC / UPI payment integration (design-only)
- Real-world field validation of model predictions
- Production security audit / formal verification of contracts
- Mainnet blockchain deployment

---

## 3. Users and Personas

| Persona | Goal | Key Interaction |
|---|---|---|
| **NGO / Community Restoration Provider** | Monetize restoration work transparently, get fair price | Registers territory with GPS polygon, uploads planting records and field photos, tracks verification status |
| **Corporate Carbon Credit Buyer** | Purchase verifiable, IPCC-compliant carbon offsets | Browses marketplace, buys ERC-1155 credit batches, generates compliance reports |
| **Verification Admin / Domain Expert** | Review and approve credit issuance requests | Reviews ML model outputs + provenance documentation, approves or rejects credit minting |
| **Field Worker** | Capture ground-truth data in remote coastal sites | Collects GPS-tagged photos via mobile app (Sem VII) |
| **Researcher / Regulator** | Audit registry, analyze sequestration trends | Queries registry via read-only API, exports reports |

---

## 4. User Stories (Sprint Target)

Only the stories actually implementable in this sprint. Others are deferred to Sem VII.

**US-01.** As an NGO, I want to register a mangrove territory with a GPS polygon so that my restoration work has a verifiable on-record location.

**US-02.** As a verification system, I want to fetch a cloud-free Sentinel-2 composite for a registered territory so that I can measure current mangrove cover.

**US-03.** As a verification system, I want to compare mangrove cover between a project's baseline year (2020) and the current year (2024) so that I can compute *additional* carbon sequestration, not pre-existing stock.

**US-04.** As a verification system, I want to run three independent models (NDVI, XGBoost, U-Net) and report their metrics so that the best-performing model can be used for credit issuance with confidence.

**US-05.** As a verification system, I want to convert predicted mangrove hectares into tCO₂e using IPCC Tier 1 defaults so that output is in a standardized carbon accounting unit.

**US-06.** As an auditor, I want every design decision to be documented with its justification so that technical reviewers can evaluate the system without ambiguity.

**US-07.** As a developer onboarding to the project, I want a single-command data setup (`python download_data.py`) so that I can start contributing within 15 minutes.

---

## 5. Functional Requirements

### 5.1 Data Pipeline (MUST)

| ID | Requirement |
|---|---|
| FR-1.1 | System shall fetch Sentinel-2 L2A cloud-free median composites via Google Earth Engine for three sites (Sundarbans, Gulf of Kutch, Pichavaram) for two years (2020, 2024) |
| FR-1.2 | System shall use six bands: B2, B3, B4, B8, B11, B12 (Blue, Green, Red, NIR, SWIR1, SWIR2) at 10m resolution |
| FR-1.3 | System shall rasterize Global Mangrove Watch polygon labels to match each Sentinel-2 image's CRS, resolution, and bounds |
| FR-1.4 | System shall extract 256×256 patches from aligned image/mask pairs with configurable stride |
| FR-1.5 | System shall split patches by site (train: 2 sites, test: 1 held-out site) to measure real generalization |
| FR-1.6 | System shall compute and persist per-band normalization statistics (mean, std) from training data only |

### 5.2 ML Models (MUST)

| ID | Requirement |
|---|---|
| FR-2.1 | System shall implement an NDVI threshold classifier as a rule-based baseline |
| FR-2.2 | System shall train an XGBoost pixel classifier using 10 features (6 bands + NDVI, EVI, NDWI, SAVI) with `scale_pos_weight` handling class imbalance |
| FR-2.3 | System shall train a U-Net (ResNet-18 encoder, ImageNet pretrained) via `segmentation_models_pytorch` with 6-channel input |
| FR-2.4 | System shall use mixed precision training (`torch.cuda.amp`) to fit training on 4 GB VRAM |
| FR-2.5 | System shall save the best U-Net checkpoint by *validation* IoU (not training IoU) |
| FR-2.6 | System shall evaluate all three models on the same held-out test set with identical metric code |

### 5.3 Evaluation (MUST)

| ID | Requirement |
|---|---|
| FR-3.1 | System shall report Precision, Recall, IoU, and F1 for each model |
| FR-3.2 | System shall produce a comparison table (CSV + PNG) for the three models, including training time |
| FR-3.3 | System shall produce a per-site breakdown of metrics (Sundarbans vs Gulf of Kutch vs Pichavaram) |
| FR-3.4 | System shall save 5 visualized prediction examples per model (RGB | ground truth | prediction) |

### 5.4 Carbon Calculation (MUST)

| ID | Requirement |
|---|---|
| FR-4.1 | System shall convert predicted mangrove mask to hectares using pixel area × pixel count |
| FR-4.2 | System shall compute carbon stock as `hectares × 230 (biomass density) × 0.47 (C fraction) × 3.67 (CO₂:C)` tCO₂e |
| FR-4.3 | System shall compute annual flux as `hectares × 7.0` tCO₂e/year (IPCC Wetlands Supplement default) |
| FR-4.4 | System shall compute flux-based credits as `(current_hectares − baseline_hectares) × 7.0 × years_elapsed` |
| FR-4.5 | All IPCC constants shall be documented with source citations in code comments and the report |

### 5.5 Architecture Deliverables (MUST)

| ID | Requirement |
|---|---|
| FR-5.1 | System shall deliver a system architecture document with microservices responsibilities |
| FR-5.2 | System shall deliver an ERD and SQL DDL for ~23 tables covering users, territories, verification, credits, marketplace, blockchain sync, audit |
| FR-5.3 | System shall deliver smart contract architecture specs for TerritoryNFT (ERC-721), CarbonCredit (ERC-1155), Marketplace |
| FR-5.4 | System shall deliver OpenAPI specifications for 7 microservices |
| FR-5.5 | System shall deliver three data-flow diagrams (NGO registration, verification, corporate purchase) |
| FR-5.6 | System shall deliver POC spikes: PostGIS spatial query, Hardhat contract compilation, GEE Sentinel-2 fetch |

### 5.6 Developer Experience (MUST)

| ID | Requirement |
|---|---|
| FR-6.1 | Repository shall be on GitHub with private access for all 4 team members |
| FR-6.2 | Raw data shall be distributed via GitHub Releases, not Git LFS or Drive |
| FR-6.3 | A single Python script (`download_data.py`) shall fetch all raw data to correct local paths |
| FR-6.4 | `.gitignore` shall exclude `data/`, `models/`, and binary artifacts from version control |
| FR-6.5 | README shall include a 3-command setup section for new team members |

---

## 6. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Reproducibility** | Anyone with repo access shall be able to regenerate patches, train models, and produce the comparison table by running documented scripts in order |
| **Hardware** | U-Net shall train within 4 GB VRAM (reduce batch size, filters, or use gradient accumulation as needed) |
| **Training time** | Full pipeline (data → 3 models → comparison table → carbon calc) shall complete within a 2-day sprint budget |
| **Citation integrity** | Every IPCC constant, dataset, and methodology shall be cited in the Phase 2 report |
| **Cost** | Zero dollars spent. All tools (GEE, GitHub, GMW, PyTorch, Colab fallback) shall be on free tiers |
| **Code quality** | Models shall share a single metric evaluation module to ensure apples-to-apples comparison |

---

## 7. ML Specification (Detail)

### 7.1 Input
- Sentinel-2 L2A (`COPERNICUS/S2_SR_HARMONIZED` in GEE)
- 6 bands: B2, B3, B4, B8, B11, B12
- 10m spatial resolution
- Patches: 256×256 pixels

### 7.2 Labels
- Global Mangrove Watch (Bunting et al., 2018, Remote Sensing 10(10):1669)
- Rasterized to match Sentinel-2 grid
- Binary: 1 = mangrove, 0 = not mangrove

### 7.3 Models

**NDVI Baseline (Rule-based):**
- `ndvi = (B8 − B4) / (B8 + B4 + ε)`
- Prediction: `ndvi > threshold`, threshold tuned on validation set
- No training; ~0 seconds

**XGBoost (Classical ML):**
- Features: 6 raw bands + 4 indices (NDVI, EVI, NDWI, SAVI) = 10 per pixel
- Hyperparameters: `n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=<ratio>`
- Early stopping on validation logloss
- Training: ~5–10 min on CPU

**U-Net (Deep Learning):**
- Architecture: `segmentation_models_pytorch.Unet` with ResNet-18 encoder
- Pretrained ImageNet weights (encoder only; decoder trained from scratch)
- Input: 6 channels, Output: 1 channel (binary mask)
- Loss: `BCEWithLogitsLoss(pos_weight=class_imbalance_ratio)`
- Optimizer: AdamW, lr=1e-4, weight decay=1e-5
- Mixed precision via `torch.cuda.amp`
- Batch size 8, ~30–50 epochs, early stopping on val IoU
- Training: ~60–90 min on RTX 3050

### 7.4 Evaluation Metrics

- **Precision** = TP / (TP + FP) — punishes over-claiming mangroves → fraud risk
- **Recall** = TP / (TP + FN) — punishes under-claiming mangroves → missed credits
- **IoU** = TP / (TP + FP + FN) — overlap between predicted and true mask, the standard for segmentation
- **F1** = harmonic mean of Precision and Recall

**Why not plain accuracy:** Mangroves are the rare class (~2% of pixels in a typical tile). A trivial "always predict not-mangrove" classifier scores ~98% accuracy and finds zero mangroves. IoU, Precision, and Recall correctly penalize this failure mode.

### 7.5 Data Split Strategy

- **Train:** Sundarbans + Gulf of Kutch
- **Validation:** 10% held-out from training sites
- **Test:** Pichavaram (fully unseen during training)

Rationale: site-level split measures *generalization to new geographies*, not memorization. Random patch splits would leak adjacent patches between train and test.

### 7.6 Carbon Calculation Methodology (IPCC Tier 1)

```
Step 1 — hectares from model: count(mask==1) × pixel_area / 10,000
Step 2 — biomass: hectares × 230 t/ha  [IPCC default for mangroves]
Step 3 — carbon: biomass × 0.47  [IPCC carbon fraction]
Step 4 — CO₂ equivalent: carbon × (44/12)  [molecular ratio]

Annual flux: hectares × 7.0 tCO₂e/ha/year  [IPCC Wetlands Supplement]
Flux credits (2020→2024): (h_2024 − h_2020) × 7.0 × 4 years
```

All constants sourced from IPCC 2013 Supplement to the 2006 Guidelines: Wetlands.

---

## 8. System Architecture Overview

### 8.1 High-Level Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (Sem VII)                                         │
│  Community Portal  |  Corporate Portal  |  Admin Dashboard  │
└────────────────────────┬────────────────────────────────────┘
                         │ REST / WebSocket
┌────────────────────────┴────────────────────────────────────┐
│  Application Services (7 microservices, design in Sem VI)  │
│  Auth | Territory | DataIngestion | Verification |          │
│  CreditIssuance | Marketplace | BlockchainAdapter            │
└─┬────────────┬──────────────┬───────────────────┬───────────┘
  │            │              │                   │
  ▼            ▼              ▼                   ▼
┌────┐   ┌─────────┐   ┌────────────┐   ┌──────────────────┐
│ DB │   │ ML API  │   │ Blockchain │   │ Object Storage   │
│Pg+ │   │FastAPI  │   │ Polygon    │   │ IPFS / GCS (S7)  │
│PostGIS│  │U-Net etc│  │ Amoy       │   │                  │
└────┘   └─────────┘   └────────────┘   └──────────────────┘
```

### 8.2 Sprint Implementation Strategy
- Design all 7 services as documented architecture
- Implement as a **modular monolith** (single repo, clear module boundaries, no separate deployments) for speed
- Sem VII splits modules into deployed microservices

### 8.3 Technology Stack Summary

| Layer | Technology | Why |
|---|---|---|
| Backend services (design) | Node.js + Express + TypeScript | Team familiarity from ORLA3 |
| ML API | Python + FastAPI | Standard ML ecosystem |
| Frontend (design) | Next.js + Tailwind | Team familiarity |
| Smart contracts (design) | Solidity + Hardhat + Polygon Amoy | Polygon = low gas; Amoy = current testnet (Mumbai deprecated April 2024) |
| Primary database | PostgreSQL 15 + PostGIS | Open-source, geospatial support |
| Time-series DB | TimescaleDB (Sem VII) | Deferred |
| Document storage | IPFS via web3.storage (Sem VII) | Aligns with Filecoin mentor |
| ML training | PyTorch + `segmentation_models_pytorch` + XGBoost + scikit-learn | Standard |
| Geospatial data | Google Earth Engine, `rasterio`, `geopandas` | GEE avoids raw tile download pain |
| Dev environment | Docker Compose, Git, GitHub | Reproducible |
| CI/CD (skeleton) | GitHub Actions | Free for private repos with Pro |

---

## 9. Data Model Overview (Key Entities)

Detailed ERD lives in `docs/erd.png` and `sql/schema.sql`. High-level entities only here:

- **User** — NGO, Corporate, Verifier, Admin (with role-based access)
- **Territory** — GPS polygon (PostGIS geometry), owner, project type, baseline condition, baseline NDVI, project start date
- **PlantingRecord** — linked to territory, records species, count, date, worker attestations
- **FieldVisit** — GPS-tagged photos, timestamp, visitor identity (Sem VII mobile app)
- **VerificationEvent** — links territory to ML model run, stores predictions and confidence
- **CarbonCreditBatch** — ERC-1155 token ID, territory ref, tCO₂e amount, vintage year, confidence score
- **MarketplaceOrder** — buyer, seller, credit batch ref, price, escrow state
- **BlockchainTransaction** — tx hash, contract, function, status (for on-chain sync)
- **AuditLog** — immutable trail of every state change

---

## 10. Team Roles and Ownership

4-person team for this sprint.

| Member | Role | Accountable for |
|---|---|---|
| **Aniruddha Lahoti** | Data Pipeline + Infra Lead | GEE fetch, GMW alignment, patch extraction, splits, repo skeleton, Docker, DB schema, `download_data.py` |
| **[U-Net Lead — TBD]** | Deep Learning Lead | PyTorch `Dataset` class, U-Net code, training loop, GPU ops, U-Net evaluation. **Owns RTX 3050 machine.** |
| **[Classical+Eval Lead — TBD]** | Classical ML + Evaluation + Carbon Lead | NDVI baseline, XGBoost training + feature importances, shared metrics module, comparison table, IPCC carbon module, prediction visualizations |
| **Ansh Chopada** | Architecture + Documentation Lead (Team Lead) | PRD (this document), architecture doc, ERD diagrams, smart contract specs, data flow diagrams, OpenAPI spec, Phase 1 report, Phase 2 report, viva prep, team coordination |

---

## 11. Success Criteria

The sprint is complete when *all* of the following are demonstrable:

1. ✅ GitHub repo exists with complete folder structure, `.gitignore`, and README "Getting Started" section
2. ✅ `python download_data.py` works end-to-end for any teammate on a fresh clone
3. ✅ Raw Sentinel-2 composites exist for 3 sites × 2 years (6 GeoTIFFs total)
4. ✅ GMW labels exist and align correctly with Sentinel-2 grids (spot-check passes)
5. ✅ ~500+ aligned 256×256 image/mask patches exist on disk
6. ✅ Train/val/test split files exist; site-level split enforced
7. ✅ NDVI baseline runs and reports metrics
8. ✅ XGBoost model trains, saves, and reports metrics + feature importances
9. ✅ U-Net trains, saves checkpoint, and reports metrics
10. ✅ Comparison table (PNG + CSV) exists with all 3 models
11. ✅ IPCC carbon calculation outputs tCO₂e per site (stock + 4-year flux)
12. ✅ Phase 1 report PDF submittable
13. ✅ Phase 2 report PDF submittable
14. ✅ Team can answer the 7 viva questions confidently (see §13)

---

## 12. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GEE authentication fails or rate-limits | Medium | High | Start sign-up immediately; Colab backup account ready |
| RTX 3050 CUDA not working | Medium | High | Verify `torch.cuda.is_available()` before sprint starts; Colab fallback |
| U-Net OOM on 4 GB VRAM | High | Medium | Reduce batch size to 4, enable mixed precision, reduce base filters |
| GMW and Sentinel-2 CRS/resolution mismatch | High | High | Use `rasterio.warp.reproject` to force-match before patching |
| XGBoost install fails on macOS (libomp missing) | Medium | Low | `brew install libomp` before sprint |
| U-Net training takes longer than budget | Medium | Medium | Reduce to 2 training sites; use smaller encoder |
| Teammate unavailable mid-sprint | Low | High | Ansh redistributes; Aniruddha is backup for Role 3 |
| "Polygon Mumbai" mentioned in synopsis is deprecated | Certain | Low | Flag correction to Dr. Karandikar proactively; design doc uses Polygon Amoy |
| Scope creep from team wanting to build frontend/deploy contracts | Medium | High | This PRD's §2 "Out of Scope" is the authoritative list. Any additions require explicit agreement. |

---

## 13. Viva Preparation Questions

The 7 questions the team must be able to answer confidently by the end of the sprint. Answers are rehearsed and documented in `docs/viva_prep.md`.

1. **How does your system prevent fraud / ensure additionality?**
2. **Why U-Net and not DeepLabV3+ or SegFormer?**
3. **Why XGBoost and not Random Forest?**
4. **Why measure flux (change over time) instead of stock (snapshot)?**
5. **How do you convert mangrove hectares to tCO₂e?**
6. **What's the class imbalance problem and how did you handle it?**
7. **Why three models instead of just one?**

---

## 14. Deliverables Checklist

### Phase 1 (Design & Architecture)
- [ ] `docs/prd.md` — this document
- [ ] `docs/architecture.md` — system architecture
- [ ] `docs/data_flow.md` — user journey diagrams
- [ ] `sql/schema.sql` — ~23-table DDL
- [ ] `docs/erd.png` — ER diagram
- [ ] `docs/smart_contracts.md` — contract specifications
- [ ] `api/openapi.yaml` — 7-service API spec
- [ ] `docs/tech_stack.md` — technology decisions with justifications
- [ ] `docker-compose.yml` — Postgres+PostGIS service
- [ ] `.github/workflows/ci.yml` — GitHub Actions stub
- [ ] POC: PostGIS spatial query demo
- [ ] POC: Hardhat `TerritoryNFT.sol` + `CarbonCredit.sol` compile
- [ ] POC: GEE Sentinel-2 fetch for Sundarbans works
- [ ] `reports/phase1_report.pdf`

### Phase 2 (Data + ML)
- [ ] `src/data_pipeline/fetch_sentinel2.py`
- [ ] `src/data_pipeline/align_masks.py`
- [ ] `src/data_pipeline/extract_patches.py`
- [ ] `src/data_pipeline/download_data.py`
- [ ] GitHub Release `v0.1-raw-data` published with 9 raw files
- [ ] `src/models/ndvi/baseline.py`
- [ ] `src/models/xgboost/train.py` + `feature_importance.png`
- [ ] `src/models/unet/model.py` + `train.py` + `evaluate.py`
- [ ] `src/evaluation/metrics.py` — shared across all models
- [ ] `src/carbon/ipcc_tier1.py`
- [ ] `results/{ndvi,xgboost,unet}.json`
- [ ] `results/comparison_table.{csv,png}`
- [ ] `results/per_site_breakdown.csv`
- [ ] `results/carbon_per_site.csv`
- [ ] `results/predictions/` — 5 visualization PNGs per model
- [ ] `models/unet_best.pt` published to GitHub Release `v0.2-trained-models`
- [ ] `reports/phase2_report.pdf`

---

## 15. Glossary

- **Additionality** — the principle that a carbon credit must represent CO₂ sequestration that *would not have happened without the project*. Natural sequestration that would have happened anyway does not generate valid credits.
- **Blue carbon** — carbon stored in coastal and marine ecosystems, primarily mangroves, seagrass meadows, and salt marshes.
- **CCTS 2023** — India's Carbon Credit Trading Scheme, notified 2023.
- **Flux** — rate of carbon absorption per unit time (tCO₂e/year). Contrast with stock.
- **GMW** — Global Mangrove Watch, a global mangrove extent dataset from UNEP-WCMC.
- **IoU** — Intersection over Union, the standard segmentation metric: overlap between predicted and ground-truth regions.
- **IPCC Tier 1** — default carbon accounting methodology using published global default values; contrast with Tier 2 (region-specific) and Tier 3 (site-specific).
- **MRV** — Monitoring, Reporting, and Verification; the methodological framework for carbon credit validation.
- **NDVI** — Normalized Difference Vegetation Index, `(NIR - Red) / (NIR + Red)`.
- **Sentinel-2** — European Space Agency Earth observation satellite constellation providing 10m multispectral imagery.
- **Stock** — total carbon currently stored in an ecosystem at a point in time (tCO₂e). Contrast with flux.
- **tCO₂e** — tonnes of CO₂ equivalent; standardized unit for carbon credits.
- **U-Net** — encoder-decoder CNN architecture originally designed for biomedical image segmentation; standard baseline for remote sensing segmentation.

---

## 16. References

1. Bunting, P. et al. (2018). *The Global Mangrove Watch — A New 2010 Global Baseline of Mangrove Extent*. Remote Sensing, 10(10), 1669.
2. IPCC (2014). *2013 Supplement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories: Wetlands*.
3. Copernicus Sentinel-2 data, ESA, accessed via Google Earth Engine.
4. Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
5. Chen, T., Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
6. Government of India (2023). *Carbon Credit Trading Scheme (CCTS)*, Ministry of Power notification.

---

**Document status:** v1 draft — ready for handoff to Claude Code and team execution.
