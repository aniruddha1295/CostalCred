# CoastalCred: Blockchain-Based Blue Carbon Registry and MRV System

## Phase 1 Report — Design & Architecture

---

**Institution:** Shri Ramdeobaba College of Engineering and Management (RCOEM), Nagpur
**Department:** Data Science
**Course:** Semester VI Mini Project, 2025--26

**Project Guide:** Dr. Aarti Karandikar (HOD, Dept. of Data Science)
**Industry Mentor:** Rishikesh Kale (Developer Relations, Filecoin / Protocol Labs)

**Team Members:**

| Name | Role |
|------|------|
| Ansh Chopada (Team Lead) | Architecture + Documentation Lead |
| Aniruddha Lahoti | Data Pipeline + Infra Lead |
| Member 3 | Deep Learning Lead |
| Member 4 | Classical ML + Evaluation + Carbon Lead |

**Date:** April 2026

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [System Architecture](#3-system-architecture)
4. [Database Design](#4-database-design)
5. [Smart Contract Architecture](#5-smart-contract-architecture)
6. [API Design](#6-api-design)
7. [Technology Stack](#7-technology-stack)
8. [Proof of Concept Results](#8-proof-of-concept-results)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)

---

## 1. Abstract

India possesses over 4,600 km of coastline with significant blue carbon sequestration potential through mangrove ecosystems, yet the existing carbon credit market suffers from slow verification cycles (18+ months), opaque provenance enabling double-counting and greenwashing, and intermediary extraction capturing 40--60% of credit value. CoastalCred addresses these systemic failures by combining AI-accelerated satellite verification using Sentinel-2 imagery and machine learning, blockchain-based immutable credit records on Polygon, and a direct marketplace with automated escrow. The system follows IPCC Tier 1 methodology and is designed to align with India's Carbon Credit Trading Scheme (CCTS 2023). This Phase 1 report documents the complete design and architecture of the system, including a 7-microservice architecture, a PostgreSQL+PostGIS database schema comprising 23 tables, smart contract specifications for ERC-721 Territory NFTs and ERC-1155 Carbon Credits, REST API specifications for all services, and three proof-of-concept spikes validating core technical assumptions. The Phase 2 deliverable builds upon this foundation with an end-to-end data pipeline, three ML models (NDVI threshold, XGBoost, U-Net) for mangrove segmentation, and IPCC-compliant carbon credit calculation using flux-based methodology.

---

## 2. Introduction

### 2.1 Problem Statement

Mangrove forests are among the most carbon-dense ecosystems on Earth, sequestering carbon at rates 3--5 times greater than terrestrial forests per unit area. India's mangrove cover spans approximately 4,975 sq km across states including West Bengal (Sundarbans), Gujarat (Gulf of Kutch), and Tamil Nadu (Pichavaram), representing substantial blue carbon potential. However, this potential remains largely unmonetized due to four structural failures in the current carbon credit market:

1. **Slow verification:** Carbon credit verification currently requires 18+ months of manual expert review, creating prohibitive delays for small-scale restoration projects.

2. **Opaque provenance:** Credit authenticity cannot be independently verified by third parties, enabling double-counting (the same sequestration sold to multiple buyers) and greenwashing (credits issued for non-additional sequestration).

3. **Intermediary extraction:** Brokers and verification bodies capture 40--60% of credit value through fees, leaving coastal communities with insufficient returns on their restoration efforts.

4. **No direct marketplace:** Small-scale restoration providers such as NGOs and village cooperatives cannot access corporate buyers without expensive intermediaries.

### 2.2 Project Goals

CoastalCred aims to address these failures through three core innovations:

- **AI-accelerated satellite verification:** Reducing verification time from 18 months to approximately 60 days by using machine learning models to detect mangrove cover change from Sentinel-2 satellite imagery.

- **Blockchain-based immutable records:** Ensuring no double-counting by recording all verification events, credit issuances, and transactions on the Polygon blockchain, where they cannot be altered retroactively.

- **Direct marketplace with automated escrow:** Enabling peer-to-peer trading of carbon credits between restoration providers and corporate buyers, with smart contract-enforced escrow eliminating intermediary fees.

### 2.3 Scope

This project spans Semester VI (design + ML) and Semester VII (full stack + deployment). Phase 1, documented in this report, covers the complete system design and architecture. Phase 2 covers the data pipeline and ML model implementation. Semester VII will deliver deployed microservices, frontend portals, and blockchain integration.

### 2.4 Study Sites

Three Indian mangrove sites were selected to represent diverse coastal ecosystems:

| Site | State | Ecosystem Type | Role in ML Pipeline |
|------|-------|---------------|-------------------|
| Sundarbans | West Bengal | Gangetic delta, mud-dominated | Training |
| Gulf of Kutch | Gujarat | Arid coastal, reef-associated | Training |
| Pichavaram | Tamil Nadu | Backwater estuary, lagoon | Test (unseen) |

Temporal scope: 2020 (baseline year) and 2024 (current year), enabling 4-year flux measurement.

---

## 3. System Architecture

### 3.1 High-Level Architecture

CoastalCred is designed as a 7-microservice system. During the Semester VI sprint, these services are implemented as a modular monolith with clear module boundaries; Semester VII will split them into independently deployable containers.

```
+---------------------------------------------------------------+
|  FRONTEND LAYER (Sem VII)                                      |
|  Community Portal  |  Corporate Portal  |  Admin Dashboard     |
+-------------------------------+-------------------------------+
                                |
                          REST / WebSocket
                                |
+-------------------------------v-------------------------------+
|  APPLICATION SERVICES (7 Microservices)                        |
|                                                                |
|  +--------+  +-----------+  +---------------+  +------------+ |
|  |  Auth   |  | Territory |  | DataIngestion |  |Verification| |
|  | Service |  |  Service  |  |   Service     |  |  Service   | |
|  +--------+  +-----------+  +---------------+  +------------+ |
|                                                                |
|  +----------------+  +-------------+  +-------------------+    |
|  | CreditIssuance |  | Marketplace |  | BlockchainAdapter |    |
|  |    Service      |  |   Service   |  |     Service       |    |
|  +----------------+  +-------------+  +-------------------+    |
+----+----------+-------------+------------------+--------------+
     |          |             |                  |
     v          v             v                  v
+--------+ +---------+  +-----------+  +------------------+
|   DB   | |  ML API |  |Blockchain |  | Object Storage   |
| Pg +   | | FastAPI |  | Polygon   |  | IPFS / GCS       |
|PostGIS | | U-Net   |  | Amoy      |  | (Sem VII)        |
+--------+ +---------+  +-----------+  +------------------+
```

### 3.2 Service Responsibilities

#### 3.2.1 Auth Service
Manages user registration, authentication (JWT-based), role-based access control (NGO, Corporate, Verifier, Admin), and KYC status tracking. Supports API key management for programmatic access.

#### 3.2.2 Territory Service
Handles registration and management of mangrove restoration territories. Each territory is defined by a GPS polygon (stored as PostGIS geometry), linked to an owner (NGO), and tracked through a lifecycle (draft, submitted, active, suspended, retired). Manages associated documents (land titles, planting plans, field photos) and planting records.

#### 3.2.3 Data Ingestion Service
Orchestrates satellite data acquisition from Google Earth Engine. Fetches Sentinel-2 L2A cloud-free median composites for registered territories, downloads Global Mangrove Watch ground truth datasets, and manages ingestion jobs with status tracking. Stores composite metadata including bands, CRS, resolution, and spatial bounds.

#### 3.2.4 Verification Service
Coordinates the MRV (Monitoring, Reporting, and Verification) workflow. Links territories to baseline and current satellite composites, triggers ML model runs (NDVI, XGBoost, U-Net), stores model predictions and metrics, and routes results to human verifiers for final decision (approved, rejected, needs revision).

#### 3.2.5 Credit Issuance Service
Converts verified mangrove change into carbon credits. Implements IPCC Tier 1 carbon calculation (hectares to tCO2e), creates ERC-1155 token batches with vintage year and methodology metadata, manages credit lifecycle (pending, minted, active, retired, revoked), and enforces issuance rules (minimum confidence threshold, maximum credits per hectare).

#### 3.2.6 Marketplace Service
Enables peer-to-peer trading of carbon credits. Sellers create listings with price per credit, buyers place orders, and the system manages escrow through a defined lifecycle (pending, funded, released, refunded, disputed). Tracks all escrow events for audit purposes.

#### 3.2.7 Blockchain Adapter Service
Provides a uniform interface between application logic and the Polygon blockchain. Manages smart contract deployment, transaction submission and confirmation tracking, and maintains a contract registry mapping contract names to deployed addresses and ABIs. All blockchain interactions are logged for auditability.

### 3.3 Communication Patterns

- **Synchronous:** REST APIs between frontend and services, and between services when immediate responses are required.
- **Asynchronous (Design Only, Sem VII):** Apache Kafka event bus for decoupled communication. Key events: `territory.submitted`, `verification.completed`, `credit.minted`, `order.placed`. During the Sem VI sprint, events are in-process function calls.
- **Blockchain:** The BlockchainAdapter service abstracts all on-chain interactions, handling gas estimation, transaction retry, and confirmation polling.

### 3.4 Sprint Implementation Strategy

The full microservice architecture is documented for design purposes, but during Semester VI the system operates as a modular monolith: a single repository with clear module boundaries but no separate deployments. This approach balances architectural rigor with sprint velocity. Semester VII splits modules into deployed microservices.

---

## 4. Database Design

### 4.1 Overview

The database uses PostgreSQL 15 with the PostGIS extension for geospatial operations and the `uuid-ossp` extension for universally unique identifiers. The schema comprises 23 tables organized by service ownership, 13 custom ENUM types for type safety, and spatial indexes (GiST) on geometry columns.

### 4.2 Schema Organization by Service

| Service | Tables | Count |
|---------|--------|-------|
| Auth Service | `users`, `user_sessions`, `api_keys` | 3 |
| Territory Service | `territories`, `territory_documents`, `planting_records`, `field_visits` | 4 |
| Data Ingestion Service | `satellite_composites`, `ground_truth_datasets`, `ingestion_jobs` | 3 |
| Verification Service | `verification_requests`, `model_runs`, `verification_decisions` | 3 |
| Credit Issuance Service | `carbon_credit_batches`, `credit_retirements`, `issuance_rules` | 3 |
| Marketplace Service | `marketplace_listings`, `marketplace_orders`, `escrow_events` | 3 |
| Blockchain Adapter Service | `blockchain_transactions`, `contract_registry` | 2 |
| Shared | `audit_log`, `notifications` | 2 |
| **Total** | | **23** |

### 4.3 Key Tables and Relationships

#### 4.3.1 Users (Auth Service)

The `users` table is the central identity table with role-based access (`ngo`, `corporate`, `verifier`, `admin`), KYC status tracking (`pending`, `submitted`, `verified`, `rejected`), and organization metadata. Session management is handled via `user_sessions` with token hashes, IP addresses, and expiration. Programmatic access is supported through `api_keys` with scoped permissions.

#### 4.3.2 Territories (Territory Service)

The `territories` table is the core geospatial entity. Key columns:

| Column | Type | Purpose |
|--------|------|---------|
| `boundary` | `geometry(Polygon, 4326)` | GPS polygon in WGS 84, indexed with GiST |
| `area_hectares` | `NUMERIC(12, 4)` | Calculated area with positive constraint |
| `project_type` | `project_type` ENUM | Restoration, conservation, or afforestation |
| `baseline_ndvi` | `NUMERIC(5, 4)` | Baseline vegetation index, range [-1, 1] |
| `status` | `territory_status` ENUM | Lifecycle: draft, submitted, active, suspended, retired |

Supporting tables: `territory_documents` stores IPFS-referenced files (land titles, field photos), `planting_records` tracks species, count, and worker attestations, and `field_visits` stores GPS-tagged visit data with point geometry.

#### 4.3.3 Satellite Composites (Data Ingestion Service)

The `satellite_composites` table records metadata for each Sentinel-2 composite:
- Satellite source, band list, acquisition date range
- Cloud cover percentage, file path, file hash
- Spatial resolution (default 10m), CRS (default EPSG:4326)
- Bounds as PostGIS polygon for spatial queries

The `ingestion_jobs` table tracks asynchronous data acquisition tasks with status lifecycle (`queued`, `running`, `completed`, `failed`, `cancelled`).

#### 4.3.4 Verification Pipeline (Verification Service)

The verification workflow is modeled across three tables:
1. `verification_requests` links a territory to baseline and current composites
2. `model_runs` stores individual ML model executions with model name, version, parameters, predicted hectares (baseline and current), delta hectares, and metrics as JSONB
3. `verification_decisions` records human verifier decisions with confidence scores and notes

#### 4.3.5 Carbon Credits (Credit Issuance Service)

`carbon_credit_batches` represents ERC-1155 token batches with:
- Territory and verification request references
- `token_id_erc1155` (unique blockchain identifier)
- `amount_tco2e` with positive constraint
- Vintage year, methodology, confidence score
- Credit lifecycle: `pending`, `minted`, `active`, `retired`, `revoked`
- Minting transaction hash for blockchain traceability

The `issuance_rules` table stores IPCC constants as configurable parameters:

| Parameter | Default Value | Source |
|-----------|---------------|--------|
| `biomass_density_t_per_ha` | 230.0 | IPCC Wetlands Supplement 2013 |
| `carbon_fraction` | 0.47 | IPCC default |
| `co2_to_c_ratio` | 3.6667 | Molecular weight ratio (44/12) |
| `annual_sequestration_rate` | 7.0 tCO2e/ha/yr | IPCC default |
| `min_confidence` | 0.70 | System threshold |
| `max_credits_per_hectare` | 7.0 | IPCC flux rate |

#### 4.3.6 Marketplace (Marketplace Service)

Three tables model the trading workflow:
- `marketplace_listings`: seller creates a listing with price per credit and available quantity
- `marketplace_orders`: buyer places an order against a listing
- `escrow_events`: tracks escrow lifecycle (`created`, `funded`, `released`, `refunded`, `disputed`, `resolved`)

#### 4.3.7 Blockchain Adapter

- `blockchain_transactions`: records every on-chain transaction with hash, contract address, function name, sender/receiver addresses, parameters (JSONB), status, block number, gas used, and network (default: `polygon_amoy`)
- `contract_registry`: maps contract names to deployed addresses with ABI hashes, enabling the adapter to route calls to the correct contracts

#### 4.3.8 Shared Tables

- `audit_log`: Append-only immutable trail of every state change. Records user, action, entity type/ID, old and new values (JSONB), and IP address. Critically, this table has no `updated_at` column -- rows are never modified or deleted.
- `notifications`: User-facing notifications with type classification (`info`, `warning`, `success`, `error`, `action_required`) and read status tracking.

### 4.4 PostGIS Usage

PostGIS is used for three primary geospatial operations:

1. **Territory boundary storage:** `geometry(Polygon, 4326)` columns with GiST spatial indexes enable efficient spatial queries (e.g., "find all territories within 50 km of a point").

2. **Field visit location tracking:** `geometry(Point, 4326)` columns on `field_visits` allow proximity queries and map visualizations.

3. **Satellite composite bounds:** Polygon bounds on `satellite_composites` enable spatial joins between territories and available imagery.

### 4.5 ENUM Types

The schema defines 13 custom ENUM types for type safety and self-documenting column constraints:

| ENUM | Values | Used By |
|------|--------|---------|
| `user_role` | ngo, corporate, verifier, admin | `users` |
| `kyc_status` | pending, submitted, verified, rejected | `users` |
| `territory_status` | draft, submitted, active, suspended, retired | `territories` |
| `project_type` | mangrove_restoration, conservation, afforestation | `territories` |
| `document_type` | land_title, satellite_image, field_photo, etc. | `territory_documents` |
| `job_status` | queued, running, completed, failed, cancelled | `ingestion_jobs` |
| `verification_status` | pending, in_progress, completed, rejected, expired | `verification_requests` |
| `verification_decision` | approved, rejected, needs_revision | `verification_decisions` |
| `credit_status` | pending, minted, active, retired, revoked | `carbon_credit_batches` |
| `listing_status` | active, paused, sold_out, cancelled | `marketplace_listings` |
| `escrow_status` | pending, funded, released, refunded, disputed | `marketplace_orders` |
| `escrow_event_type` | created, funded, released, refunded, disputed, resolved | `escrow_events` |
| `tx_status` | pending, submitted, confirmed, failed, reverted | `blockchain_transactions` |

---

## 5. Smart Contract Architecture

The smart contract architecture is a design-only deliverable for Semester VI. Deployment and integration will occur in Semester VII on the Polygon Amoy testnet.

> **Note:** The original project synopsis references "Polygon Mumbai" as the target testnet. Mumbai was deprecated in April 2024. All design documents reference **Polygon Amoy**, the current Polygon testnet. This correction has been flagged to Dr. Karandikar.

### 5.1 TerritoryNFT (ERC-721)

**Purpose:** Represents verified mangrove restoration territories as unique, non-fungible tokens.

**Key Properties:**
- Each territory becomes a unique NFT upon successful verification
- Metadata includes GPS boundary polygon hash, owner identity, project type, baseline condition, and verification history
- Ownership is transferable (enables territory trading or transfer to new custodians)
- Only the Verification Service (via admin role) can mint new TerritoryNFTs

**Key Functions:**
| Function | Description |
|----------|-------------|
| `mint(owner, metadataURI)` | Mint new territory NFT (admin only) |
| `updateMetadata(tokenId, newURI)` | Update territory metadata (admin only) |
| `suspend(tokenId)` | Suspend territory (prevents credit issuance) |
| `retire(tokenId)` | Permanently retire territory |
| `getTerritory(tokenId)` | View territory metadata |

**Design Rationale:** ERC-721 was chosen because each territory is unique (different boundaries, different owners, different conditions). ERC-721's non-fungibility correctly models this -- two territories are never interchangeable.

### 5.2 CarbonCredit (ERC-1155)

**Purpose:** Represents fungible carbon credit batches as semi-fungible tokens.

**Key Properties:**
- Each batch has a unique `tokenId` corresponding to a specific territory, vintage year, and verification event
- Credits within the same batch are fungible (1 tCO2e from batch X is equivalent to another tCO2e from batch X)
- Credits across batches are distinguishable (batch metadata differs)
- Supports batch minting and batch transfers for gas efficiency
- Credits can be retired (burned) by buyers to claim the offset

**Key Functions:**
| Function | Description |
|----------|-------------|
| `mintBatch(to, amount, data)` | Mint credits for a verified territory (admin only) |
| `retire(tokenId, amount, reason)` | Retire (burn) credits to claim offset |
| `safeTransferFrom(from, to, tokenId, amount)` | Transfer credits between addresses |
| `balanceOf(owner, tokenId)` | Query credit balance |
| `getBatchMetadata(tokenId)` | View batch metadata (territory, vintage, tCO2e) |

**Design Rationale:** ERC-1155 was chosen over ERC-20 because it supports multiple token types (batches) in a single contract, reducing deployment costs and simplifying the marketplace. Each batch retains its own metadata (vintage year, territory source, confidence score) while remaining fungible within the batch.

### 5.3 Marketplace Contract

**Purpose:** Enables peer-to-peer trading of carbon credits with on-chain escrow.

**Key Properties:**
- Sellers list credit batches with price per credit and available quantity
- Buyers fund orders via escrow
- Smart contract holds funds until conditions are met (delivery of credits)
- Dispute resolution mechanism with admin arbitration
- All trades are transparent and auditable on-chain

**Key Functions:**
| Function | Description |
|----------|-------------|
| `createListing(tokenId, amount, pricePerCredit)` | List credits for sale |
| `buyCredits(listingId, amount)` | Purchase credits (funds go to escrow) |
| `releaseFunds(orderId)` | Release escrow to seller (after credit transfer) |
| `refund(orderId)` | Refund buyer (cancellation or dispute resolution) |
| `dispute(orderId, reason)` | Raise dispute on an order |
| `resolveDispute(orderId, decision)` | Admin resolves dispute |

**Escrow State Machine:**
```
Created --> Funded --> Released (normal flow)
                  \--> Refunded (cancellation)
                  \--> Disputed --> Resolved --> Released or Refunded
```

### 5.4 Contract Interaction Flow

The three contracts interact in a defined sequence:

1. **Territory Registration:** Admin mints a TerritoryNFT after verification approval.
2. **Credit Issuance:** Admin mints CarbonCredit ERC-1155 tokens linked to the TerritoryNFT, with amount based on IPCC Tier 1 calculation.
3. **Trading:** Territory owner lists credits on the Marketplace contract. Corporate buyer purchases credits, which transfers ERC-1155 tokens and releases escrow funds.
4. **Retirement:** Corporate buyer retires (burns) credits to claim the carbon offset for their sustainability reporting.

---

## 6. API Design

### 6.1 Overview

The system exposes REST APIs for each of the 7 microservices, documented via OpenAPI/Swagger specifications. All APIs follow consistent conventions:

- **Authentication:** Bearer JWT tokens (Auth Service issues tokens)
- **Versioning:** URL path prefix `/api/v1/`
- **Response Format:** JSON with consistent error schema
- **Pagination:** Cursor-based for list endpoints
- **Rate Limiting:** Per-API-key or per-user token bucket

### 6.2 Auth Service API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register new user (NGO, Corporate) |
| POST | `/api/v1/auth/login` | Authenticate and receive JWT |
| POST | `/api/v1/auth/refresh` | Refresh expired JWT |
| POST | `/api/v1/auth/logout` | Invalidate session |
| GET | `/api/v1/auth/me` | Get current user profile |
| PUT | `/api/v1/auth/kyc` | Submit KYC documentation |
| POST | `/api/v1/auth/api-keys` | Create API key |
| DELETE | `/api/v1/auth/api-keys/{id}` | Revoke API key |

### 6.3 Territory Service API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/territories` | Register new territory with GPS polygon |
| GET | `/api/v1/territories` | List territories (with spatial filters) |
| GET | `/api/v1/territories/{id}` | Get territory details |
| PUT | `/api/v1/territories/{id}` | Update territory metadata |
| POST | `/api/v1/territories/{id}/documents` | Upload document (IPFS) |
| POST | `/api/v1/territories/{id}/planting-records` | Add planting record |
| GET | `/api/v1/territories/{id}/field-visits` | List field visits |
| GET | `/api/v1/territories/nearby?lat=&lng=&radius=` | Spatial proximity query |

### 6.4 Data Ingestion Service API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/ingestion/sentinel2` | Trigger Sentinel-2 composite fetch |
| POST | `/api/v1/ingestion/gmw` | Trigger GMW dataset download |
| GET | `/api/v1/ingestion/composites` | List available composites |
| GET | `/api/v1/ingestion/composites/{id}` | Get composite metadata |
| GET | `/api/v1/ingestion/jobs` | List ingestion jobs |
| GET | `/api/v1/ingestion/jobs/{id}` | Get job status |

### 6.5 Verification Service API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/verification/requests` | Create verification request |
| GET | `/api/v1/verification/requests` | List verification requests |
| GET | `/api/v1/verification/requests/{id}` | Get request with model run results |
| POST | `/api/v1/verification/requests/{id}/run-model` | Trigger ML model run |
| GET | `/api/v1/verification/requests/{id}/model-runs` | List model runs |
| POST | `/api/v1/verification/requests/{id}/decide` | Submit verifier decision |

### 6.6 Credit Issuance Service API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/credits/mint` | Mint carbon credit batch |
| GET | `/api/v1/credits/batches` | List credit batches |
| GET | `/api/v1/credits/batches/{id}` | Get batch details |
| POST | `/api/v1/credits/retire` | Retire credits (claim offset) |
| GET | `/api/v1/credits/retirements` | List retirements |
| GET | `/api/v1/credits/rules` | Get active issuance rules |

### 6.7 Marketplace Service API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/marketplace/listings` | Create listing |
| GET | `/api/v1/marketplace/listings` | Browse listings (with filters) |
| GET | `/api/v1/marketplace/listings/{id}` | Get listing details |
| PUT | `/api/v1/marketplace/listings/{id}` | Update listing (pause, cancel) |
| POST | `/api/v1/marketplace/orders` | Place order (triggers escrow) |
| GET | `/api/v1/marketplace/orders/{id}` | Get order status |
| POST | `/api/v1/marketplace/orders/{id}/dispute` | Raise dispute |

### 6.8 Blockchain Adapter Service API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/blockchain/submit` | Submit transaction |
| GET | `/api/v1/blockchain/transactions/{hash}` | Get transaction status |
| GET | `/api/v1/blockchain/contracts` | List deployed contracts |
| GET | `/api/v1/blockchain/contracts/{name}` | Get contract ABI and address |

---

## 7. Technology Stack

### 7.1 Complete Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Backend Services** (design) | Node.js + Express + TypeScript | Team familiarity; widely adopted for REST APIs |
| **ML API** | Python + FastAPI | Standard ML ecosystem; async support; auto-generated OpenAPI docs |
| **Frontend** (design) | Next.js + Tailwind CSS | Server-side rendering for SEO; team familiarity |
| **Smart Contracts** (design) | Solidity + Hardhat | Industry standard for EVM contracts; robust testing framework |
| **Blockchain** | Polygon Amoy (testnet) | Low gas costs; EVM-compatible; active testnet (Mumbai deprecated April 2024) |
| **Primary Database** | PostgreSQL 15 + PostGIS | Open-source; mature geospatial support; ACID compliance |
| **Time-Series DB** (Sem VII) | TimescaleDB | Optimized for IoT sensor streams and monthly composites |
| **Document Storage** (Sem VII) | IPFS via web3.storage | Decentralized; aligns with Filecoin mentor expertise |
| **ML Training** | PyTorch + segmentation_models_pytorch + XGBoost + scikit-learn | Standard ML/DL ecosystem |
| **Geospatial Data** | Google Earth Engine + rasterio + geopandas | GEE avoids raw tile download complexity; rasterio for local processing |
| **Containerization** | Docker Compose | Reproducible local development; database and service orchestration |
| **CI/CD** | GitHub Actions | Free for private repos; native GitHub integration |
| **Version Control** | Git + GitHub | Industry standard; GitHub Releases for large data artifacts |

### 7.2 Key Technology Decisions

#### 7.2.1 PostgreSQL + PostGIS over MongoDB

**Decision:** PostgreSQL with PostGIS for the primary database.

**Rationale:**
- Territory boundaries are polygons requiring spatial indexing and spatial queries (intersection, containment, proximity). PostGIS provides mature GiST indexes and ST_* functions for these operations.
- The data model has strong relational structure (users own territories, territories have verification requests, requests produce model runs, model runs lead to credit batches). Relational integrity via foreign keys prevents orphaned records.
- PostgreSQL ENUM types provide type safety for status fields (13 ENUMs in the schema).
- MongoDB's document model would require application-level joins and lacks native geospatial polygon operations at the same maturity level.

#### 7.2.2 Polygon over Ethereum Mainnet

**Decision:** Polygon Amoy testnet (with Polygon mainnet as the production target).

**Rationale:**
- Gas costs on Ethereum mainnet make per-credit minting economically infeasible (estimated $5--50 per transaction). Polygon's Layer 2 solution reduces gas costs by 100--1000x.
- Polygon is EVM-compatible, meaning Solidity contracts and Hardhat tooling work unchanged.
- The original synopsis referenced Polygon Mumbai, which was deprecated in April 2024. Polygon Amoy is the current testnet.
- Industry mentor Rishikesh Kale (Filecoin/Protocol Labs) confirmed Polygon as the recommended chain for carbon credit applications.

#### 7.2.3 Sentinel-2 over Landsat

**Decision:** Sentinel-2 L2A surface reflectance imagery via Google Earth Engine.

**Rationale:**
- Sentinel-2 provides 10m spatial resolution (vs. Landsat's 30m), a 3x improvement critical for detecting small mangrove patches.
- 5-day revisit time (vs. Landsat's 16 days) provides more cloud-free observations per composite.
- L2A products are atmospherically corrected (surface reflectance), eliminating the need for atmospheric correction preprocessing.
- Google Earth Engine provides cloud-based compositing, avoiding the need to download and process hundreds of raw scenes.

#### 7.2.4 ERC-1155 over ERC-20 for Carbon Credits

**Decision:** ERC-1155 semi-fungible tokens for carbon credit batches.

**Rationale:**
- Each credit batch has unique metadata (territory source, vintage year, verification event, confidence score). ERC-20 tokens are fully fungible with no per-batch metadata.
- ERC-1155 supports multiple token types in a single contract, reducing deployment costs.
- Batch transfers in ERC-1155 reduce gas costs for marketplace operations.
- Credits within the same batch are fungible (1 tCO2e = 1 tCO2e), but batches are distinguishable -- this semi-fungibility is exactly the token model carbon credits require.

#### 7.2.5 IPCC Tier 1 over Tier 2/3

**Decision:** IPCC Tier 1 methodology with global default constants.

**Rationale:**
- Tier 1 uses published global defaults (230 t/ha biomass density, 0.47 carbon fraction, 7.0 tCO2e/ha/yr sequestration rate) that are reproducible and verifiable by any reviewer.
- Tier 2 requires country-specific factors (not available for Indian mangroves at the required granularity).
- Tier 3 requires site-specific field measurements (outside Semester VI scope).
- Tier 1 is explicitly designed as the starting methodology in the IPCC Guidelines; Tier 2/3 refinement is an upgrade path, not a prerequisite.

---

## 8. Proof of Concept Results

Three proof-of-concept spikes were conducted to validate core technical assumptions before committing to full implementation.

### 8.1 POC 1: PostGIS Spatial Query

**Objective:** Verify that PostgreSQL + PostGIS can store territory polygon boundaries and perform spatial queries efficiently.

**Setup:**
- Docker Compose service: `postgres:15` with `postgis/postgis:15-3.4` image
- Schema loaded from `sql/schema.sql`
- Sample territory polygon inserted for Sundarbans region

**Test Query:**
```sql
-- Find all territories within 50 km of a given point
SELECT id, name, ST_Area(boundary::geography) / 10000 AS area_ha
FROM territories
WHERE ST_DWithin(
    boundary::geography,
    ST_SetSRID(ST_MakePoint(88.3, 21.9), 4326)::geography,
    50000  -- 50 km radius
);
```

**Result:** Query executed successfully in <10ms on a single territory. GiST index on the `boundary` column was utilized. PostGIS correctly handled EPSG:4326 coordinate reference system, geographic distance calculations, and polygon area computation.

**Conclusion:** PostGIS is suitable for the territory management use case. Spatial indexing will support production-scale queries across hundreds of territories.

### 8.2 POC 2: Hardhat Smart Contract Compilation

**Objective:** Verify that Solidity smart contracts for TerritoryNFT (ERC-721) and CarbonCredit (ERC-1155) compile successfully using the Hardhat framework.

**Setup:**
- Hardhat development environment with Solidity 0.8.x compiler
- OpenZeppelin contract library for ERC-721 and ERC-1155 base implementations
- Skeleton contracts implementing the interfaces described in Section 5

**Result:** Both `TerritoryNFT.sol` and `CarbonCredit.sol` compiled without errors. Gas estimation for mint operations was within acceptable bounds for Polygon. OpenZeppelin's ERC-721 and ERC-1155 implementations provided secure, audited base contracts.

**Conclusion:** The smart contract architecture is implementable with standard tooling. Deployment to Polygon Amoy is feasible for Semester VII.

### 8.3 POC 3: Google Earth Engine Sentinel-2 Fetch + NDVI

**Objective:** Verify that Sentinel-2 L2A composites can be fetched via Google Earth Engine and that NDVI computation produces meaningful vegetation indices for mangrove regions.

**Setup:**
- GCP project `costal-492719` with Earth Engine API enabled
- Service account authentication: `coastalcred-ee@costal-492719.iam.gserviceaccount.com`
- Target: Sundarbans region, 2024, cloud-free median composite

**Process:**
1. Authenticated with GEE using service account credentials
2. Filtered `COPERNICUS/S2_SR_HARMONIZED` collection for Sundarbans bounding box, 2024 date range, <20% cloud cover
3. Computed median composite across all qualifying scenes
4. Selected 6 bands: B2, B3, B4, B8, B11, B12
5. Exported as GeoTIFF at 10m resolution

**Result:**
- Successfully fetched a 1.1 GB GeoTIFF for Sundarbans 2024 (280 scenes composited)
- All 6 bands present with correct data range (uint16, 0--10000 scale)
- NDVI = (B8 - B4) / (B8 + B4) computed correctly, with values >0.5 corresponding to known mangrove regions
- RGB preview (B4, B3, B2) visually confirmed the Sundarbans delta geography

**Data Acquisition Summary (Full Pipeline):**

| Composite | File Size | Scenes |
|-----------|-----------|--------|
| Sundarbans 2024 | 1.1 GB | 280 |
| Sundarbans 2020 | 958 MB | -- |
| Gulf of Kutch 2024 | 472 MB | -- |
| Gulf of Kutch 2020 | 458 MB | -- |
| Pichavaram 2024 | 27 MB | -- |
| Pichavaram 2020 | 26 MB | -- |

All composites were published to GitHub Release `v0.1-raw-data` for team access.

**Conclusion:** The Google Earth Engine pipeline is functional and produces research-quality composites. The 6-band selection at 10m resolution is confirmed suitable for NDVI computation and mangrove detection.

---

## 9. Conclusion

### 9.1 Phase 1 Deliverables Summary

Phase 1 has delivered the complete design and architecture foundation for CoastalCred:

| Deliverable | Status | Location |
|-------------|--------|----------|
| Product Requirements Document | Complete | `docs/coastalcred-prd.md` |
| System Architecture (7 services) | Complete | This report, Section 3 |
| Database Schema (23 tables) | Complete | `sql/schema.sql` |
| Smart Contract Architecture | Complete | This report, Section 5 |
| REST API Specification | Complete | This report, Section 6; `api/openapi.yaml` |
| Technology Stack Justifications | Complete | This report, Section 7 |
| Docker Compose Configuration | Complete | `docker-compose.yml` |
| Repository Skeleton | Complete | GitHub repository with full directory structure |
| POC: PostGIS Spatial Query | Validated | Section 8.1 |
| POC: Hardhat Compilation | Validated | Section 8.2 |
| POC: GEE Sentinel-2 Fetch | Validated | Section 8.3 |

### 9.2 Key Design Decisions

The architecture embodies several deliberate decisions, each with documented rationale:

1. **Flux-based carbon accounting** (not stock) -- credits represent change over time, following IPCC and Verra methodology.
2. **Three-model comparison ladder** -- NDVI (rules) to XGBoost (classical ML) to U-Net (deep learning), enabling quantitative assessment of complexity vs. performance.
3. **Site-level train/test split** -- prevents data leakage and honestly measures generalization.
4. **Polygon Amoy** (not deprecated Mumbai) -- current testnet for EVM-compatible low-cost transactions.
5. **PostgreSQL + PostGIS** (not MongoDB) -- relational integrity and mature geospatial support.
6. **ERC-1155** (not ERC-20) for carbon credits -- semi-fungibility correctly models batch-level metadata with within-batch fungibility.
7. **Modular monolith** for sprint, microservices for production -- balances velocity with architectural rigor.

### 9.3 Phase 2 Preview

Phase 2 (Data & ML) builds directly on this architecture to deliver:

- **End-to-end data pipeline:** Sentinel-2 composites for 3 sites x 2 years, GMW ground truth alignment, 256x256 patch extraction, site-level train/val/test splits
- **Three ML models:** NDVI threshold baseline, XGBoost pixel classifier (10 spectral features), U-Net semantic segmentation (ResNet-18 encoder, 6-channel input)
- **Standardized evaluation:** Precision, Recall, IoU, F1 computed by shared evaluation code across all models
- **IPCC Tier 1 carbon calculation:** Hectares to tCO2e stock and flux credits
- **Comparison table:** Quantitative assessment of model performance across sites

The data pipeline (Phases B and C) is complete, with 6 Sentinel-2 composites fetched, GMW masks aligned, 7,000+ patches extracted, and site-level splits created. The ML implementation (Phases D, E, F) is in progress.

---

## 10. References

1. IPCC (2014). *2013 Supplement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories: Wetlands*. Hiraishi, T., Krug, T., Tanabe, K., Srivastava, N., Baasansuren, J., Fukuda, M. and Troxler, T.G. (eds). Published: IPCC, Switzerland.

2. Bunting, P., Rosenqvist, A., Lucas, R.M., Rebelo, L.-M., Hilarides, L., Thomas, N., Hardy, A., Itoh, T., Shimada, M., and Finlayson, C.M. (2018). "The Global Mangrove Watch -- A New 2010 Global Baseline of Mangrove Extent." *Remote Sensing*, 10(10), 1669. https://doi.org/10.3390/rs10101669

3. European Space Agency (ESA). *Copernicus Sentinel-2 Mission*. Sentinel-2 L2A surface reflectance data [2020, 2024], accessed via Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED`).

4. Ronneberger, O., Fischer, P., and Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Springer, LNCS, Vol. 9351, pp. 234--241.

5. Chen, T. and Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785--794.

6. Government of India, Ministry of Power (2023). *Carbon Credit Trading Scheme (CCTS)*. Gazette notification dated June 28, 2023.

7. Alongi, D.M. (2014). "Carbon Cycling and Storage in Mangrove Forests." *Annual Review of Marine Science*, 6, pp. 195--219.

8. Giri, C., Ochieng, E., Tieszen, L.L., Zhu, Z., Singh, A., Loveland, T., Masek, J., and Duke, N. (2011). "Status and Distribution of Mangrove Forests of the World Using Earth Observation Satellite Data." *Global Ecology and Biogeography*, 20(1), pp. 154--159.

9. OpenZeppelin (2024). *OpenZeppelin Contracts v5.x*. https://docs.openzeppelin.com/contracts/

10. Polygon Technology (2024). *Polygon Amoy Testnet Documentation*. https://docs.polygon.technology/

11. Google Earth Engine (2024). *Earth Engine Data Catalog: Sentinel-2 MSI*. https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED

---

*End of Phase 1 Report*
