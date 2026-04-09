-- ============================================================================
-- CoastalCred Database Schema — PostgreSQL 15 + PostGIS
-- ============================================================================
-- Blockchain-based blue carbon registry and MRV system for mangrove
-- ecosystems in India.
--
-- This is the design-only schema for Sem VI. Actual deployment and
-- service-level database separation will happen in Sem VII.
--
-- Services: Auth, Territory, DataIngestion, Verification,
--           CreditIssuance, Marketplace, BlockchainAdapter
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- ENUM TYPES
-- ============================================================================

CREATE TYPE user_role AS ENUM ('ngo', 'corporate', 'verifier', 'admin');

CREATE TYPE kyc_status AS ENUM ('pending', 'submitted', 'verified', 'rejected');

CREATE TYPE territory_status AS ENUM (
    'draft', 'submitted', 'active', 'suspended', 'retired'
);

CREATE TYPE project_type AS ENUM (
    'mangrove_restoration', 'mangrove_conservation', 'mangrove_afforestation'
);

CREATE TYPE document_type AS ENUM (
    'land_title', 'satellite_image', 'field_photo', 'government_permit',
    'planting_plan', 'baseline_report', 'monitoring_report', 'other'
);

CREATE TYPE job_status AS ENUM (
    'queued', 'running', 'completed', 'failed', 'cancelled'
);

CREATE TYPE verification_status AS ENUM (
    'pending', 'in_progress', 'completed', 'rejected', 'expired'
);

CREATE TYPE verification_decision AS ENUM (
    'approved', 'rejected', 'needs_revision'
);

CREATE TYPE credit_status AS ENUM (
    'pending', 'minted', 'active', 'retired', 'revoked'
);

CREATE TYPE listing_status AS ENUM (
    'active', 'paused', 'sold_out', 'cancelled'
);

CREATE TYPE escrow_status AS ENUM (
    'pending', 'funded', 'released', 'refunded', 'disputed'
);

CREATE TYPE escrow_event_type AS ENUM (
    'created', 'funded', 'released', 'refunded', 'disputed', 'resolved'
);

CREATE TYPE tx_status AS ENUM (
    'pending', 'submitted', 'confirmed', 'failed', 'reverted'
);

CREATE TYPE notification_type AS ENUM (
    'info', 'warning', 'success', 'error', 'action_required'
);

CREATE TYPE ingestion_job_type AS ENUM (
    'sentinel2_fetch', 'gmw_download', 'ndvi_compute', 'composite_build'
);


-- ============================================================================
-- AUTH SERVICE (3 tables)
-- ============================================================================

-- 1. users
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) NOT NULL UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,
    role            user_role    NOT NULL DEFAULT 'ngo',
    full_name       VARCHAR(255) NOT NULL,
    organization    VARCHAR(255),
    phone           VARCHAR(20),
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    kyc_status      kyc_status   NOT NULL DEFAULT 'pending',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_users_role ON users (role);
CREATE INDEX idx_users_kyc_status ON users (kyc_status);

-- 2. user_sessions
CREATE TABLE IF NOT EXISTS user_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash      VARCHAR(255) NOT NULL UNIQUE,
    ip_address      INET,
    user_agent      TEXT,
    expires_at      TIMESTAMPTZ  NOT NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_user_sessions_user_id ON user_sessions (user_id);
CREATE INDEX idx_user_sessions_token_hash ON user_sessions (token_hash);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions (expires_at);

-- 3. api_keys
CREATE TABLE IF NOT EXISTS api_keys (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash        VARCHAR(255) NOT NULL UNIQUE,
    label           VARCHAR(255) NOT NULL,
    scopes          TEXT[]       NOT NULL DEFAULT '{}',
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    last_used_at    TIMESTAMPTZ,
    expires_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_api_keys_user_id ON api_keys (user_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys (key_hash);


-- ============================================================================
-- TERRITORY SERVICE (4 tables)
-- ============================================================================

-- 4. territories
CREATE TABLE IF NOT EXISTS territories (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id            UUID             NOT NULL REFERENCES users(id),
    name                VARCHAR(255)     NOT NULL,
    description         TEXT,
    boundary            geometry(Polygon, 4326) NOT NULL,
    area_hectares       NUMERIC(12, 4)   NOT NULL CHECK (area_hectares > 0),
    project_type        project_type     NOT NULL,
    baseline_condition  TEXT,
    baseline_ndvi       NUMERIC(5, 4)    CHECK (baseline_ndvi BETWEEN -1 AND 1),
    project_start_date  DATE,
    status              territory_status NOT NULL DEFAULT 'draft',
    created_at          TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_territories_owner_id ON territories (owner_id);
CREATE INDEX idx_territories_status ON territories (status);
CREATE INDEX idx_territories_boundary ON territories USING GIST (boundary);

-- 5. territory_documents
CREATE TABLE IF NOT EXISTS territory_documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    territory_id    UUID          NOT NULL REFERENCES territories(id) ON DELETE CASCADE,
    document_type   document_type NOT NULL,
    file_hash       VARCHAR(128)  NOT NULL,
    ipfs_cid        VARCHAR(128),
    file_name       VARCHAR(255),
    file_size_bytes BIGINT        CHECK (file_size_bytes > 0),
    uploaded_by     UUID          NOT NULL REFERENCES users(id),
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_territory_documents_territory_id ON territory_documents (territory_id);
CREATE INDEX idx_territory_documents_uploaded_by ON territory_documents (uploaded_by);

-- 6. planting_records
CREATE TABLE IF NOT EXISTS planting_records (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    territory_id             UUID          NOT NULL REFERENCES territories(id) ON DELETE CASCADE,
    species                  VARCHAR(255)  NOT NULL,
    count                    INTEGER       NOT NULL CHECK (count > 0),
    planting_date            DATE          NOT NULL,
    worker_name              VARCHAR(255),
    worker_attestation_hash  VARCHAR(128),
    notes                    TEXT,
    created_at               TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_planting_records_territory_id ON planting_records (territory_id);
CREATE INDEX idx_planting_records_planting_date ON planting_records (planting_date);

-- 7. field_visits
CREATE TABLE IF NOT EXISTS field_visits (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    territory_id    UUID        NOT NULL REFERENCES territories(id) ON DELETE CASCADE,
    visitor_id      UUID        NOT NULL REFERENCES users(id),
    location        geometry(Point, 4326),
    visit_date      DATE        NOT NULL,
    notes           TEXT,
    photo_hashes    TEXT[]      NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_field_visits_territory_id ON field_visits (territory_id);
CREATE INDEX idx_field_visits_visitor_id ON field_visits (visitor_id);
CREATE INDEX idx_field_visits_location ON field_visits USING GIST (location);


-- ============================================================================
-- DATA INGESTION SERVICE (3 tables)
-- ============================================================================

-- 8. satellite_composites
CREATE TABLE IF NOT EXISTS satellite_composites (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    territory_id        UUID           REFERENCES territories(id),
    satellite           VARCHAR(50)    NOT NULL DEFAULT 'Sentinel-2',
    bands               TEXT[]         NOT NULL DEFAULT '{B2,B3,B4,B8,B11,B12}',
    acquisition_start   DATE           NOT NULL,
    acquisition_end     DATE           NOT NULL,
    cloud_cover_pct     NUMERIC(5, 2)  CHECK (cloud_cover_pct BETWEEN 0 AND 100),
    file_path           TEXT           NOT NULL,
    file_hash           VARCHAR(128),
    resolution_m        NUMERIC(6, 2)  NOT NULL DEFAULT 10.0,
    crs                 VARCHAR(20)    NOT NULL DEFAULT 'EPSG:4326',
    bounds              geometry(Polygon, 4326),
    created_at          TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ    NOT NULL DEFAULT NOW(),

    CHECK (acquisition_end >= acquisition_start)
);

CREATE INDEX idx_satellite_composites_territory_id ON satellite_composites (territory_id);
CREATE INDEX idx_satellite_composites_bounds ON satellite_composites USING GIST (bounds);

-- 9. ground_truth_datasets
CREATE TABLE IF NOT EXISTS ground_truth_datasets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name     VARCHAR(255) NOT NULL,
    version         VARCHAR(50)  NOT NULL,
    download_url    TEXT,
    file_path       TEXT,
    file_hash       VARCHAR(128),
    description     TEXT,
    license         VARCHAR(255),
    citation        TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 10. ingestion_jobs
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    territory_id    UUID              REFERENCES territories(id),
    job_type        ingestion_job_type NOT NULL,
    status          job_status        NOT NULL DEFAULT 'queued',
    parameters      JSONB,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    created_at      TIMESTAMPTZ       NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ       NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ingestion_jobs_territory_id ON ingestion_jobs (territory_id);
CREATE INDEX idx_ingestion_jobs_status ON ingestion_jobs (status);


-- ============================================================================
-- VERIFICATION SERVICE (3 tables)
-- ============================================================================

-- 11. verification_requests
CREATE TABLE IF NOT EXISTS verification_requests (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    territory_id            UUID                NOT NULL REFERENCES territories(id),
    requested_by            UUID                NOT NULL REFERENCES users(id),
    baseline_composite_id   UUID                REFERENCES satellite_composites(id),
    current_composite_id    UUID                REFERENCES satellite_composites(id),
    status                  verification_status NOT NULL DEFAULT 'pending',
    created_at              TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ         NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_verification_requests_territory_id ON verification_requests (territory_id);
CREATE INDEX idx_verification_requests_requested_by ON verification_requests (requested_by);
CREATE INDEX idx_verification_requests_status ON verification_requests (status);

-- 12. model_runs
CREATE TABLE IF NOT EXISTS model_runs (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_request_id  UUID           NOT NULL REFERENCES verification_requests(id) ON DELETE CASCADE,
    model_name               VARCHAR(100)   NOT NULL,
    model_version            VARCHAR(50)    NOT NULL,
    parameters               JSONB,
    baseline_hectares        NUMERIC(12, 4),
    current_hectares         NUMERIC(12, 4),
    delta_hectares           NUMERIC(12, 4),
    metrics                  JSONB,
    prediction_file_path     TEXT,
    started_at               TIMESTAMPTZ,
    completed_at             TIMESTAMPTZ,
    created_at               TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_runs_verification_request_id ON model_runs (verification_request_id);
CREATE INDEX idx_model_runs_model_name ON model_runs (model_name);

-- 13. verification_decisions
CREATE TABLE IF NOT EXISTS verification_decisions (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_request_id  UUID                  NOT NULL REFERENCES verification_requests(id),
    decided_by               UUID                  NOT NULL REFERENCES users(id),
    decision                 verification_decision NOT NULL,
    confidence_score         NUMERIC(5, 4)         CHECK (confidence_score BETWEEN 0 AND 1),
    notes                    TEXT,
    created_at               TIMESTAMPTZ           NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ           NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_verification_decisions_request_id ON verification_decisions (verification_request_id);
CREATE INDEX idx_verification_decisions_decided_by ON verification_decisions (decided_by);


-- ============================================================================
-- CREDIT ISSUANCE SERVICE (3 tables)
-- ============================================================================

-- 14. carbon_credit_batches
CREATE TABLE IF NOT EXISTS carbon_credit_batches (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    territory_id        UUID            NOT NULL REFERENCES territories(id),
    verification_id     UUID            NOT NULL REFERENCES verification_requests(id),
    token_id_erc1155    BIGINT          UNIQUE,
    amount_tco2e        NUMERIC(14, 4)  NOT NULL CHECK (amount_tco2e > 0),
    vintage_year        INTEGER         NOT NULL CHECK (vintage_year BETWEEN 2000 AND 2100),
    methodology         VARCHAR(100)    NOT NULL DEFAULT 'IPCC_TIER1',
    confidence_score    NUMERIC(5, 4)   CHECK (confidence_score BETWEEN 0 AND 1),
    status              credit_status   NOT NULL DEFAULT 'pending',
    minted_tx_hash      VARCHAR(128),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_carbon_credit_batches_territory_id ON carbon_credit_batches (territory_id);
CREATE INDEX idx_carbon_credit_batches_verification_id ON carbon_credit_batches (verification_id);
CREATE INDEX idx_carbon_credit_batches_status ON carbon_credit_batches (status);
CREATE INDEX idx_carbon_credit_batches_vintage_year ON carbon_credit_batches (vintage_year);

-- 15. credit_retirements
CREATE TABLE IF NOT EXISTS credit_retirements (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id            UUID           NOT NULL REFERENCES carbon_credit_batches(id),
    retired_by          UUID           NOT NULL REFERENCES users(id),
    amount_tco2e        NUMERIC(14, 4) NOT NULL CHECK (amount_tco2e > 0),
    reason              TEXT           NOT NULL,
    beneficiary_name    VARCHAR(255),
    retirement_tx_hash  VARCHAR(128),
    created_at          TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_credit_retirements_batch_id ON credit_retirements (batch_id);
CREATE INDEX idx_credit_retirements_retired_by ON credit_retirements (retired_by);

-- 16. issuance_rules
CREATE TABLE IF NOT EXISTS issuance_rules (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    methodology                 VARCHAR(100)   NOT NULL UNIQUE,
    min_confidence              NUMERIC(5, 4)  NOT NULL CHECK (min_confidence BETWEEN 0 AND 1),
    max_credits_per_hectare     NUMERIC(10, 4) NOT NULL CHECK (max_credits_per_hectare > 0),
    annual_sequestration_rate   NUMERIC(10, 4) NOT NULL CHECK (annual_sequestration_rate > 0),
    biomass_density_t_per_ha    NUMERIC(10, 4) NOT NULL DEFAULT 230.0,
    carbon_fraction             NUMERIC(5, 4)  NOT NULL DEFAULT 0.47,
    co2_to_c_ratio              NUMERIC(5, 4)  NOT NULL DEFAULT 3.67,
    is_active                   BOOLEAN        NOT NULL DEFAULT TRUE,
    created_at                  TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);


-- ============================================================================
-- MARKETPLACE SERVICE (3 tables)
-- ============================================================================

-- 17. marketplace_listings
CREATE TABLE IF NOT EXISTS marketplace_listings (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id            UUID            NOT NULL REFERENCES carbon_credit_batches(id),
    seller_id           UUID            NOT NULL REFERENCES users(id),
    price_per_credit    NUMERIC(14, 4)  NOT NULL CHECK (price_per_credit > 0),
    currency            VARCHAR(10)     NOT NULL DEFAULT 'INR',
    quantity_available  NUMERIC(14, 4)  NOT NULL CHECK (quantity_available >= 0),
    status              listing_status  NOT NULL DEFAULT 'active',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_marketplace_listings_batch_id ON marketplace_listings (batch_id);
CREATE INDEX idx_marketplace_listings_seller_id ON marketplace_listings (seller_id);
CREATE INDEX idx_marketplace_listings_status ON marketplace_listings (status);

-- 18. marketplace_orders
CREATE TABLE IF NOT EXISTS marketplace_orders (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    listing_id      UUID            NOT NULL REFERENCES marketplace_listings(id),
    buyer_id        UUID            NOT NULL REFERENCES users(id),
    quantity        NUMERIC(14, 4)  NOT NULL CHECK (quantity > 0),
    total_price     NUMERIC(14, 4)  NOT NULL CHECK (total_price > 0),
    escrow_status   escrow_status   NOT NULL DEFAULT 'pending',
    order_tx_hash   VARCHAR(128),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_marketplace_orders_listing_id ON marketplace_orders (listing_id);
CREATE INDEX idx_marketplace_orders_buyer_id ON marketplace_orders (buyer_id);
CREATE INDEX idx_marketplace_orders_escrow_status ON marketplace_orders (escrow_status);

-- 19. escrow_events
CREATE TABLE IF NOT EXISTS escrow_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id        UUID              NOT NULL REFERENCES marketplace_orders(id) ON DELETE CASCADE,
    event_type      escrow_event_type NOT NULL,
    amount          NUMERIC(14, 4)    NOT NULL CHECK (amount >= 0),
    tx_hash         VARCHAR(128),
    created_at      TIMESTAMPTZ       NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ       NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_escrow_events_order_id ON escrow_events (order_id);


-- ============================================================================
-- BLOCKCHAIN ADAPTER SERVICE (2 tables)
-- ============================================================================

-- 20. blockchain_transactions
CREATE TABLE IF NOT EXISTS blockchain_transactions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tx_hash             VARCHAR(128) UNIQUE,
    contract_address    VARCHAR(42)  NOT NULL,
    function_name       VARCHAR(100) NOT NULL,
    from_address        VARCHAR(42)  NOT NULL,
    to_address          VARCHAR(42),
    parameters          JSONB,
    status              tx_status    NOT NULL DEFAULT 'pending',
    block_number        BIGINT,
    gas_used            BIGINT,
    network             VARCHAR(50)  NOT NULL DEFAULT 'polygon_amoy',
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    confirmed_at        TIMESTAMPTZ
);

CREATE INDEX idx_blockchain_transactions_tx_hash ON blockchain_transactions (tx_hash);
CREATE INDEX idx_blockchain_transactions_status ON blockchain_transactions (status);
CREATE INDEX idx_blockchain_transactions_contract ON blockchain_transactions (contract_address);

-- 21. contract_registry
CREATE TABLE IF NOT EXISTS contract_registry (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_name       VARCHAR(100) NOT NULL,
    contract_address    VARCHAR(42)  NOT NULL,
    network             VARCHAR(50)  NOT NULL DEFAULT 'polygon_amoy',
    abi_hash            VARCHAR(128) NOT NULL,
    deployed_at         TIMESTAMPTZ,
    is_active           BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    UNIQUE (contract_name, network)
);

CREATE INDEX idx_contract_registry_network ON contract_registry (network);


-- ============================================================================
-- SHARED TABLES (2 tables)
-- ============================================================================

-- 22. audit_log — IMMUTABLE: application must never UPDATE or DELETE rows
CREATE TABLE IF NOT EXISTS audit_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID         REFERENCES users(id),
    action          VARCHAR(100) NOT NULL,
    entity_type     VARCHAR(100) NOT NULL,
    entity_id       UUID,
    old_value       JSONB,
    new_value       JSONB,
    ip_address      INET,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
    -- No updated_at: this table is append-only
);

CREATE INDEX idx_audit_log_user_id ON audit_log (user_id);
CREATE INDEX idx_audit_log_entity ON audit_log (entity_type, entity_id);
CREATE INDEX idx_audit_log_created_at ON audit_log (created_at);

-- 23. notifications
CREATE TABLE IF NOT EXISTS notifications (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID              NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type            notification_type NOT NULL DEFAULT 'info',
    title           VARCHAR(255)      NOT NULL,
    message         TEXT              NOT NULL,
    is_read         BOOLEAN           NOT NULL DEFAULT FALSE,
    entity_type     VARCHAR(100),
    entity_id       UUID,
    created_at      TIMESTAMPTZ       NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ       NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_notifications_user_id ON notifications (user_id);
CREATE INDEX idx_notifications_is_read ON notifications (user_id, is_read);


-- ============================================================================
-- SEED DATA: Default issuance rule (IPCC Tier 1)
-- ============================================================================

INSERT INTO issuance_rules (
    methodology,
    min_confidence,
    max_credits_per_hectare,
    annual_sequestration_rate,
    biomass_density_t_per_ha,
    carbon_fraction,
    co2_to_c_ratio
) VALUES (
    'IPCC_TIER1',
    0.70,
    7.0,       -- tCO2e per hectare per year (IPCC flux rate)
    7.0,       -- annual sequestration rate tCO2e/ha/yr
    230.0,     -- biomass density t/ha (IPCC Wetlands Supplement 2013)
    0.47,      -- carbon fraction of dry biomass
    3.6667     -- 44/12 CO2:C molecular ratio
) ON CONFLICT (methodology) DO NOTHING;
