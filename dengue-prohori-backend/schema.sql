-- Dengue Prohori — database schema (PostgreSQL + PostGIS)
-- Run:  psql -d prohori -f schema.sql
CREATE EXTENSION IF NOT EXISTS postgis;

-- Areas (thanas / districts) with vulnerability factors
CREATE TABLE IF NOT EXISTS areas (
  id           TEXT PRIMARY KEY,          -- 'mirpur', 'dhaka'
  name         TEXT NOT NULL,
  name_bn      TEXT,
  parent       TEXT NOT NULL,             -- 'Dhaka'
  lat          DOUBLE PRECISION NOT NULL,
  lon          DOUBLE PRECISION NOT NULL,
  urban        REAL DEFAULT 0.7,          -- density factor 0..1
  wlog         REAL DEFAULT 0.5,          -- waterlogging tendency 0..1
  geom         GEOGRAPHY(POINT,4326)
);

-- Daily weather per area (from Open-Meteo / BMD)
CREATE TABLE IF NOT EXISTS weather_daily (
  area_id      TEXT REFERENCES areas(id),
  date         DATE NOT NULL,
  tmean        REAL, rain REAL, humidity REAL,
  wind_speed   REAL, wind_dir REAL, sunshine REAL,
  PRIMARY KEY (area_id, date)
);

-- Weekly NDWI / urban index from Google Earth Engine
CREATE TABLE IF NOT EXISTS gee_indices (
  area_id      TEXT REFERENCES areas(id),
  week_start   DATE NOT NULL,
  ndwi         REAL, urban_index REAL, population REAL,
  PRIMARY KEY (area_id, week_start)
);

-- Official dengue cases (DGHS / IEDCR / WHO)
CREATE TABLE IF NOT EXISTS dengue_cases (
  area_id      TEXT REFERENCES areas(id),
  week_start   DATE NOT NULL,
  cases        INTEGER,
  source       TEXT,                       -- 'DGHS','IEDCR','WHO'
  PRIMARY KEY (area_id, week_start)
);

-- Model output: forecasts produced by the daily job
CREATE TABLE IF NOT EXISTS forecasts (
  area_id      TEXT REFERENCES areas(id),
  run_date     DATE NOT NULL,
  horizon_wk   SMALLINT NOT NULL,          -- 1..4
  pred         REAL, lo REAL, hi REAL,
  risk_pct     REAL, band TEXT,            -- low/mod/high/ext
  drivers      JSONB,                      -- {rain:.., ndwi:.., ...} for the why-card
  PRIMARY KEY (area_id, run_date, horizon_wk)
);

-- Email subscriptions (double opt-in)
CREATE TABLE IF NOT EXISTS subscriptions (
  id           BIGSERIAL PRIMARY KEY,
  email        TEXT NOT NULL,
  area_id      TEXT REFERENCES areas(id),
  verified     BOOLEAN DEFAULT FALSE,
  token        TEXT,                        -- verify / unsubscribe token
  last_band    TEXT,                        -- to avoid duplicate alerts
  created_at   TIMESTAMPTZ DEFAULT now(),
  UNIQUE (email, area_id)
);

-- Community breeding-spot reports
CREATE TABLE IF NOT EXISTS reports (
  id           BIGSERIAL PRIMARY KEY,
  type         TEXT,                        -- water/drain/construction/container
  description  TEXT,
  address      TEXT,
  lat          DOUBLE PRECISION, lon DOUBLE PRECISION,
  geom         GEOGRAPHY(POINT,4326),
  photo_url    TEXT,                        -- object-storage key (S3/GCS)
  status       TEXT DEFAULT 'pending',      -- pending/verified/rejected
  area_id      TEXT,
  created_at   TIMESTAMPTZ DEFAULT now()
);

-- Patient cases (sensitive — store minimally, consented, access-controlled)
CREATE TABLE IF NOT EXISTS patient_cases (
  id           BIGSERIAL PRIMARY KEY,
  area_id      TEXT REFERENCES areas(id),
  method       TEXT, status TEXT,           -- NS1.. ; home/hospitalised/ICU
  nid_hash     TEXT,                         -- HASH only, never raw NID
  report_url   TEXT,                         -- encrypted object-storage key
  consented    BOOLEAN DEFAULT FALSE,
  created_at   TIMESTAMPTZ DEFAULT now()
);

-- Admin notices
CREATE TABLE IF NOT EXISTS notices (
  id           BIGSERIAL PRIMARY KEY,
  level        TEXT,                         -- info/warning/urgent
  title        TEXT, body TEXT,
  active       BOOLEAN DEFAULT TRUE,
  created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reports_geom ON reports USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_forecasts_run ON forecasts (run_date);
