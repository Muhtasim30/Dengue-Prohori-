# Dengue Prohori — Backend & Data Pipeline

This kit turns the static `dengue-prohori.html` into a real, live, multi-user system:
a trained model, a database, daily data ingestion (weather + NDWI + dengue cases),
email early-warnings, community reports, and admin notices.

```
backend/
├─ requirements.txt        Python deps
├─ .env.example            copy to .env and fill in
├─ schema.sql              PostgreSQL + PostGIS tables
├─ features.py             feature engineering (shared by train + predict)
├─ train_model.py          train XGBoost on YOUR dataset → models/
├─ predict.py              load model → forecast + drivers
├─ ingest.py               Open-Meteo weather + dengue-CSV importer
├─ gee_ndwi.py             Google Earth Engine weekly NDWI / urban index
├─ main.py                 FastAPI API the website calls
├─ scheduler.py            nightly pipeline + alert trigger
├─ email_worker.py         SendGrid verification + alert emails
└─ frontend_integration.js how to point the HTML at the API
```

## 0. Prerequisites
- Python 3.11+, PostgreSQL 15+ with PostGIS, a domain + HTTPS, a SendGrid (or SES) account,
  a Google Cloud project with Earth Engine enabled.

## 1. Database
```bash
createdb prohori
psql -d prohori -f schema.sql
# then INSERT your areas (id,name,parent,lat,lon,urban,wlog) — same list as the HTML AREAS[]
```

## 2. Install & configure
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill DATABASE_URL, SENDGRID_API_KEY, GEE_*, ADMIN_TOKEN
```

## 3. Train the model (when you have your dataset)
Your CSV (weekly rows): `area_id,date,rain,humidity,temp,sunshine,wind_speed,wind_dir,ndwi,urban_index,population_index,cases`
```bash
python train_model.py --csv master_dataset.csv
# → models/model_h1..h4.joblib  (API)  + model_h1..h4.json (browser)  + metrics.json
```
- XGBoost trains for horizons +1..+4 weeks; `metrics.json` gives R²/RMSE/MAE for your thesis.
- No xgboost installed? It falls back to scikit-learn GradientBoosting automatically.
- **LSTM:** train a Keras model on the same features, then
  `tensorflowjs_converter --input_format keras model.h5 web_model/` and load it in the browser
  with `tf.loadLayersModel()`. Use it for the time-series headline; keep XGBoost for the why-card.

## 4. Run the API
```bash
uvicorn main:app --port 8000
# GET /forecast?area_id=mirpur&lat=23.82&lon=90.37
# GET /heatmap?parent=Dhaka   POST /subscribe   POST /report   GET /notices ...
```

## 5. Connect the website
Open `frontend_integration.js` and set `API`. In `dengue-prohori.html`, call `apiForecast(area)`
instead of `summaryFor(...)` (keep the local engine as offline fallback). The model's `drivers`
object feeds the existing "Why?" card and `/heatmap` drives the map — no UI changes needed.

## 6. Daily automation
```bash
python scheduler.py        # runs 06:00 Asia/Dhaka: ingest → predict → store → email alerts
```
Deploy as a systemd service or a Docker container with `restart: always`.

---

## Where each dataset comes from
| Data | Source | How | Frequency |
|---|---|---|---|
| Weather (temp, rain, humidity, wind, sun) | **Open-Meteo** (`ingest.py`) | free API, no key; archive API for history | daily |
| NDWI (surface water) | **Google Earth Engine** Landsat 8/9 (`gee_ndwi.py`) | service account, green/NIR bands | weekly |
| Urban index | **VIIRS nighttime lights** via GEE | `urban_index()` | monthly |
| Population | **WorldPop / LandScan** via GEE or download | join into `gee_indices` | yearly |
| Dengue cases | **DGHS daily reports, IEDCR, WHO SEARO** | no API — scrape/OCR PDFs or CSV import (`ingest.py`) | daily/weekly |
| Ground truth | **Community reports** (this app) | `/report`, `/case` → DB | real-time |

## Risk score & early-warning methodology
- Model predicts weekly cases for +1..+4 wk per area.
- `risk_pct = 0.45·MSI + 0.35·min(1, cases/800) + 0.20·seasonal_factor` (0–100).
- Bands: Low <147, Moderate 147–374, High 374–800, Extreme >800 (your thesis thresholds).
- Alert fires when an area's band **rises** vs. the last alert (stored in `subscriptions.last_band`),
  or when a 4-day short-horizon forecast crosses a threshold.

## Community report → model loop
1. Report saved `status=pending` with PostGIS point.
2. Validate: photo + geo sanity; **cluster** (N reports within radius/week auto-flags a hotspot).
3. Verified reports become a `report_density` feature feeding the next model run, and show as map hotspots.
4. Patient cases stay aggregated/consented; **never store raw NID** (only a salted hash + report in encrypted storage).

## Hosting (recommended)
- **Frontend:** any static host (Netlify, Vercel, GitHub Pages, Cloudflare Pages).
- **API + scheduler:** one small VM (or Render/Railway/Fly/Cloud Run) + managed Postgres (Supabase/Neon/RDS).
- **Object storage:** S3 / GCS for report photos (presigned uploads).
- Put the API behind HTTPS + rate limiting; restrict `/notices` POST with `ADMIN_TOKEN` and real auth.

## What's still TODO for you
1. Provide the training CSV → run `train_model.py`.
2. Implement DGHS dengue scraping/OCR for your district→area mapping (`scrape_dghs_placeholder`).
3. Create the GEE service account + key, fill `.env`.
4. Insert your `areas` rows, deploy API + scheduler, point the HTML at the API.
5. Add real admin authentication and object storage for photos.
6. (Optional) Train + export the LSTM to TensorFlow.js.
