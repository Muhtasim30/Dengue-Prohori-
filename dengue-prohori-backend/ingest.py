"""ingest_weather.py + ingest_dengue.py logic combined.

Weather: Open-Meteo (live + historical archive, free, no key).
Dengue:  Bangladesh sources have NO open API. Options shown below.
"""
import datetime as dt, requests, pandas as pd

# ---------------- WEATHER (real, works now) ----------------
def fetch_weather(lat, lon, past_days=90, forecast_days=16):
    u = ("https://api.open-meteo.com/v1/forecast"
         f"?latitude={lat}&longitude={lon}"
         "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
         "relative_humidity_2m_mean,wind_speed_10m_max,wind_direction_10m_dominant,sunshine_duration"
         f"&past_days={past_days}&forecast_days={forecast_days}&timezone=Asia%2FDhaka")
    d = requests.get(u, timeout=30).json()["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "tmean": [(a+b)/2 for a, b in zip(d["temperature_2m_max"], d["temperature_2m_min"])],
        "rain": d["precipitation_sum"],
        "humidity": d["relative_humidity_2m_mean"],
        "wind_speed": d["wind_speed_10m_max"],
        "wind_dir": d["wind_direction_10m_dominant"],
        "sunshine": [s/3600 if s else 0 for s in d.get("sunshine_duration", [0]*len(d["time"]))],
    })
    return df

def fetch_weather_archive(lat, lon, start, end):
    """Long history for training (ERA5). start/end: 'YYYY-MM-DD'."""
    u = ("https://archive-api.open-meteo.com/v1/archive"
         f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
         "&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean,"
         "wind_speed_10m_max,wind_direction_10m_dominant&timezone=Asia%2FDhaka")
    return requests.get(u, timeout=60).json()

def to_weekly(daily_df):
    return (daily_df.set_index("date").resample("W-MON")
            .agg({"tmean":"mean","rain":"sum","humidity":"mean",
                  "wind_speed":"mean","wind_dir":"mean","sunshine":"mean"})
            .reset_index())

# ---------------- DENGUE CASES (needs a pipeline) ----------------
# Bangladesh has no machine-readable dengue API. Realistic options:
#   A) DGHS "Daily Dengue Status" press releases / dashboard  -> scrape or OCR PDFs
#   B) IEDCR yearly situation reports                          -> manual/period import
#   C) WHO SEARO dengue situation updates                      -> periodic CSV/scrape
#   D) Your own community patient-case reports (this app)      -> real-time signal
# Below is a CSV importer; point it at whatever you can collect/digitise.
def load_dengue_csv(path):
    """CSV columns: area_id, week_start (YYYY-MM-DD), cases, source"""
    df = pd.read_csv(path, parse_dates=["week_start"])
    return df

def scrape_dghs_placeholder():
    """TODO: implement DGHS daily-report scraping/OCR.
    Steps: GET the daily PDF/HTML -> parse district table -> map district->area_id
    -> aggregate to weekly -> upsert dengue_cases. Cache raw files for audit."""
    raise NotImplementedError("Implement DGHS scraping for your district mapping.")

if __name__ == "__main__":
    df = to_weekly(fetch_weather(23.81, 90.41))
    print(df.tail())
