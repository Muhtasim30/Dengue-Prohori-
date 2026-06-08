"""features.py — feature engineering shared by training and prediction.
Keeps the SAME logic on both sides so the model sees identical inputs.
"""
import numpy as np
import pandas as pd

# Feature columns the model is trained on (order matters for some exporters)
FEATURES = [
    "rain_1w", "rain_2w", "rain_4w", "rain_8w",      # lagged rainfall
    "humidity", "temp", "sunshine",
    "wind_u", "wind_v",                               # wind as vector
    "ndwi", "urban_index", "population_index",
    "msi", "mobility_proxy",
    "cases_lag1", "cases_lag2",                        # autoregressive
    "month_sin", "month_cos",                          # seasonality
]
TARGET = "cases"   # weekly dengue cases (the model predicts t+k)

def _roll(series, weeks):
    return series.rolling(weeks, min_periods=1).sum()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: weekly rows per area with columns:
       date, rain, humidity, temp, sunshine, wind_speed, wind_dir,
       ndwi, urban_index, population_index, cases
    Returns df with engineered FEATURES + TARGET, per area.
    """
    df = df.sort_values(["area_id", "date"]).copy()
    out = []
    for area, g in df.groupby("area_id"):
        g = g.copy()
        g["rain_1w"] = g["rain"]
        g["rain_2w"] = _roll(g["rain"], 2)
        g["rain_4w"] = _roll(g["rain"], 4)
        g["rain_8w"] = _roll(g["rain"], 8)
        # wind vector
        rad = np.deg2rad(g["wind_dir"].fillna(0))
        g["wind_u"] = g["wind_speed"].fillna(0) * np.cos(rad)
        g["wind_v"] = g["wind_speed"].fillna(0) * np.sin(rad)
        # Mosquito Suitability Index (normalised blend — thesis weights)
        def n(x):
            x = x.astype(float)
            rng = (x.max() - x.min()) or 1
            return (x - x.min()) / rng
        g["msi"] = (0.40*n(g["rain_4w"]) + 0.30*n(g["ndwi"]) +
                    0.12*n(g["humidity"]) + 0.10*n(g["temp"]) +
                    0.08*n(g["urban_index"])).clip(0, 1)
        g["mobility_proxy"] = (n(g["urban_index"]) * n(g["population_index"])).clip(0, 1)
        g["cases_lag1"] = g["cases"].shift(1).fillna(0)
        g["cases_lag2"] = g["cases"].shift(2).fillna(0)
        m = pd.to_datetime(g["date"]).dt.month
        g["month_sin"] = np.sin(2*np.pi*m/12)
        g["month_cos"] = np.cos(2*np.pi*m/12)
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    return res.fillna(0)

def make_supervised(df: pd.DataFrame, horizon: int = 1):
    """Target = cases at t+horizon (per area). Returns X, y aligned."""
    feats = build_features(df)
    rows = []
    for area, g in feats.groupby("area_id"):
        g = g.sort_values("date").copy()
        g["y"] = g[TARGET].shift(-horizon)
        rows.append(g)
    full = pd.concat(rows, ignore_index=True).dropna(subset=["y"])
    return full[FEATURES], full["y"], full
