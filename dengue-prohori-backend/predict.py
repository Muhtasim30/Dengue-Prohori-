"""predict.py — load trained models and produce the forecast the API serves."""
import json, math, joblib, numpy as np, pandas as pd
from features import build_features, FEATURES

MED, HIGH, EXT = 147, 374, 800
_models = {}

def _load():
    if not _models:
        for h in (1, 2, 3, 4):
            try:
                _models[h] = joblib.load(f"models/model_h{h}.joblib")
            except FileNotFoundError:
                pass
    return _models

def band(v):
    return "ext" if v > EXT else "high" if v > HIGH else "mod" if v >= MED else "low"

def risk_pct(msi, wk1, month, monthly_max=330):
    seasonal = {1:45,2:28,3:88,4:72,5:155,6:270,7:330,8:315,9:198,10:72,11:25,12:8}[month]/monthly_max
    return round(min(1, 0.45*msi + 0.35*min(1, wk1/EXT) + 0.20*seasonal)*100)

def forecast_area(df_area: pd.DataFrame):
    """df_area: recent weekly rows for ONE area (>= 12 weeks). Returns dict."""
    models = _load()
    feats = build_features(df_area).sort_values("date")
    latest = feats.iloc[[-1]][FEATURES]
    msi = float(feats.iloc[-1]["msi"])
    month = pd.to_datetime(feats.iloc[-1]["date"]).month
    preds = {}
    for h in (1, 2, 3, 4):
        if h in models:
            m = models[h]["model"]
            yhat = max(0, float(m.predict(latest)[0]))
        else:  # fallback if model not trained yet
            base = {1:45,2:28,3:88,4:72,5:155,6:270,7:330,8:315,9:198,10:72,11:25,12:8}[month]/4.345
            yhat = base*(0.6+3.4*msi)
        hw = 0.25 + 0.06*h
        preds[h] = {"pred": round(yhat), "lo": round(max(0, yhat*(1-hw))), "hi": round(yhat*(1+hw))}
    wk1 = preds[1]["pred"]
    pct = risk_pct(msi, wk1, month)
    # drivers for the why-card (use feature importances if available)
    drivers = driver_explanations(feats.iloc[-1])
    return {"preds": preds, "wk1": wk1, "band": band(wk1), "risk_pct": pct,
            "msi": round(msi, 2), "drivers": drivers}

def driver_explanations(row):
    out = {}
    out["rainfall"] = round(float(row.get("rain_4w", 0)), 1)
    out["humidity"] = round(float(row.get("humidity", 0)))
    out["ndwi"] = round(float(row.get("ndwi", 0)), 3)
    out["temp"] = round(float(row.get("temp", 0)), 1)
    out["msi"] = round(float(row.get("msi", 0)), 2)
    return out
