"""train_model.py — train the dengue forecaster on YOUR dataset.

Usage:
    python train_model.py --csv master_dataset.csv

Your CSV needs (weekly rows): area_id, date, rain, humidity, temp, sunshine,
    wind_speed, wind_dir, ndwi, urban_index, population_index, cases
(If you only have Dhaka, set area_id='dhaka' for every row.)

Outputs:
    models/model_h1.json ... model_h4.json   -> XGBoost trees (also for the browser)
    models/model_h1.joblib ...               -> for the FastAPI backend
    models/metrics.json                       -> R2 / RMSE / MAE per horizon
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from features import make_supervised, FEATURES

os.makedirs("models", exist_ok=True)

def train(csv):
    df = pd.read_csv(csv, parse_dates=["date"])
    metrics = {}
    try:
        import xgboost as xgb
        have_xgb = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        have_xgb = False
        print("xgboost not installed -> using sklearn GradientBoosting fallback")

    for h in (1, 2, 3, 4):
        X, y, _ = make_supervised(df, horizon=h)
        # temporal split (no shuffle) — last 20% is the test set
        cut = int(len(X) * 0.8)
        Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

        if have_xgb:
            model = xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror")
            model.fit(Xtr, ytr)
            model.save_model(f"models/model_h{h}.json")        # browser-loadable trees
        else:
            model = GradientBoostingRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=3)
            model.fit(Xtr, ytr)

        import joblib; joblib.dump({"model": model, "features": FEATURES},
                                   f"models/model_h{h}.joblib")
        pred = np.clip(model.predict(Xte), 0, None)
        metrics[f"h{h}"] = {
            "R2": round(float(r2_score(yte, pred)), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(yte, pred))), 2),
            "MAE": round(float(mean_absolute_error(yte, pred)), 2),
            "n_test": int(len(yte)),
        }
        print(f"horizon {h}: {metrics[f'h{h}']}")

    json.dump(metrics, open("models/metrics.json", "w"), indent=2)
    print("\nSaved models/ and metrics.json")
    print("Feature importances (h1) tell you what drives risk — use for the why-card.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    train(ap.parse_args().csv)
