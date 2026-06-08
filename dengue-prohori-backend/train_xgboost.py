"""train_xgboost.py — train the REAL XGBoost half of the ensemble.

Run in Colab or locally where xgboost is installed:
    pip install xgboost pandas numpy scikit-learn openpyxl joblib
    python train_xgboost.py --xlsx preprocessed_master_dataset.xlsx

Outputs: models/xgb_h1..h4.json (native) + xgb_h1..h4.joblib + xgb_metrics.json
Uses the SAME 15 features as the website so it drops straight into export_browser.py.
"""
import argparse, json, os, numpy as np, pandas as pd
import xgboost as xgb, joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

FEATURES = ['Rainfall','rain_4w','rain_8w','Sunshine','Humidity','Temperature',
            'wind_u','wind_v','urban_index','population index','meanNDWI',
            'cases_lag1','cases_lag2','month_sin','month_cos']

def engineer(df):
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['rain_4w'] = df['Rainfall'].rolling(4, min_periods=1).sum()
    df['rain_8w'] = df['Rainfall'].rolling(8, min_periods=1).sum()
    rad = np.deg2rad(df['WD10M- (direct)'].fillna(0))
    df['wind_u'] = df['WS10M- speed'].fillna(0)*np.cos(rad)
    df['wind_v'] = df['WS10M- speed'].fillna(0)*np.sin(rad)
    df['cases_lag1'] = df['dengue_cases_last_week'].fillna(0)
    df['cases_lag2'] = df['dengue_cases_2weeks_ago'].fillna(0)
    m = df['Date'].dt.month
    df['month_sin'] = np.sin(2*np.pi*m/12); df['month_cos'] = np.cos(2*np.pi*m/12)
    return df

def main(path):
    os.makedirs('models', exist_ok=True)
    df = engineer(pd.read_excel(path)); df.columns = [c.strip() for c in df.columns]
    df = engineer(df)
    metrics = {}
    for h in (1, 2, 3, 4):
        d = df.copy(); d['y'] = d['dengue cases'].shift(-h); d = d.dropna(subset=['y'])
        X, y = d[FEATURES].values, d['y'].values
        cut = int(len(X)*0.8)
        model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8,
                                 objective='reg:squarederror', random_state=42)
        model.fit(X[:cut], y[:cut])
        model.save_model(f'models/xgb_h{h}.json')
        joblib.dump(model, f'models/xgb_h{h}.joblib')
        p = np.clip(model.predict(X[cut:]), 0, None)
        metrics[f'h{h}'] = {'R2': round(float(r2_score(y[cut:], p)), 3),
                            'RMSE': round(float(np.sqrt(mean_squared_error(y[cut:], p))), 1),
                            'MAE': round(float(mean_absolute_error(y[cut:], p)), 1)}
        print(f'h{h}: {metrics[f"h{h}"]}')
    json.dump(metrics, open('models/xgb_metrics.json', 'w'), indent=2)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--xlsx', required=True)
    main(ap.parse_args().xlsx)
