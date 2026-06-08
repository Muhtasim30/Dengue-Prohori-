"""train_lstm.py — train the REAL LSTM half and export it for the browser.

Run where TensorFlow is available (Colab recommended):
    pip install tensorflow tensorflowjs pandas numpy scikit-learn openpyxl
    python train_lstm.py --xlsx preprocessed_master_dataset.xlsx

It learns from sequences (look-back window) of the 15 features and predicts
cases at t+h. Exports:
    models/lstm_h{h}.keras         (Keras model)
    web_model_h{h}/                (TensorFlow.js — load in the browser with tf.loadLayersModel)
    models/lstm_metrics.json
In the browser, ensemble = blend * xgboost_pred + (1-blend) * lstm_pred
(see frontend_integration.js).
"""
import argparse, json, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, Sequential

LOOKBACK = 8   # weeks of history fed to the LSTM
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

def windows(X, y, lookback, horizon):
    xs, ys = [], []
    for i in range(lookback, len(X)-horizon):
        xs.append(X[i-lookback:i]); ys.append(y[i+horizon])
    return np.array(xs), np.array(ys)

def main(path):
    os.makedirs('models', exist_ok=True)
    df = pd.read_excel(path); df.columns = [c.strip() for c in df.columns]; df = engineer(df)
    Xall = df[FEATURES].values.astype('float32'); yall = df['dengue cases'].values.astype('float32')
    sc = StandardScaler().fit(Xall); Xs = sc.transform(Xall)
    metrics = {}
    for h in (1, 2, 3, 4):
        Xw, yw = windows(Xs, yall, LOOKBACK, h)
        cut = int(len(Xw)*0.8)
        model = Sequential([layers.Input((LOOKBACK, len(FEATURES))),
                            layers.LSTM(48, return_sequences=False),
                            layers.Dense(24, activation='relu'),
                            layers.Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.fit(Xw[:cut], yw[:cut], validation_split=0.1, epochs=80, batch_size=16,
                  verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
        p = np.clip(model.predict(Xw[cut:], verbose=0).ravel(), 0, None)
        metrics[f'h{h}'] = {'R2': round(float(r2_score(yw[cut:], p)), 3),
                            'RMSE': round(float(np.sqrt(mean_squared_error(yw[cut:], p))), 1),
                            'MAE': round(float(mean_absolute_error(yw[cut:], p)), 1)}
        print(f'h{h}: {metrics[f"h{h}"]}')
        model.save(f'models/lstm_h{h}.keras')
        try:
            import tensorflowjs as tfjs; tfjs.converters.save_keras_model(model, f'web_model_h{h}')
        except Exception as e:
            print('tfjs export skipped:', e)
    json.dump({'lookback': LOOKBACK, 'scaler_mean': sc.mean_.tolist(),
               'scaler_scale': sc.scale_.tolist(), 'metrics': metrics},
              open('models/lstm_metrics.json', 'w'), indent=2)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--xlsx', required=True)
    main(ap.parse_args().xlsx)
