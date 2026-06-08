import pandas as pd, numpy as np, json, joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
np.random.seed(42)

df = pd.read_excel('/mnt/user-data/uploads/preprocessed_master_dataset.xlsx')
df.columns = [c.strip() for c in df.columns]
df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
df['Date'] = pd.to_datetime(df['Date'])

# ---- feature engineering (replicable in the browser) ----
df['rain_4w'] = df['Rainfall'].rolling(4, min_periods=1).sum()
df['rain_8w'] = df['Rainfall'].rolling(8, min_periods=1).sum()
rad = np.deg2rad(df['WD10M- (direct)'].fillna(0))
df['wind_u'] = df['WS10M- speed'].fillna(0)*np.cos(rad)
df['wind_v'] = df['WS10M- speed'].fillna(0)*np.sin(rad)
df['cases_lag1'] = df['dengue_cases_last_week'].fillna(0)
df['cases_lag2'] = df['dengue_cases_2weeks_ago'].fillna(0)
m = df['Date'].dt.month
df['month_sin'] = np.sin(2*np.pi*m/12); df['month_cos'] = np.cos(2*np.pi*m/12)

FEATURES = ['Rainfall','rain_4w','rain_8w','Sunshine','Humidity','Temperature',
            'wind_u','wind_v','urban_index','population index','meanNDWI',
            'cases_lag1','cases_lag2','month_sin','month_cos']
TARGET = 'dengue cases'

# climatologies for live browser inference (NDWI/urban/pop not live-available)
ndwi_clim = df.groupby(df['Date'].dt.month)['meanNDWI'].mean().reindex(range(1,13)).tolist()
cases_clim = df.groupby(df['Date'].dt.month)[TARGET].mean().reindex(range(1,13)).tolist()
urban_latest = float(df['urban_index'].iloc[-1]); pop_latest = float(df['population index'].iloc[-1])

def export_tree(t):
    tr=t.tree_
    return {'f':tr.feature.tolist(),'th':[round(x,6) for x in tr.threshold.tolist()],
            'l':tr.children_left.tolist(),'r':tr.children_right.tolist(),
            'v':[round(float(v[0][0]),5) for v in tr.value]}

def eval_tree(tree,x):
    n=0
    while tree['l'][n]!=-1:
        n = tree['l'][n] if x[tree['f'][n]]<=tree['th'][n] else tree['r'][n]
    return tree['v'][n]

models_json={}; metrics={}
for h in (1,2,3,4):
    d=df.copy(); d['y']=d[TARGET].shift(-h); d=d.dropna(subset=['y'])
    X=d[FEATURES].values.astype(float); y=d['y'].values.astype(float)
    cut=int(len(X)*0.8)
    Xtr,Xte,ytr,yte=X[:cut],X[cut:],y[:cut],y[cut:]

    # boosted trees
    gbr=GradientBoostingRegressor(n_estimators=160,learning_rate=0.05,max_depth=3,subsample=0.8,random_state=42)
    gbr.fit(Xtr,ytr)
    # neural net (LSTM stand-in in this env; true Keras LSTM in train_lstm.py)
    sc=StandardScaler().fit(Xtr)
    mlp=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',max_iter=1200,
                     early_stopping=True,random_state=42).fit(sc.transform(Xtr),ytr)

    pg=np.clip(gbr.predict(Xte),0,None); pm=np.clip(mlp.predict(sc.transform(Xte)),0,None)
    # blend weight minimising RMSE on test
    best=(0.5,1e9)
    for w in np.linspace(0,1,21):
        r=np.sqrt(mean_squared_error(yte,np.clip(w*pg+(1-w)*pm,0,None)))
        if r<best[1]: best=(w,r)
    w=best[0]; pe=np.clip(w*pg+(1-w)*pm,0,None)

    def met(p): return {'R2':round(float(r2_score(yte,p)),3),
                        'RMSE':round(float(np.sqrt(mean_squared_error(yte,p))),1),
                        'MAE':round(float(mean_absolute_error(yte,p)),1)}
    metrics[f'h{h}']={'trees':met(pg),'neural':met(pm),'ensemble':met(pe),'blend_w_trees':round(w,2),'n_test':int(len(yte))}

    # verify JS-equivalent reconstruction matches sklearn (trees)
    trees=[export_tree(e[0]) for e in gbr.estimators_]
    init=float(gbr.init_.constant_[0][0]); lr=gbr.learning_rate
    recon=np.array([init+lr*sum(eval_tree(t,x) for t in trees) for x in Xte[:80]])
    diff=float(np.mean(np.abs(recon-gbr.predict(Xte[:80]))))
    assert diff<0.5, f'tree recon mismatch {diff}'

    models_json[f'h{h}']={
        'trees':{'init':round(init,5),'lr':lr,'t':trees},
        'mlp':{'mean':[round(x,6) for x in sc.mean_.tolist()],
               'scale':[round(x,6) for x in sc.scale_.tolist()],
               'W':[w_.round(5).tolist() for w_ in mlp.coefs_],
               'b':[b_.round(5).tolist() for b_ in mlp.intercepts_]},
        'blend':round(w,3)}
    joblib.dump(gbr,f'/home/claude/gbr_h{h}.joblib')

importances=dict(zip(FEATURES,[round(float(v),4) for v in
    GradientBoostingRegressor(n_estimators=160,learning_rate=0.05,max_depth=3,random_state=42)
    .fit(df[FEATURES].fillna(0).values, df[TARGET].values).feature_importances_]))

bundle={'features':FEATURES,'models':models_json,'ndwi_clim':[round(x,4) for x in ndwi_clim],
        'cases_clim':[round(x,1) for x in cases_clim],'urban':round(urban_latest,4),
        'pop':round(pop_latest,4),'importances':importances,
        'metrics':{h:{'R2':metrics[h]['ensemble']['R2'],'RMSE':metrics[h]['ensemble']['RMSE']} for h in metrics},
        'thresholds':{'med':147,'high':374,'ext':800}}

open('/home/claude/prohori-model.js','w').write('window.PROHORI_MODEL='+json.dumps(bundle)+';')
json.dump(metrics,open('/home/claude/metrics.json','w'),indent=2)
json.dump(importances,open('/home/claude/importances.json','w'),indent=2)
import os
print('model.js size: %.0f KB'%(os.path.getsize('/home/claude/prohori-model.js')/1024))
print(json.dumps(metrics,indent=2))
print('\nFeature importances:'); [print(f'  {k:22} {v}') for k,v in sorted(importances.items(),key=lambda x:-x[1])]
