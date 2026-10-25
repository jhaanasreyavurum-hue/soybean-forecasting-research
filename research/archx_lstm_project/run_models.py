"""Weekly soybean price forecasting: ARCHX-style volatility model and LSTM.

Targets: Indore and Gondal (Rajkot district), separately.
Oil is treated as USD/barrel only provisionally; verify original source/units before final reporting.
ARCH-X here means oil return enters the ARX conditional-mean equation, while ARCH models the residual
conditional variance. The Python arch package does not directly estimate a general oil-X variance equation.
"""
from pathlib import Path
import warnings, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)

INDORE=DATA/'price_indore.csv'; GONDAL=DATA/'price_gondal_rajkot.xlsx'; OIL=DATA/'crude_oil.csv'


def load_market(path, date_col, label):
    df = pd.read_csv(path) if path.suffix.lower()=='.csv' else pd.read_excel(path)
    df.columns=[str(c).strip() for c in df.columns]
    df[date_col]=pd.to_datetime(df[date_col], errors='coerce')
    df['p_modal']=pd.to_numeric(df['p_modal'], errors='coerce')
    df=df.dropna(subset=[date_col,'p_modal']).copy()
    df=df[(df['p_modal']>0)]
    # one price per date; mean duplicates only if any exist, and report this in audit
    dup=int(df.duplicated(date_col).sum())
    s=df.groupby(date_col)['p_modal'].mean().sort_index()
    s.name=label
    return s, {'series':label,'raw_rows':len(df),'duplicate_date_rows_aggregated':dup,
               'first_date':str(s.index.min().date()),'last_date':str(s.index.max().date()),
               'unique_daily_dates':len(s),'unit':'INR/quintal (assumed from Agmarknet modal price)'}

def load_oil():
    d=pd.read_csv(OIL); d.columns=[str(c).strip() for c in d.columns]
    if not {'Date','Price'}.issubset(d.columns): raise ValueError('Crude oil CSV must contain Date and Price columns.')
    d['Date']=pd.to_datetime(d['Date'],errors='coerce'); d['Price']=pd.to_numeric(d['Price'],errors='coerce')
    d=d.dropna(subset=['Date','Price']); d=d[d.Price>0]
    dup=int(d.duplicated('Date').sum()); s=d.groupby('Date').Price.mean().sort_index(); s.name='oil_price'
    return s, {'series':'crude_oil','raw_valid_rows':len(d),'duplicate_date_rows_aggregated':dup,
               'first_date':str(s.index.min().date()),'last_date':str(s.index.max().date()),
               'unique_daily_dates':len(s),'observed_min':float(s.min()),'observed_median':float(s.median()),
               'observed_max':float(s.max()),'unit':'UNVERIFIED; values resemble USD/barrel but must be confirmed from source'}

def make_weekly(market, oil):
    # W-FRI bins; only observed daily prices contribute. No forward-fill of soybean prices.
    weekly=pd.concat([market.resample('W-FRI').mean(),oil.resample('W-FRI').mean()],axis=1)
    weekly['market_logret']=np.log(weekly[market.name]).diff()
    weekly['oil_logret']=np.log(weekly['oil_price']).diff()
    # Oil is lagged by one complete week so no same-week/future oil information is used.
    weekly['oil_logret_lag1']=weekly['oil_logret'].shift(1)
    weekly=weekly.dropna(subset=[market.name,'oil_price','market_logret','oil_logret_lag1']).copy()
    return weekly

def metrics(y, pred):
    y=np.asarray(y,dtype=float); pred=np.asarray(pred,dtype=float)
    return {'RMSE':float(np.sqrt(mean_squared_error(y,pred))),
            'MAE':float(mean_absolute_error(y,pred)),
            'MAPE_percent':float(np.mean(np.abs((y-pred)/np.where(y==0,np.nan,y)))*100)}

def fit_archx(train, test, market_col):
    try:
        from arch import arch_model
    except ImportError as e:
        raise ImportError('Install arch: pip install arch') from e
    # Returns in percent; exogenous oil return is lagged one week. ARCH volatility with ARX mean.
    y=(train['market_logret']*100).dropna()
    x=train.loc[y.index,['oil_logret_lag1']]
    model=arch_model(y, x=x, mean='ARX', lags=1, vol='ARCH', p=1, dist='t', rescale=False)
    res=model.fit(disp='off',show_warning=False)
    preds=[]; actual=[]; dates=[]
    # fixed estimated parameters; each test point receives only known lagged oil return and prior observed return
    hist_y=(pd.concat([train['market_logret'],test['market_logret']])*100)
    all_x=pd.concat([train['oil_logret_lag1'],test['oil_logret_lag1']])
    start=len(train)
    for i in range(len(test)):
        pos=start+i
        # one-step mean forecast using fitted ARX; use actual past return and the lagged oil covariate
        prev=float(hist_y.iloc[pos-1])
        oilx=float(all_x.iloc[pos])
        params=res.params
        mu=float(params.get('Const',0)+params.get('market_logret[1]',params.get('y[1]',0))*prev+params.get('oil_logret_lag1',0)*oilx)
        # robustly use statsmodels-style forecast if parameter names differ
        if not np.isfinite(mu): mu=0.0
        preds.append(mu/100.0); actual.append(float(test['market_logret'].iloc[i])); dates.append(test.index[i])
    # convert predicted log return to price using prior week's actual price (rolling one-step-ahead)
    prices=[]
    for d, r in zip(dates,preds):
        prevdate=d-pd.Timedelta(days=7)
        prev=train[market_col].get(prevdate, np.nan)
        if pd.isna(prev):
            # W-FRI index should be contiguous here; use prior row from the full weekly frame
            prev=float(pd.concat([train[market_col],test[market_col]]).loc[:d].iloc[-2])
        prices.append(float(prev*np.exp(r)))
    return np.array(prices), {'model':'ARCHX (ARX mean + ARCH(1) variance)','converged':bool(res.convergence_flag==0),'n_train':len(train),'params':{str(k):float(v) for k,v in res.params.items()}}

def fit_lstm(train, test, market_col, use_oil, lookback=12, epochs=60):
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as e:
        raise ImportError('Install TensorFlow in Python 3.10–3.12: pip install tensorflow') from e
    # LSTM predicts next-week log return. For LSTM-X, oil return lagged one week is included.
    features=['market_logret'] + (['oil_logret_lag1'] if use_oil else [])
    allf=pd.concat([train,test])[features].copy()
    all_y=pd.concat([train,test])['market_logret'].copy()
    cut=len(train)
    sx=StandardScaler(); sx.fit(train[features].values)
    X=sx.transform(allf.values)
    # Scale target using training returns only
    sy=StandardScaler(); sy.fit(train[['market_logret']].values)
    yy=sy.transform(all_y.values.reshape(-1,1)).ravel()
    def seqs(end_exclusive):
        xx=[]; y=[]
        for i in range(lookback,end_exclusive): xx.append(X[i-lookback:i]); y.append(yy[i])
        return np.asarray(xx),np.asarray(y)
    Xtr,ytr=seqs(cut)
    if len(Xtr)<20: raise ValueError(f'Only {len(Xtr)} training sequences; not enough for LSTM.')
    model=Sequential([LSTM(32,input_shape=(lookback,len(features))),Dropout(.15),Dense(16,activation='relu'),Dense(1)])
    model.compile(optimizer='adam',loss='mse')
    model.fit(Xtr,ytr,epochs=epochs,batch_size=32,validation_split=.15,shuffle=False,
              callbacks=[EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)],verbose=0)
    predret=[]
    for i in range(cut,len(allf)):
        x=X[i-lookback:i].reshape(1,lookback,len(features))
        predret.append(float(sy.inverse_transform(model.predict(x,verbose=0).reshape(-1,1))[0,0]))
    # Rolling one-step prices: prior observed week's price * exp(predicted log-return)
    fullprice=pd.concat([train[market_col],test[market_col]])
    predprice=[]
    for j,r in enumerate(predret): predprice.append(float(fullprice.iloc[cut+j-1]*np.exp(r)))
    return np.asarray(predprice), {'model':'LSTM'+('-X (lagged crude oil)' if use_oil else ' (price-only)'),
                                    'lookback_weeks':lookback,'epochs_requested':epochs,'epochs_run':len(model.history.history['loss']),
                                    'features':features,'n_train_sequences':len(Xtr)}

def main():
    ind, ai=load_market(INDORE,'date','Indore')
    gon, ag=load_market(GONDAL,'Date','Gondal')
    oil, ao=load_oil()
    audit=[]
    for m in [ai,ag,ao]: audit.append(m)
    pd.DataFrame(audit).to_csv(OUT/'input_audit.csv',index=False)
    result_rows=[]; run_notes=[]
    for market in [ind,gon]:
        weekly=make_weekly(market,oil)
        # Return row represents change into this week; split chronologically, never random shuffle.
        cut=int(len(weekly)*0.8); train=weekly.iloc[:cut].copy(); test=weekly.iloc[cut:].copy()
        if len(train)<100 or len(test)<20: raise ValueError(f'Not enough aligned weekly rows for {market.name}: {len(weekly)}')
        weekly.to_csv(OUT/f'{market.name.lower()}_weekly_aligned.csv',index_label='week_ending')
        run_notes.append({'market':market.name,'weekly_rows_after_alignment':len(weekly),'first_week':str(weekly.index.min().date()),
                          'last_week':str(weekly.index.max().date()),'train_rows':len(train),'test_rows':len(test),
                          'split_date':str(test.index.min().date()),'oil_price_median':float(weekly.oil_price.median()),
                          'oil_unit_warning':'Oil units/source not independently verified; do not state INR conversion without FX data.'})
        # Naive benchmark: last observed weekly price
        naive=np.asarray([float(pd.concat([train[market.name],test[market.name]]).iloc[cut+i-1]) for i in range(len(test))])
        base={'market':market.name,'model':'Naive last-week price',**metrics(test[market.name],naive)}
        result_rows.append(base)
        try:
            p,meta=fit_archx(train,test,market.name)
            result_rows.append({'market':market.name,'model':meta['model'],**metrics(test[market.name],p)})
            pd.DataFrame({'date':test.index,'actual_price':test[market.name].values,'predicted_price':p}).to_csv(OUT/f'{market.name.lower()}_archx_predictions.csv',index=False)
            run_notes.append({'market':market.name,**meta})
        except Exception as e:
            run_notes.append({'market':market.name,'model':'ARCHX','error':str(e)})
        for use_oil in [False,True]:
            try:
                p,meta=fit_lstm(train,test,market.name,use_oil=use_oil)
                result_rows.append({'market':market.name,'model':meta['model'],**metrics(test[market.name],p)})
                tag='lstm_x' if use_oil else 'lstm'
                pd.DataFrame({'date':test.index,'actual_price':test[market.name].values,'predicted_price':p}).to_csv(OUT/f'{market.name.lower()}_{tag}_predictions.csv',index=False)
                run_notes.append({'market':market.name,**meta})
            except Exception as e:
                run_notes.append({'market':market.name,'model':'LSTM-X' if use_oil else 'LSTM','error':str(e)})
    pd.DataFrame(result_rows).to_csv(OUT/'out_of_sample_metrics.csv',index=False)
    (OUT/'run_notes.json').write_text(json.dumps(run_notes,indent=2,default=str))
    print('\nINPUT AUDIT'); print(pd.DataFrame(audit).to_string(index=False))
    print('\nOUT-OF-SAMPLE METRICS'); print(pd.DataFrame(result_rows).to_string(index=False))
    print('\nNotes written to',OUT/'run_notes.json')
if __name__=='__main__': main()
