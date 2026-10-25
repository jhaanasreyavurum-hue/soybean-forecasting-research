"""Rebuild the small data files the dashboard reads.
Scorecard numbers are copied from the executed notebooks (see NOTES in each row).
Monthly prices are recomputed from the raw price files if you pass their folder:
    python scripts/build_data.py path/to/raw   # needs price_indore.csv, price_gondal_rajkot.xlsx
"""
import sys, pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data"

# (track, market, model, rmse, mae, mape, naive_rmse, source notebook); naive MAE/MAPE added below
ROWS = [
 # Daily, 1-step rolling  -- ARIMA/SARIMA notebook
 ("Daily · ARIMA family","Indore","ARIMA(2,1,3)",383.589,213.422,5.557,438.746,"Soybean_ARIMA_SARIMA_Indore_Gondal"),
 ("Daily · ARIMA family","Indore","SARIMA(2,1,2)x(1,0,1,6)",384.210,212.074,5.528,438.746,"Soybean_ARIMA_SARIMA_Indore_Gondal"),
 ("Daily · ARIMA family","Gondal","ARIMA(0,1,2)",100.202,62.372,1.459,106.179,"Soybean_ARIMA_SARIMA_Indore_Gondal"),
 # Daily, ML notebooks
 ("Daily · ML regressors","Indore","Linear Regression",375.336,206.755,None,438.433,"soybean_price_prediction_indore"),
 ("Daily · ML regressors","Indore","Random Forest",379.655,224.639,None,438.433,"soybean_price_prediction_indore"),
 ("Daily · ML regressors","Indore","XGBoost",380.510,225.905,None,438.433,"soybean_price_prediction_indore"),
 ("Daily · ML regressors","Gondal","Linear Regression",119.857,80.240,None,107.158,"soybean_price_prediction_rajkot"),
 ("Daily · ML regressors","Gondal","Random Forest",115.382,81.137,None,107.158,"soybean_price_prediction_rajkot"),
 ("Daily · ML regressors","Gondal","XGBoost",127.849,91.914,None,107.158,"soybean_price_prediction_rajkot"),
 # Monthly, Indore only -- ARIMAX notebook
 ("Monthly · ARIMAX / VAR","Indore","VAR (levels)",284.4,228.4,5.2,230.2,"soybean_arimax_sarimax_executed"),
 ("Monthly · ARIMAX / VAR","Indore","ARIMAX",289.4,223.3,5.1,230.2,"soybean_arimax_sarimax_executed"),
 ("Monthly · ARIMAX / VAR","Indore","VECM",292.1,240.9,5.5,230.2,"soybean_arimax_sarimax_executed"),
 ("Monthly · ARIMAX / VAR","Indore","VAR (differenced)",300.7,245.7,5.6,230.2,"soybean_arimax_sarimax_executed"),
 ("Monthly · ARIMAX / VAR","Indore","SARIMAX",348.8,289.4,6.6,230.2,"soybean_arimax_sarimax_executed"),
]
cols = ["track","market","model","rmse","mae","mape","naive_rmse","source"]
df = pd.DataFrame(ROWS, columns=cols)
NAIVE = {'Daily · ARIMA family': {'Indore': (223.581, 5.695), 'Gondal': (56.299, 1.314)}, 'Daily · ML regressors': {'Indore': (222.76, None), 'Gondal': (57.22, None)}, 'Monthly · ARIMAX / VAR': {'Indore': (172.0, 3.9)}}
df["naive_mae"] = [NAIVE[t][m][0] for t,m in zip(df.track, df.market)]
df["naive_mape"] = [NAIVE[t][m][1] for t,m in zip(df.track, df.market)]

# Weekly ARCHX / LSTM -- computed from the metrics file shipped with that project
w = pd.read_csv(OUT/"weekly_metrics.csv")
nv = w[w.model.str.startswith("Naive")].set_index("market")
wk = w[~w.model.str.startswith("Naive")].copy()
wk["track"]="Weekly · ARCHX / LSTM"; wk["naive_rmse"]=wk.market.map(nv.RMSE); wk["naive_mae"]=wk.market.map(nv.MAE); wk["naive_mape"]=wk.market.map(nv.MAPE_percent)
wk = wk.rename(columns={"RMSE":"rmse","MAE":"mae","MAPE_percent":"mape"})
wk["source"]="soybean_archx_lstm_project"
wk["model"]=wk.model.str.replace(r" \(.*\)","",regex=True).replace({"LSTM":"LSTM (price-only)"})
df = pd.concat([df, wk[cols+["naive_mae","naive_mape"]]], ignore_index=True)
df.to_csv(OUT/"scorecard.csv", index=False)

# Harvest-window storage gain (ARIMAX notebook, Cell 21): Oct-Dec avg vs next Feb-Apr avg, Indore
sg = pd.DataFrame({"harvest_year":range(2011,2025),
 "oct_dec":[2100.5,3141.6,3435.4,3034.1,3609.7,2886.7,2811.9,3164.2,3774.9,3489.8,5574.3,5015.7,4623.1,3993.0],
 "next_feb_apr":[2717.8,3471.1,3981.9,3291.3,3687.3,2879.8,3576.7,3664.4,3696.5,5085.9,7129.9,5296.1,4390.3,4173.3],
 "gain_pct":[29.4,10.5,15.9,8.5,2.2,-0.2,27.2,15.8,-2.1,45.7,27.9,5.6,-5.0,4.5]})
sg.to_csv(OUT/"storage_gain.csv", index=False)

# Monthly prices + MSP (needs raw files)
if len(sys.argv) > 1:
    raw = Path(sys.argv[1])
    i = pd.read_csv(raw/"price_indore.csv", parse_dates=["date"]).set_index("date").p_modal
    g = pd.read_excel(raw/"price_gondal_rajkot.xlsx", parse_dates=["Date"]).set_index("Date").p_modal
    m = pd.concat([i.resample("MS").mean().rename("Indore"), g.resample("MS").mean().rename("Gondal")], axis=1).round(1)
    MSP = {2021:3950, 2022:4300, 2023:4600, 2024:4892, 2025:5328}   # Kharif marketing season (Oct-Sep), start year
    m["msp"] = [MSP.get(t.year if t.month>=10 else t.year-1) for t in m.index]
    m.index.name="month"; m.to_csv(OUT/"monthly_prices.csv")
