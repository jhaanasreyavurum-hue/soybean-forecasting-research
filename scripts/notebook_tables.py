"""Small tables copied from the executed notebooks' printed outputs (not recomputable without refitting)."""
import pandas as pd
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data"

pd.DataFrame([
 ("Indore","ARIMA(2,1,3)","Daily",48711.7,48748.7,"Beats naive (DM squared loss p=0.0002); no sig. diff on absolute loss"),
 ("Indore","SARIMA(2,1,2)x(1,0,1,6)","Daily",48722.9,48766.0,"Beats naive (DM squared p=0.0002); same as ARIMA (p=0.64)"),
 ("Gondal","ARIMA(0,1,2)","Daily",46273.9,46292.4,"No sig. diff on squared loss (p=0.17); naive better on absolute loss (p=0.0001)"),
 ("Gondal","SARIMA(0,1,2)x(0,0,0,6)","Daily",46273.9,46292.4,"Identical to ARIMA (no seasonal terms selected)"),
 ("Indore","ARIMAX(0,1,3)","Monthly",2030.8,2054.2,"Worse than naive (DM p=0.052)"),
 ("Indore","SARIMAX (0,1,1,12)","Monthly",None,None,"Worse than naive (DM p=0.030)"),
], columns=["market","model","frequency","AIC","BIC","vs_naive"]).to_csv(OUT/"model_details.csv", index=False)

pd.DataFrame([
 ("Indore","Own price history",140.57),("Indore","Crude oil",3.65),("Indore","Other-market price",2.01),("Indore","Own arrivals",0.48),("Indore","Calendar",0.22),
 ("Gondal","Own price history",289.87),("Gondal","Other-market price",1.69),("Gondal","Crude oil",0.05),("Gondal","Own arrivals",-0.01),
], columns=["market","group","rmse_increase"]).to_csv(OUT/"feature_importance.csv", index=False)

pd.DataFrame([
 ("Full model (ARIMAX)",289.4),("− crude",289.9),("− Indore arrivals",280.8),("− Gondal arrivals",290.7),("− harvest dummy",304.1),("No exogenous (ARIMA)",284.8),
], columns=["variant","test_rmse"]).to_csv(OUT/"ablation.csv", index=False)

m = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ind = [162.35,166.24,237.52,199.58,256.80,99.57,132.32,139.33,442.37,221.17,254.17,226.19]
gon = [37.58,37.85,42.69,80.74,81.60,55.77,49.78,71.31,63.37,99.08,66.81,58.66]
pd.DataFrame({"month":m,"Indore":ind,"Gondal":gon}).to_csv(OUT/"month_error_arima.csv", index=False)

pd.DataFrame([
 ("Q1 low",0.03,-1.11,44),("Q2",0.01,-0.52,43),("Q3",-1.82,-0.86,43),("Q4 high",4.54,3.59,44),
], columns=["quartile","mean_pct","median_pct","months"]).to_csv(OUT/"arrival_pressure.csv", index=False)

pd.DataFrame([
 ("Indore","ARIMA",732,65.6,35.6,261,101,151,219),("Gondal","ARIMA",550,60.0,39.0,166,122,98,164),
], columns=["market","model","n_moves","direction_acc","prev_direction_acc","tn","fp","fn","tp"]).to_csv(OUT/"direction_daily.csv", index=False)

pd.DataFrame([
 ("Indore","ARIMA(2,1,3)",89.5,989.2),("Gondal","ARIMA(0,1,2)",99.1,758.4),
], columns=["market","model","coverage_95","mean_width"]).to_csv(OUT/"interval_coverage.csv", index=False)

pd.DataFrame([
 ("Indore","ARIMA",["2026-01-01","2026-01-02","2026-01-03","2026-01-05","2026-01-06","2026-01-07","2026-01-08","2026-01-09"],
  [4595.3,4587.5,4590.3,4585.8,4589.1,4585.6,4588.7,4585.8],[4042.3,3987.1,3956.1,3917.9,3893.5,3860.7,3838.4,3808.7],[5148.3,5188.0,5224.5,5253.7,5284.8,5310.6,5339.0,5362.9]),
 ("Gondal","ARIMA",["2026-01-01","2026-01-02","2026-01-03","2026-01-05","2026-01-06","2026-01-07","2026-01-08","2026-01-09"],
  [4445.0,4449.8,4449.8,4449.8,4449.8,4449.8,4449.8,4449.8],[4094.0,4044.8,4018.9,3994.5,3971.3,3949.2,3928.0,3907.7],[4796.0,4854.8,4880.7,4905.1,4928.3,4950.4,4971.6,4991.9]),
], columns=["market","model","date","forecast","lower95","upper95"]).explode(["date","forecast","lower95","upper95"]).to_csv(OUT/"future_forecast_daily.csv", index=False)
