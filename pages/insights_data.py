import pandas as pd
import streamlit as st

import common as c

f = c.filters()
sc = c.scorecard()
d = c.daily()
c.header("Insights & data", "What the results say, and what they were built from", "Findings are computed from the same tables the charts use.")

# ---- findings
beat = sc[sc.U < 0.97]
tie = sc[(sc.U >= 0.97) & (sc.U <= 1.03)]
lose = sc[sc.U > 1.03]
by = sc.groupby("frequency").U.median()
ind_d = sc[(sc.market == "Indore") & (sc.frequency == "Daily")]
gon_d = sc[(sc.market == "Gondal") & (sc.frequency == "Daily")]
st.markdown("#### Key findings")
st.markdown(f"""
1. **Prices are close to a random walk.** Of {len(sc)} model runs, {len(beat)} beat the naïve forecast by more than 3% (RMSE), {len(tie)} tied and {len(lose)} lost.
2. **The edge fades with frequency.** Median RMSE ratio: daily {by['Daily']:.2f}, weekly {by['Weekly']:.2f}, monthly {by['Monthly']:.2f} (below 1 = better than naïve).
3. **Indore daily is the one clear win** — all {len(ind_d)} runs beat naïve by {((1 - ind_d.U.max()) * 100):.0f}–{((1 - ind_d.U.min()) * 100):.0f}%. Indore’s naïve baseline is weak (jumps and carried-forward values around closures), so some of this is a soft target.
4. **Gondal daily is mixed.** ARIMA gains 5.6% on RMSE but is 10.8% worse on MAE; the ML regressors ({', '.join(gon_d[gon_d.family.str.contains('ML')].model)}) lose to naïve.
5. **Extra inputs don’t help out of sample.** Crude oil and arrivals barely register in permutation importance, and removing them leaves monthly ARIMAX no worse.
6. **Volatility is the forecastable part.** ARCH effects are strong in both markets; GARCH cuts AIC by ≈2,000 (Indore) and ≈560 (Gondal).
7. **Policy context:** Indore averaged below MSP in most months since Oct 2023, and the Jan-26 forecast (₹4,471) sits below the ₹5,328 MSP.
""")

# ---- data inventory
st.markdown("#### Data behind the results")
inv = pd.DataFrame([
    ("Indore price", "price_indore_market_cleaned_ffill_no_holidays_covid.csv", "p_min, p_max, p_modal (₹/qtl)", d.ind_p_modal),
    ("Gondal price", "price_data_cleaned_rajkot.xlsx", "p_min, p_max, p_modal (₹/qtl)", d.gon_p_modal),
    ("Indore arrivals", "arrivals_indore_market_cleaned_ffill_no_holidays_covid.csv", "quantity_tonnes", d.ind_arrivals_t),
    ("Gondal arrivals", "merged_quantity_2011_2025_filled.csv", "qty (unit not stated in file)", d.gon_arrivals_qty),
    ("Crude oil", "crude_oil_filled_ffill.csv", "price (looks like USD/bbl, unverified); weekends/holidays forward-filled", d.crude),
], columns=["series", "source file", "what it holds", "_s"])
inv["rows"] = [int(s.dropna().shape[0]) for s in inv._s]
inv["first"] = [s.dropna().index.min().date() for s in inv._s]
inv["last"] = [s.dropna().index.max().date() for s in inv._s]
st.dataframe(inv[["series", "source file", "rows", "first", "last", "what it holds"]], hide_index=True, width="stretch")

q = st.columns(4)
for i, (lab, s) in enumerate([("Indore", d.ind_p_modal), ("Gondal", d.gon_p_modal)]):
    s = s.dropna()
    c.kpi(q[i], f"{(s.diff() == 0).mean() * 100:.0f}%", f"{lab} days with price identical to the previous day", c.GRADE_COL["C"])
c.kpi(q[2], f"{d.ind_p_modal.isna().sum()}", "calendar days with Gondal data but no Indore price")
c.kpi(q[3], f"{d.gon_p_modal.isna().sum()}", "calendar days with Indore data but no Gondal price")
c.note("“Identical to previous day” includes both genuinely unchanged mandi prices and values carried forward over closures — the cleaned files don’t flag which. "
       "This is why daily Indore/Gondal errors look small and why naïve is hard to beat on typical days.")

# ---- downloads
st.markdown("#### Download")
fs = c.filtered_scorecard(f)
a, b, e = st.columns(3)
a.download_button("Scorecard (all runs)", sc.drop(columns=["frequency", "family", "grade"]).to_csv(index=False), "scorecard.csv", "text/csv")
b.download_button("Current filtered view", fs.drop(columns=["label"]).to_csv(index=False), "scorecard_filtered.csv", "text/csv")
e.download_button("Merged daily data", d.reset_index().to_csv(index=False), "daily.csv", "text/csv")

# ---- method
st.markdown("#### Method & caveats")
st.markdown("""
| Track | Frequency & test window | Models | Notebook |
|---|---|---|---|
| Daily · ARIMA family | daily, 1-step rolling, last 20% | ARIMA, SARIMA | `Soybean_ARIMA_SARIMA_Indore_Gondal` |
| Daily · ML regressors | daily, 1-step, last 20% | Linear Reg., Random Forest, XGBoost | `soybean_price_prediction_indore / _rajkot` |
| Weekly · ARCHX / LSTM | weekly (Fri), 1-step, last 20% (Jan 2023 →) | ARX + ARCH(1), LSTM, LSTM + lagged crude | `soybean_archx_lstm_project` |
| Monthly · ARIMAX / VAR | monthly, 1-step, last 36 months, Indore only | ARIMAX, SARIMAX, VAR, VECM | `soybean_arimax_sarimax_executed` |

- **Grades** are the ratio of model error to naïve error on the same test period, banded A < 0.90 · B < 0.97 · C ≤ 1.03 · D ≤ 1.15 · F above. My banding, not a statistical test; Diebold–Mariano results are on the Model comparison page.
- **Naïve** = last observed price. Each track uses its own naïve on its own test window, so ₹ errors are comparable only within a track.
- **ARCHX** is an ARX mean equation with ARCH(1) variance; crude oil enters the mean, not the variance. Weekly Indore ARCHX has α = 1.0 (boundary estimate).
- **Weekly series** drop weeks without market data or lagged oil rather than filling them.
- **12-month VAR/VECM** run (Indore RMSE ₹282, Gondal ₹158) has no naïve benchmark in its notebook, so it is not graded.
- Notebook-only figures (AIC, DM tests, feature importance, ablation, month-wise errors, intervals, GARCH, VECM) were copied from executed outputs; `scripts/notebook_tables.py` lists them.
""")
