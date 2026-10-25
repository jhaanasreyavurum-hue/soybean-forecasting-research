import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c

f = c.filters()
c.header("Forecast lab", "Dissecting the forecasts",
         "Weekly hold-out (Jan 2023 → Jan 2026) for ARCHX and LSTM, plus daily ARIMA diagnostics from the notebooks.")
mkt = st.radio("Market", ["Indore", "Gondal"], horizontal=True) if f["market"] == "Both" else f["market"]
w = c.weekly_preds(mkt)
models = ["ARCHX", "LSTM (price-only)", "LSTM-X"]
pick = st.multiselect("Models", models, default=["LSTM-X"])
pal = {"ARCHX": "#C98A1B", "LSTM (price-only)": "#3B7DA6", "LSTM-X": "#A63D2F"}

# ---- metrics table (recomputed from the prediction files)
rows = []
for name in models + ["Naive (last week)"]:
    s = w[["actual_price", name]].dropna()
    e = s[name] - s.actual_price
    rows.append((name, np.sqrt((e ** 2).mean()), e.abs().mean(), (e.abs() / s.actual_price).mean() * 100, len(s)))
mt = pd.DataFrame(rows, columns=["model", "RMSE", "MAE", "MAPE %", "weeks"])
mt["RMSE vs naïve %"] = (mt.RMSE / mt.loc[mt.model == "Naive (last week)", "RMSE"].iloc[0] - 1) * 100
k = st.columns(4)
for i, r in mt.iloc[:3].iterrows():
    c.kpi(k[i], f"{r['RMSE vs naïve %']:+.1f}%", f"{r.model}: RMSE vs naïve (₹{r.RMSE:,.0f})", c.GRADE_COL["A"] if r["RMSE vs naïve %"] < -3 else c.GRADE_COL["C"] if abs(r["RMSE vs naïve %"]) <= 3 else c.GRADE_COL["F"])
c.kpi(k[3], f"{int((w.gap_days > 7).sum())}", f"gaps > 1 week in the {len(w)} test weeks (longest {int(w.gap_days.max())} days)")
c.note("Recomputed here from the prediction files; the first test week has no prior price, so figures can differ from the scorecard by a fraction of a percent.")

# ---- main
st.markdown("##### Actual vs predicted")
fig = go.Figure()
for gdate in w.index[w.gap_days > 7]:
    fig.add_vrect(x0=gdate - pd.Timedelta(days=int(w.gap_days[gdate])), x1=gdate, fillcolor="#A63D2F", opacity=.10, line_width=0, layer="below")
for name in pick:
    fig.add_scatter(x=w.index, y=w[name], name=name, line=dict(color=pal[name], width=1.8))
fig.add_scatter(x=w.index, y=w.actual_price, name="Actual", line=dict(color=c.INK, width=3))
fig.update_yaxes(title="₹ / quintal")
c.show(fig, 400)
c.note("Red bands = weeks with no market data (dropped, not filled). The aligned series therefore isn’t gapless.")

# ---- shadow
st.markdown("##### The lag-1 shadow: is the model just copying last week?")
name = pick[0] if pick else "LSTM-X"
s = w[["actual_price", name, "Naive (last week)"]].dropna()
a, b = st.columns(2)
for col_, xk, title in [(a, "Naive (last week)", f"{name} vs LAST week’s price"), (b, "actual_price", f"{name} vs THIS week’s price")]:
    r = s[name].corr(s[xk])
    fig = go.Figure(go.Scatter(x=s[xk], y=s[name], mode="markers", marker=dict(color=pal[name], size=6, opacity=.7)))
    lo, hi = s[[xk, name]].min().min(), s[[xk, name]].max().max()
    fig.add_scatter(x=[lo, hi], y=[lo, hi], line=dict(color=c.INK, dash="dot"), showlegend=False)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="₹ / quintal"); fig.update_yaxes(title="prediction")
    with col_:
        st.markdown(f"**{title}** · correlation {r:.4f}")
        c.show(fig, 320, "closest")
c.note("When predictions sit on the diagonal against last week’s price, the model has learned “next week ≈ this week”. That is exactly what a random walk does, so it ties naïve.")

# ---- errors
st.markdown("##### Errors")
e = (w[name] - w.actual_price).dropna()
a, b, d3 = st.columns(3)
with a:
    fig = go.Figure(go.Bar(x=e.index, y=e.values, marker_color=[c.GRADE_COL["F"] if v < 0 else c.GRADE_COL["A"] for v in e.values]))
    fig.update_yaxes(title="predicted − actual (₹)")
    st.markdown("**Over time**"); c.show(fig, 300, "x")
with b:
    fig = go.Figure(go.Histogram(x=e, nbinsx=30, marker_color=pal[name]))
    fig.add_vline(x=0, line_color=c.INK)
    st.markdown(f"**Distribution** · mean {e.mean():+.0f}, sd {e.std():.0f}"); c.show(fig, 300, "closest")
with d3:
    vals, ci = c.acf(e, 20)
    fig = go.Figure(go.Bar(x=list(range(1, 21)), y=vals, marker_color=[c.GRADE_COL["F"] if abs(v) > ci else "#B9B2A0" for v in vals]))
    fig.add_hline(y=ci, line_dash="dot", line_color=c.MUTED); fig.add_hline(y=-ci, line_dash="dot", line_color=c.MUTED)
    st.markdown("**Error autocorrelation**"); c.show(fig, 300, "closest")
c.note("Red bars break the 95% band: leftover pattern the model could still learn. Mostly grey = errors look like noise.")

# ---- month heatmap + direction
a, b = st.columns(2)
with a:
    st.markdown("##### Hardest months (daily ARIMA, MAE ₹)")
    me = c.csv("month_error_arima.csv").set_index("month")
    cols = c.markets(f)
    z = (me[cols] / me[cols].mean()).T
    fig = go.Figure(go.Heatmap(z=z.values, x=me.index, y=cols, colorscale=[[0, "#F4EBD3"], [1, c.GRADE_COL["F"]]], text=me[cols].T.round(0).astype(int).astype(str).values,
                               texttemplate="%{text}", hovertemplate="%{y} %{x}<br>%{text} ₹ (×%{z:.2f} of avg)<extra></extra>", showscale=False))
    c.show(fig, 130 + 40 * len(cols), "closest")
    c.note("Colour = month’s MAE relative to that market’s average. Indore: Sep error is ≈2.1× its monthly average (₹442 vs ₹212). Gondal: Oct (₹99) is worst, then Apr–May.")
with b:
    st.markdown("##### Up or down? (weekly, recomputed)")
    out = []
    for nm in models:
        s2 = w[["actual_price", nm, "Naive (last week)"]].dropna()
        act, prd = np.sign(s2.actual_price - s2["Naive (last week)"]), np.sign(s2[nm] - s2["Naive (last week)"])
        mv = act != 0
        out.append((nm, (act[mv] == prd[mv]).mean() * 100, int(mv.sum())))
    dd = pd.DataFrame(out, columns=["model", "direction accuracy %", "weeks with a move"])
    fig = go.Figure(go.Bar(x=dd.model, y=dd["direction accuracy %"], marker_color=[pal[m] for m in dd.model], text=dd["direction accuracy %"].map(lambda v: f"{v:.0f}%"), textposition="outside"))
    fig.add_hline(y=50, line_dash="dot", line_color=c.MUTED, annotation_text="coin flip")
    fig.update_yaxes(range=[0, 80], title="%")
    c.show(fig, 260, "closest")
    dc = c.csv("direction_daily.csv"); dc = dc[dc.market == mkt].iloc[0]
    c.note(f"Daily ARIMA ({mkt}): {dc.direction_acc:.1f}% correct on {int(dc.n_moves)} move days vs {dc.prev_direction_acc:.1f}% for “same direction as yesterday”. "
           f"Confusion matrix (rows actual fall/rise): [[{int(dc.tn)}, {int(dc.fp)}], [{int(dc.fn)}, {int(dc.tp)}]].")

# ---- future
st.markdown("##### Next-days forecast from the daily ARIMA (notebook output, from 1 Jan 2026)")
ff = c.csv("future_forecast_daily.csv"); ff = ff[ff.market == mkt]
fig = go.Figure()
fig.add_scatter(x=ff.date, y=ff.upper95, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
fig.add_scatter(x=ff.date, y=ff.lower95, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(63,94,43,.18)", name="95% interval", hoverinfo="skip")
fig.add_scatter(x=ff.date, y=ff.forecast, name="ARIMA forecast", line=dict(color=c.MKT[mkt], width=3))
fig.update_yaxes(title="₹ / quintal")
c.show(fig, 300)
c.note("The forecast is essentially flat (a random-walk-like model) and the band widens with the horizon. Treat as a range, not a call.")
