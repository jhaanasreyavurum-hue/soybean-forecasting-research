import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c

f = c.filters()
mk = c.markets(f)
d = c.daily().loc[str(f["dates"][0]):str(f["dates"][1])]
c.header("Volatility", "Price level is hard to forecast. Risk is not.",
         "Volatility clusters: calm spells and turbulent spells. GARCH models that — and it is where the ARCH family earns its keep.")
col = {"Indore": "ind", "Gondal": "gon"}
clip = st.toggle("Exclude daily moves beyond ±20% (mostly closures / data jumps)", value=True)
ret, dropped = {}, {}
for m_ in ["Indore", "Gondal"]:
    r_ = np.log(d[f"{col[m_]}_p_modal"].dropna()).diff().dropna() * 100
    big = r_.abs() > 20
    dropped[m_] = int(big.sum())
    ret[m_] = r_[~big] if clip else r_

k = st.columns(4)
for i, m in enumerate(["Indore", "Gondal"]):
    r = ret[m]
    c.kpi(k[i * 2], f"{r.std():.2f}%", f"{m} daily return sd", c.MKT[m])
    c.kpi(k[i * 2 + 1], f"{(r == 0).mean() * 100:.0f}%", f"{m} days with 0% change", c.MKT[m])
st.write("")
c.note(f"{'Excluded ' + str(dropped['Indore']) + ' Indore and ' + str(dropped['Gondal']) + ' Gondal days with moves above ±20%. ' if clip else 'Raw returns: a few extreme jumps dominate the standard deviation. '}A large share of zero-change days comes from the cleaned files carrying prices forward over closures/holidays, so daily volatility here is understated on those days.")

st.markdown("##### Rolling 30-day volatility of daily returns")
fig = go.Figure()
for m in mk:
    rv = ret[m].rolling(30, min_periods=15).std()
    fig.add_scatter(x=rv.index, y=rv, name=m, line=dict(color=c.MKT[m], width=1.6))
fig.update_yaxes(title="sd of daily % change")
c.show(fig, 360)

a, b = st.columns(2)
with a:
    st.markdown("##### Volatility by year")
    fig = go.Figure()
    for m in mk:
        y = ret[m].groupby(ret[m].index.year).std()
        fig.add_bar(x=y.index.astype(str), y=y.values, name=m, marker_color=c.MKT[m])
    fig.update_layout(barmode="group"); fig.update_yaxes(title="sd of daily % change")
    c.show(fig, 330, "closest")
with b:
    st.markdown("##### Clustering test: autocorrelation of |returns|")
    fig = go.Figure()
    for m in mk:
        v, ci = c.acf(ret[m].abs(), 30)
        fig.add_bar(x=list(range(1, 31)), y=v, name=m, marker_color=c.MKT[m])
    fig.add_hline(y=ci, line_dash="dot", line_color=c.MUTED)
    fig.update_layout(barmode="group"); fig.update_xaxes(title="lag (days)")
    c.show(fig, 330, "closest")
    c.note("Bars above the dotted 95% line: big moves follow big moves. This is what ARCH-LM tests formally.")

st.markdown("##### ARCH vs GARCH (notebook `arch_garch`)")
vol = pd.DataFrame({"market": ["Indore", "Gondal"], "ARIMA": ["(2,1,3)", "(1,1,3)"], "ARCH-LM": [294.6, 356.3], "ARCH-LM p": [0.0, 0.0],
                    "AIC (ARCH)": [46595.0, 45950.0], "AIC (GARCH)": [44598.7, 45393.5], "ARCH p after GARCH": [0.378, 0.052]})
vol["ΔAIC"] = vol["AIC (GARCH)"] - vol["AIC (ARCH)"]
vol = vol[vol.market.isin(mk)]
a, b = st.columns([5, 2])
with a:
    st.dataframe(vol, hide_index=True, width="stretch", column_config={
        "ARCH-LM": st.column_config.NumberColumn(format="%.1f"), "AIC (ARCH)": st.column_config.NumberColumn(format="%.0f"),
        "AIC (GARCH)": st.column_config.NumberColumn(format="%.0f"), "ΔAIC": st.column_config.NumberColumn(format="%.0f"),
        "ARCH p after GARCH": st.column_config.NumberColumn(format="%.3f")})
with b:
    fig = go.Figure()
    fig.add_bar(x=vol.market, y=-vol["ΔAIC"], marker_color=[c.MKT[m] for m in vol.market], text=(-vol["ΔAIC"]).map(lambda v: f"{v:,.0f}"), textposition="outside")
    fig.update_yaxes(title="AIC improvement from GARCH")
    c.show(fig, 260, "closest")
c.note("GARCH lowers AIC a lot in both markets and removes the leftover ARCH effect in Indore (p = 0.38; Gondal borderline at 0.052). "
       "But the same notebook’s multi-step price forecasts are poor (RMSE ₹1,098 Indore / ₹1,475 Gondal): GARCH models the variance, not the level.")

st.markdown("##### Are the 95% intervals honest? (daily ARIMA, 1-step)")
cv = c.csv("interval_coverage.csv"); cv = cv[cv.market.isin(mk)]
cols = st.columns(max(len(cv), 1))
for cc, (_, r) in zip(cols, cv.iterrows()):
    off = r.coverage_95 - 95
    c.kpi(cc, f"{r.coverage_95:.1f}%", f"{r.market}: actual coverage of a 95% band — {'too narrow' if off < -2 else 'too wide' if off > 2 else 'well calibrated'} ({off:+.1f} pts) · mean width ₹{r.mean_width:,.0f}", c.MKT[r.market])
