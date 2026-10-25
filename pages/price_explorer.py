import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c

f = c.filters()
mk = c.markets(f)
d = c.daily().loc[str(f["dates"][0]):str(f["dates"][1])]
m = c.monthly().loc[str(f["dates"][0]):str(f["dates"][1])]
c.header("Price explorer", "What the prices actually did",
         "Daily mandi modal prices, 2011–2025, from the cleaned files every model was fitted on.")
if d.empty:
    st.stop()

col = {"Indore": "ind", "Gondal": "gon"}

# ---- KPIs
k = st.columns(4)
for i, mkt in enumerate(["Indore", "Gondal"]):
    s = d[f"{col[mkt]}_p_modal"].dropna()
    c.kpi(k[i * 2], f"₹{s.iloc[-1]:,.0f}", f"{mkt} latest modal price ({s.index[-1]:%d %b %Y})", c.MKT[mkt])
    c.kpi(k[i * 2 + 1], f"₹{s.min():,.0f}–{s.max():,.0f}", f"{mkt} range in selected period", c.MKT[mkt])
st.write("")

# ---- main price chart
st.markdown("##### Modal price (zoom with the slider)")
band = st.toggle("Show daily min–max band", value=False)
fig = go.Figure()
for mkt in mk:
    p = col[mkt]
    if band:
        fig.add_scatter(x=d.index, y=d[f"{p}_p_max"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=d.index, y=d[f"{p}_p_min"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(120,120,90,.15)", showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=d.index, y=d[f"{p}_p_modal"], name=mkt, line=dict(color=c.MKT[mkt], width=1.6), connectgaps=False)
fig.update_xaxes(rangeslider=dict(visible=True, thickness=.06))
fig.update_yaxes(title="₹ / quintal")
c.show(fig, 440)

# ---- spread
a, b = st.columns(2)
with a:
    st.markdown("##### Indore − Gondal spread")
    sp = (d.ind_p_modal - d.gon_p_modal).dropna()
    fig = go.Figure()
    fig.add_scatter(x=sp.index, y=sp, name="daily spread", line=dict(color="#B9B2A0", width=1))
    fig.add_scatter(x=sp.index, y=sp.rolling(90, min_periods=30).mean(), name="90-day mean", line=dict(color=c.INK, width=2.4))
    fig.add_hline(y=0, line_dash="dot", line_color=c.MUTED)
    fig.update_yaxes(title="₹ / quintal")
    c.show(fig, 330)
    c.note(f"Mean spread {sp.mean():+.0f} ₹ · Indore above Gondal on {(sp > 0).mean() * 100:.0f}% of shared days. The two markets move together (VECM finds 1 cointegrating relation).")
with b:
    st.markdown("##### Price range inside the day (max − min, % of modal)")
    fig = go.Figure()
    for mkt in mk:
        p = col[mkt]
        r = ((d[f"{p}_p_max"] - d[f"{p}_p_min"]) / d[f"{p}_p_modal"] * 100).dropna()
        fig.add_scatter(x=r.index, y=r.rolling(60, min_periods=20).mean(), name=f"{mkt} (60-day mean)", line=dict(color=c.MKT[mkt], width=2))
    fig.update_yaxes(title="% of modal price")
    c.show(fig, 330)
    c.note("A wide daily range = traders disagree on price; a simple price-dispersion signal available without any model.")

# ---- heatmap + seasonal
a, b = st.columns(2)
tgt = "Indore" if f["market"] == "Both" else f["market"]
with a:
    st.markdown(f"##### {tgt}: average price by year × month")
    s = m[tgt].dropna()
    hm = s.groupby([s.index.year, s.index.month]).mean().unstack()
    fig = go.Figure(go.Heatmap(z=hm.values, x=[c.MONTHS[i - 1] for i in hm.columns], y=hm.index.astype(str), colorscale="YlGn",
                               hovertemplate="%{y} %{x}<br>₹%{z:,.0f}<extra></extra>", colorbar=dict(title="₹")))
    fig.update_yaxes(autorange="reversed")
    c.show(fig, 420, "closest")
with b:
    st.markdown("##### Seasonal index (month ÷ that year’s average)")
    fig = go.Figure()
    for mkt in mk:
        s = m[mkt].dropna()
        yr = s.groupby(s.index.year).transform("mean")
        idx = (s / yr).groupby(s.index.month)
        mean, q1, q3 = idx.mean(), idx.quantile(.25), idx.quantile(.75)
        fig.add_scatter(x=c.MONTHS, y=q3.values, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=c.MONTHS, y=q1.values, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(120,120,90,.15)", showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=c.MONTHS, y=mean.values, name=mkt, line=dict(color=c.MKT[mkt], width=2.5), mode="lines+markers")
    fig.add_hline(y=1, line_dash="dot", line_color=c.MUTED)
    fig.update_yaxes(title="index (1 = annual mean)")
    c.show(fig, 420)
    c.note("Shaded = middle 50% of years. Wide bands mean the seasonal pattern isn’t stable — why SARIMAX failed out of sample.")

# ---- arrivals
st.markdown("##### Arrivals vs price")
a, b = st.columns(2)
with a:
    s = m[["Indore", "ind_arr"]].dropna()
    fig = go.Figure()
    fig.add_bar(x=s.index, y=s.ind_arr, name="Indore arrivals (t/month)", marker_color="#D9CFB4", yaxis="y2")
    fig.add_scatter(x=s.index, y=s.Indore, name="Indore price", line=dict(color=c.MKT["Indore"], width=2))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="tonnes"), yaxis=dict(title="₹ / quintal"))
    c.show(fig, 340)
with b:
    s = m[["Indore", "ind_arr"]].dropna()
    s = s.assign(yr=s.index.year, lp=np.log(s.Indore))
    fig = go.Figure(go.Scatter(x=s.ind_arr, y=s.Indore, mode="markers", marker=dict(color=s.yr, colorscale="Viridis", size=7, showscale=True, colorbar=dict(title="year")),
                               customdata=s.index.strftime("%b %Y"), hovertemplate="%{customdata}<br>%{x:,.0f} t · ₹%{y:,.0f}<extra></extra>"))
    fig.update_xaxes(title="monthly arrivals (tonnes)"); fig.update_yaxes(title="₹ / quintal")
    c.show(fig, 340, "closest")
    r = s[["ind_arr", "Indore"]].corr().iloc[0, 1]
    c.note(f"Level correlation {r:+.2f} — mostly shared harvest seasonality, not a causal arrivals effect (the ARIMAX notebook’s arrival coefficient was insignificant once calendar effects were included).")

# ---- crude
st.markdown("##### Crude oil vs soybean price")
a, b = st.columns(2)
with a:
    fig = go.Figure()
    for mkt in mk:
        fig.add_scatter(x=m.index, y=m[mkt], name=mkt, line=dict(color=c.MKT[mkt], width=2))
    fig.add_scatter(x=m.index, y=m.crude, name="Crude (right axis)", line=dict(color=c.OIL, width=1.8, dash="dot"), yaxis="y2")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="crude price"), yaxis=dict(title="₹ / quintal"))
    c.show(fig, 340)
with b:
    fig = go.Figure()
    lr = np.log(m[["Indore", "Gondal", "crude"]]).diff()
    for mkt in mk:
        rc = lr[mkt].rolling(24).corr(lr.crude.shift(1))
        fig.add_scatter(x=rc.index, y=rc, name=f"{mkt} vs crude (lag 1m)", line=dict(color=c.MKT[mkt], width=2))
    fig.add_hline(y=0, line_dash="dot", line_color=c.MUTED)
    fig.update_yaxes(title="24-month rolling correlation", range=[-.8, .8])
    c.show(fig, 340)
    c.note("Monthly log-returns, crude lagged one month. The link comes and goes — consistent with crude adding no out-of-sample accuracy. Crude file is forward-filled over weekends/holidays; units unverified (looks like USD/bbl).")
