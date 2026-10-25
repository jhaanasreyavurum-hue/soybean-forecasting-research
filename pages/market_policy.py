import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c

f = c.filters()
mk = c.markets(f)
m = c.monthly()
c.header("Market & policy", "Support price, storage and market linkage",
         "MSP is Soybean (Yellow), Kharif marketing seasons 2021-22 → 2025-26, as recorded in the ARIMAX notebook.")

# ---- MSP
mm = m.dropna(subset=["msp"])
fig = go.Figure()
for mkt in mk:
    fig.add_scatter(x=mm.index, y=mm[mkt], name=mkt, line=dict(color=c.MKT[mkt], width=2.4), connectgaps=True)
fig.add_scatter(x=mm.index, y=mm.msp, name="MSP", line=dict(color=c.GRADE_COL["F"], width=2.2, dash="dot", shape="hv"))
fig.update_yaxes(title="₹ / quintal")
st.markdown("##### Monthly modal price vs MSP")
c.show(fig, 360)

k = st.columns(4)
for mkt, cc in zip(["Indore", "Gondal"], [k[0], k[1]]):
    s = mm[mm.index >= "2023-10-01"].dropna(subset=[mkt])
    below = int((s[mkt] < s.msp).sum())
    c.kpi(cc, f"{below}/{len(s)}", f"{mkt} months below MSP · Oct-23 → Dec-25", c.GRADE_COL["F"])
s = mm.dropna(subset=["Indore"]).tail(12)
c.kpi(k[2], f"{(s.Indore / s.msp).mean():.2f}×", "Indore price ÷ MSP, last 12 months")
c.kpi(k[3], "₹4,471", "Jan-26 monthly ARIMAX forecast, Indore (95% PI ₹3,809–5,134) vs MSP ₹5,328", c.GRADE_COL["F"])
st.write("")

a, b = st.columns(2)
with a:
    st.markdown("##### Share of months below MSP, by marketing season")
    t = mm.copy()
    t["season"] = [f"{y}-{str(y + 1)[2:]}" for y in (np.where(t.index.month >= 10, t.index.year, t.index.year - 1))]
    fig = go.Figure()
    for mkt in mk:
        g = t.dropna(subset=[mkt]).groupby("season").apply(lambda x: (x[mkt] < x.msp).mean() * 100, include_groups=False)
        fig.add_bar(x=g.index, y=g.values, name=mkt, marker_color=c.MKT[mkt])
    fig.update_layout(barmode="group"); fig.update_yaxes(title="% of months", range=[0, 105])
    c.show(fig, 320, "closest")
    c.note("Seasons are Oct–Sep; 2021-22 starts mid-sample so its first months are missing from the MSP table.")
with b:
    st.markdown("##### Gap to MSP (price ÷ MSP − 1)")
    fig = go.Figure()
    for mkt in mk:
        fig.add_scatter(x=mm.index, y=(mm[mkt] / mm.msp - 1) * 100, name=mkt, line=dict(color=c.MKT[mkt], width=2), connectgaps=True)
    fig.add_hline(y=0, line_color=c.INK)
    fig.update_yaxes(title="% above (+) / below (−) MSP")
    c.show(fig, 320)

# ---- storage
st.markdown("##### Does holding through harvest pay? Oct–Dec average → next Feb–Apr average (Indore)")
sg = c.csv("storage_gain.csv")
a, b = st.columns([3, 2])
with a:
    fig = go.Figure(go.Bar(x=sg.harvest_year.astype(str), y=sg.gain_pct, marker_color=[c.GRADE_COL["A"] if v >= 0 else c.GRADE_COL["F"] for v in sg.gain_pct],
                           text=sg.gain_pct.map(lambda v: f"{v:+.0f}%"), textposition="outside", cliponaxis=False, hovertemplate="Harvest %{x}<br>%{y:+.1f}%<extra></extra>"))
    fig.update_yaxes(title="storage gain %", range=[-12, 55])
    c.show(fig, 320, "closest")
with b:
    kk = st.columns(2)
    c.kpi(kk[0], f"{(sg.gain_pct > 0).sum()}/{len(sg)}", "years with a gain", c.GRADE_COL["A"])
    c.kpi(kk[1], f"{sg.gain_pct.median():+.1f}%", f"median gain (mean {sg.gain_pct.mean():+.1f}%)")
    st.write("")
    st.markdown("**Arrival pressure** (notebook): price change next month by quartile of arrivals ÷ 12-month average")
    ap = c.csv("arrival_pressure.csv")
    fig = go.Figure(go.Bar(x=ap.quartile, y=ap.mean_pct, marker_color=[c.GRADE_COL["A"] if v > 0 else "#B9B2A0" for v in ap.mean_pct], text=ap.mean_pct.map(lambda v: f"{v:+.1f}%"), textposition="outside"))
    fig.update_yaxes(range=[-4, 7], title="mean m/m price change %")
    c.show(fig, 220, "closest")
c.note("Gains ranged −5% to +46% by year. The arrival-pressure effect looks strong (HAC p ≈ 0.000) but collapses with calendar-month effects (p = 0.39): it is the harvest-glut → recovery cycle, not an independent arrivals signal.")

# ---- cointegration
st.markdown("##### Are Indore and Gondal one market? (VAR/VECM notebook, monthly)")
a, b = st.columns([2, 3])
with a:
    adf = pd.DataFrame({"series": ["Gondal level", "Gondal 1st diff", "Indore level", "Indore 1st diff"],
                        "ADF stat": [-2.19, -11.52, -2.01, -7.63], "p-value": [0.210, 0.0, 0.282, 0.0], "stationary": ["no", "yes", "no", "yes"]})
    st.dataframe(adf, hide_index=True, width="stretch", column_config={"p-value": st.column_config.NumberColumn(format="%.3f")})
    st.markdown("**Johansen trace test:** r = 0 → 44.55 vs 15.49 critical (reject); r ≤ 1 → 3.11 vs 3.84 (don’t reject) → **rank 1**.")
    st.markdown("**Adjustment:** α = (0.44 Gondal, 0.95 Indore), β = (1, −1.036) → β′α = −0.54, the pair is pulled back toward its long-run relation.")
with b:
    sp = (m.Gondal - 1.03565647 * m.Indore).dropna()
    fig = go.Figure()
    fig.add_scatter(x=sp.index, y=sp, name="Gondal − 1.036 × Indore", line=dict(color=c.INK, width=1.8))
    fig.add_hline(y=sp.mean(), line_dash="dot", line_color=c.MUTED, annotation_text="mean")
    fig.update_yaxes(title="₹ / quintal")
    c.show(fig, 300)
    c.note("The long-run relation hovers around its mean instead of drifting, even though each price on its own wanders (non-stationary). The VECM also has a constant in the relation, so read the mean line, not zero.")
