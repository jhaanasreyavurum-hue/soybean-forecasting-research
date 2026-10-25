import plotly.graph_objects as go
import streamlit as st

import common as c

f = c.filters()
d = c.filtered_scorecard(f)
c.header("Agri-analytics · mandi price forecasting · 2011–2025", "Soybean Forecast Analytics",
         "Indore (MP) vs Gondal (Gujarat). Every model is measured against the simplest forecast there is — “next price = last price”.")
if d.empty:
    st.warning("No model runs match the current filters (MAPE is missing for the daily ML runs). Change the sidebar filters.")
    st.stop()

d["grade"] = d.ratio.map(c.grade)
best = d.loc[d.ratio.idxmin()]
beat = int((d.ratio < 0.97).sum())
mp = c.monthly()
last = mp.dropna(subset=["Indore"]).iloc[-1]

k = st.columns(5)
c.kpi(k[0], f"{len(d)}", "model runs in view")
c.kpi(k[1], f"{beat}/{len(d)}", f"beat naïve on {f['metric']}", c.GRADE_COL["A"])
c.kpi(k[2], f"{(1 - best.ratio) * 100:.1f}%", f"best {f['metric']} gain · {best.model}", c.GRADE_COL["A"])
c.kpi(k[3], f"₹{last.Indore:,.0f}", f"Indore avg price {last.name:%b %Y}")
c.kpi(k[4], f"{last.Indore / last.msp:.2f}×", f"Indore price ÷ MSP ({last.name:%b %Y}, MSP ₹{last.msp:,.0f})", c.GRADE_COL["F"] if last.Indore < last.msp else c.GRADE_COL["A"])

st.write("")
c.take(f"<b>Read-out:</b> {beat} of {len(d)} runs beat the naïve forecast by more than 3% on {f['metric']}; "
       f"{int(((d.ratio >= 0.97) & (d.ratio <= 1.03)).sum())} tie and {int((d.ratio > 1.03).sum())} are worse. "
       "Gains are real only for daily Indore; weekly runs tie and monthly runs lose.")

left, right = st.columns([3, 2])
with left:
    st.markdown(f"##### Leaderboard — {f['metric']} ratio vs naïve")
    p = d.sort_values("ratio").reset_index(drop=True)
    fig = go.Figure(go.Bar(y=p.label, x=p.ratio - 1, base=1, orientation="h", marker_color=p.grade.map(c.GRADE_COL),
                           text=p.ratio.map(lambda v: f"{v:.2f}"), textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>ratio %{text}<extra></extra>"))
    fig.add_vline(x=1, line_color=c.INK, line_width=2)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="← better than naïve | worse →", range=[min(0.5, p.ratio.min() - 0.3), p.ratio.max() + 0.15])
    c.show(fig, h=max(360, 24 * len(p) + 80), hover="closest")
with right:
    st.markdown("##### Grade mix")
    cnt = d.grade.value_counts().reindex(list("ABCDF"), fill_value=0)
    fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=[c.GRADE_COL[g] for g in cnt.index], text=cnt.values, textposition="outside"))
    fig.update_yaxes(visible=False, range=[0, cnt.max() * 1.25 + 0.5])
    c.show(fig, h=220, hover="closest")
    st.caption("A < 0.90 · B < 0.97 · C ≤ 1.03 (tie) · D ≤ 1.15 · F above — my banding of the ratio.")
    st.markdown("##### By frequency")
    byf = d.groupby("frequency").ratio.median().reindex(["Daily", "Weekly", "Monthly"]).dropna()
    for fr, v in byf.items():
        st.markdown(f'<span class="stamp" style="background:{c.GRADE_COL[c.grade(v)]}">{c.grade(v)}</span>&nbsp; **{fr}** — median ratio {v:.2f}', unsafe_allow_html=True)

st.markdown("##### Heatmap — where models beat or lose to naïve")
d["col"] = d.market + " · " + d.frequency
piv = d.pivot_table(index="model", columns="col", values="ratio")
fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, zmin=0.7, zmax=1.3, zmid=1,
                           colorscale=[[0, c.GRADE_COL["A"]], [.5, "#F4EBD3"], [1, c.GRADE_COL["F"]]],
                           text=[[("" if v != v else f"{v:.2f}") for v in r] for r in piv.values], texttemplate="%{text}",
                           hovertemplate="%{y}<br>%{x}<br>ratio %{z:.3f}<extra></extra>", colorbar=dict(title="ratio")))
fig.update_yaxes(autorange="reversed")
c.show(fig, h=max(320, 34 * len(piv) + 90), hover="closest")
c.note("Blank = that model wasn’t run for that market/frequency. A model name can appear in more than one frequency (e.g. LSTM only weekly).")
