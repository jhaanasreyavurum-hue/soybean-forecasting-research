import plotly.graph_objects as go
import streamlit as st

import common as c

f = c.filters()
d = c.filtered_scorecard(f)
c.header("Model comparison", "Which models earn their complexity?",
         f"Everything below uses the sidebar filters. Metric: {f['metric']}. Ratio = model ÷ naïve on the same test period.")
if d.empty:
    st.warning("No runs match the filters.")
    st.stop()

# ---- raw vs naive grouped bars
st.markdown(f"##### {f['metric']}: model vs its own naïve baseline")
p = d.sort_values(["frequency", "market", "ratio"]).reset_index(drop=True)
unit = "%" if f["metric"] == "MAPE" else "₹"
fig = go.Figure()
fig.add_bar(x=p.label, y=p.value, name="Model", marker_color=[c.MKT[m] for m in p.market])
fig.add_scatter(x=p.label, y=p.naive, name="Naïve baseline", mode="markers", marker=dict(symbol="line-ew", size=26, color=c.INK, line=dict(width=3, color=c.INK)))
fig.update_yaxes(title=f"{f['metric']} ({unit})")
fig.update_xaxes(tickangle=-40)
c.show(fig, 440, "closest")
c.note("Bar colour = market (green Indore, amber Gondal). The black tick is the naïve error for that run — a bar above its tick loses to “no change”.")

a, b = st.columns(2)
with a:
    st.markdown("##### Skill vs naïve (% improvement)")
    q = d.sort_values("ratio")
    fig = go.Figure(go.Bar(y=q.label, x=(1 - q.ratio) * 100, orientation="h", marker_color=[c.GRADE_COL[c.grade(r)] for r in q.ratio],
                           text=((1 - q.ratio) * 100).map(lambda v: f"{v:+.1f}%"), textposition="outside", cliponaxis=False))
    fig.add_vline(x=0, line_color=c.INK)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="% lower error than naïve (negative = worse)", range=[min(-5, (1 - q.ratio.max()) * 100 - 10), (1 - q.ratio.min()) * 100 + 12])
    c.show(fig, max(340, 24 * len(q) + 70), "closest")
with b:
    st.markdown("##### Does the edge survive as frequency drops?")
    fig = go.Figure()
    for mkt in c.markets(f):
        s = d[d.market == mkt]
        fig.add_scatter(x=s.frequency, y=s.ratio, mode="markers", name=mkt, marker=dict(size=12, color=c.MKT[mkt], opacity=.75),
                        text=s.model, hovertemplate="%{text}<br>ratio %{y:.3f}<extra></extra>")
    fig.add_hline(y=1, line_color=c.INK, annotation_text="naïve", annotation_position="top right")
    fig.update_xaxes(categoryorder="array", categoryarray=["Daily", "Weekly", "Monthly"])
    fig.update_yaxes(title=f"{f['metric']} ratio")
    c.show(fig, 340, "closest")
    c.note("Each dot is a model run. Daily dots sit below 1 for Indore; weekly ones hug 1.0; monthly ones are well above.")

# ---- details
st.markdown("##### Model details & significance tests")
md = c.csv("model_details.csv")
if f["market"] != "Both":
    md = md[md.market == f["market"]]
st.dataframe(md.rename(columns={"vs_naive": "vs naïve (Diebold–Mariano)"}), hide_index=True, width="stretch",
             column_config={"AIC": st.column_config.NumberColumn(format="%.1f"), "BIC": st.column_config.NumberColumn(format="%.1f")})

# ---- feature importance + ablation
a, b = st.columns(2)
with a:
    st.markdown("##### What the ML models actually use")
    fi = c.csv("feature_importance.csv")
    fig = go.Figure()
    for mkt in c.markets(f):
        s = fi[fi.market == mkt]
        fig.add_bar(y=s.group, x=s.rmse_increase, name=mkt, orientation="h", marker_color=c.MKT[mkt])
    fig.update_layout(barmode="group")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="RMSE increase when shuffled (₹/qtl)", type="log")
    c.show(fig, 340, "closest")
    c.note("Permutation importance (daily regressors; Indore: Linear Regression, Gondal: Random Forest). Own recent price dominates by roughly 40× or more; crude and arrivals barely register. Log axis.")
with b:
    st.markdown("##### Predictor ablation — monthly ARIMAX (Indore)")
    ab = c.csv("ablation.csv")
    fig = go.Figure(go.Bar(x=ab.variant, y=ab.test_rmse, marker_color=["#2F5D3A" if i == 0 else "#B9B2A0" for i in range(len(ab))],
                           text=ab.test_rmse.map(lambda v: f"{v:.0f}"), textposition="outside"))
    fig.add_hline(y=230.2, line_color=c.GRADE_COL["F"], line_dash="dot", annotation_text="naïve 230.2", annotation_position="bottom right")
    fig.update_yaxes(title="test RMSE (₹/qtl)", range=[200, 320])
    c.show(fig, 340, "closest")
    c.note("Removing Indore arrivals or all exogenous variables helped slightly; no variant beats naïve.")
