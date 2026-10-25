"""Soybean Forecast Analytics — Indore vs Gondal (2011–2025). Entry point: streamlit run app.py"""
import pandas as pd
import streamlit as st

import common as c

st.set_page_config(page_title="Soybean Forecast Analytics", page_icon="🌱", layout="wide")
st.markdown(c.CSS, unsafe_allow_html=True)

sc = c.scorecard()
with st.sidebar:
    st.markdown("### 🌱 Soybean analytics")
    st.caption("Filters apply to every page.")
    st.segmented_control("Market", ["Both", "Indore", "Gondal"], default="Both", key="f_market", selection_mode="single")
    st.multiselect("Frequency", ["Daily", "Weekly", "Monthly"], default=["Daily", "Weekly", "Monthly"], key="f_freq")
    st.multiselect("Model family", sorted(sc.family.unique()), default=sorted(sc.family.unique()), key="f_family")
    st.radio("Metric", ["RMSE", "MAE", "MAPE"], horizontal=True, key="f_metric")
    st.slider("Date range (price pages)", pd.Timestamp("2011-01-01").date(), pd.Timestamp("2025-12-31").date(),
              (pd.Timestamp("2011-01-01").date(), pd.Timestamp("2025-12-31").date()), key="f_dates")
    st.divider()
    st.caption("Metric = model error ÷ naïve “last price” error. Below 1 beats naïve. MAPE isn’t available for the daily ML runs.")

pg = st.navigation([
    st.Page("pages/overview.py", title="Overview", icon="🏁", default=True),
    st.Page("pages/price_explorer.py", title="Price explorer", icon="📈"),
    st.Page("pages/model_comparison.py", title="Model comparison", icon="⚖️"),
    st.Page("pages/forecast_lab.py", title="Forecast lab", icon="🔬"),
    st.Page("pages/volatility.py", title="Volatility", icon="🌊"),
    st.Page("pages/market_policy.py", title="Market & policy", icon="🏛️"),
    st.Page("pages/insights_data.py", title="Insights & data", icon="📋"),
])
pg.run()
