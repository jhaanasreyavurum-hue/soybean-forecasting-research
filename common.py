"""Shared palette, CSS, loaders and helpers for every page."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"

PAPER, INK, MUTED, GRID = "#F7F2E8", "#23301F", "#6B715F", "#e8e0cd"
GRADE_COL = {"A": "#3F7D3A", "B": "#7FA33F", "C": "#C98A1B", "D": "#C86B2E", "F": "#A63D2F"}
MKT = {"Indore": "#2F5D3A", "Gondal": "#C98A1B"}
OIL = "#3B7DA6"
MSP = {2021: 3950, 2022: 4300, 2023: 4600, 2024: 4892, 2025: 5328}  # Kharif season start year -> INR/qtl
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp {{ font-family:'Inter',sans-serif; color:{INK}; }}
.stApp {{ background:{PAPER}; }}
h1,h2,h3 {{ font-family:'Fraunces',serif !important; letter-spacing:-.01em; color:{INK}; }}
.block-container {{ padding-top:3.4rem; max-width:1200px; }}
[data-testid="stSidebar"] {{ background:#EFE7D4; }}
.eyebrow {{ text-transform:uppercase; letter-spacing:.14em; font-size:.74rem; color:{MUTED}; font-weight:600; }}
.ptitle {{ font-family:'Fraunces',serif; font-size:2.3rem; font-weight:800; margin:.1rem 0 .2rem; line-height:1.1; }}
.psub {{ color:{MUTED}; max-width:48rem; margin:0 0 1rem; }}
.kpi {{ background:#fffdf7; border:1px solid #e4dccb; border-radius:14px; padding:.85rem 1rem; height:100%; }}
.kpi .v {{ font-family:'Fraunces',serif; font-size:1.7rem; font-weight:800; line-height:1.05; }}
.kpi .l {{ font-size:.74rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.07em; margin-top:.3rem; }}
.take {{ border-left:4px solid {GRADE_COL['C']}; background:#fff8e6; padding:.6rem 1rem; border-radius:0 10px 10px 0; margin:.4rem 0 1rem; font-size:.96rem; }}
.note {{ color:{MUTED}; font-size:.86rem; margin:-.2rem 0 1rem; }}
.stamp {{ display:inline-flex; width:1.7rem; height:1.7rem; border-radius:50%; align-items:center; justify-content:center; color:#fff;
  font-family:'Fraunces',serif; font-weight:800; font-size:.9rem; transform:rotate(-6deg); }}
footer, #MainMenu {{ visibility:hidden; }}
</style>
"""


# ---------------- loaders ----------------
@st.cache_data
def scorecard() -> pd.DataFrame:
    d = pd.read_csv(DATA / "scorecard.csv")
    d["frequency"] = d.track.str.split(" · ").str[0]
    d["family"] = d.track.str.split(" · ").str[1]
    d["U"] = d.rmse / d.naive_rmse
    d["grade"] = d.U.map(grade)
    return d


@st.cache_data
def daily() -> pd.DataFrame:
    d = pd.read_csv(DATA / "daily.csv", parse_dates=["date"]).set_index("date")
    d["crude"] = d.crude.ffill()
    return d


@st.cache_data
def monthly() -> pd.DataFrame:
    d = daily()
    m = pd.DataFrame({
        "Indore": d.ind_p_modal.resample("MS").mean(), "Gondal": d.gon_p_modal.resample("MS").mean(),
        "ind_arr": d.ind_arrivals_t.resample("MS").sum(min_count=1), "gon_arr": d.gon_arrivals_qty.resample("MS").sum(min_count=1),
        "crude": d.crude.resample("MS").mean(),
    })
    m["msp"] = [MSP.get(t.year if t.month >= 10 else t.year - 1) for t in m.index]
    return m


@st.cache_data
def weekly_preds(market: str) -> pd.DataFrame:
    mk = market.lower()
    files = {"ARCHX": f"{mk}_archx_predictions.csv", "LSTM (price-only)": f"{mk}_lstm_predictions.csv", "LSTM-X": f"{mk}_lstm_x_predictions.csv"}
    out = None
    for name, f in files.items():
        d = pd.read_csv(DATA / f, parse_dates=["date"]).set_index("date")
        if out is None:
            out = d[["actual_price"]].copy()
        out[name] = d.predicted_price
    out["Naive (last week)"] = out.actual_price.shift(1)
    out["gap_days"] = out.index.to_series().diff().dt.days
    return out


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


# ---------------- helpers ----------------
def grade(u: float) -> str:
    if u < 0.90: return "A"
    if u < 0.97: return "B"
    if u <= 1.03: return "C"
    if u <= 1.15: return "D"
    return "F"


def header(eyebrow: str, title: str, sub: str):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="ptitle">{title}</div><p class="psub">{sub}</p>', unsafe_allow_html=True)


def kpi(col, value, label, color=INK):
    col.markdown(f'<div class="kpi"><div class="v" style="color:{color}">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


def take(text: str):
    st.markdown(f'<div class="take">{text}</div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def layout(fig, h=360, hover="x unified"):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color=INK, size=13), legend=dict(orientation="h", y=1.12, x=0), hovermode=hover)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show(fig, h=360, hover="x unified"):
    st.plotly_chart(layout(fig, h, hover), width="stretch")


def acf(x, nlags=30):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    x = x - x.mean()
    d = (x * x).sum()
    return np.array([(x[k:] * x[:len(x) - k]).sum() / d for k in range(1, nlags + 1)]), 1.96 / np.sqrt(len(x))


def filters():
    """Current global sidebar selections."""
    ss = st.session_state
    return dict(market=ss.get("f_market", "Both"), freq=ss.get("f_freq", ["Daily", "Weekly", "Monthly"]),
                family=ss.get("f_family"), metric=ss.get("f_metric", "RMSE"),
                dates=ss.get("f_dates", (pd.Timestamp("2011-01-01").date(), pd.Timestamp("2025-12-31").date())))


def markets(f):
    return ["Indore", "Gondal"] if f["market"] == "Both" else [f["market"]]


def filtered_scorecard(f) -> pd.DataFrame:
    d = scorecard()
    d = d[d.market.isin(markets(f)) & d.frequency.isin(f["freq"])]
    if f["family"]:
        d = d[d.family.isin(f["family"])]
    col = {"RMSE": ("rmse", "naive_rmse"), "MAE": ("mae", "naive_mae"), "MAPE": ("mape", "naive_mape")}[f["metric"]]
    d = d.assign(value=d[col[0]], naive=d[col[1]])
    d["ratio"] = d.value / d.naive
    d["label"] = d.market + " · " + d.model + " (" + d.frequency + ")"
    return d.dropna(subset=["ratio"]).reset_index(drop=True)
