"""
GEX PRO Analyzer — v2.0 Full Revamp
Gamma · Delta · Vanna · Max Pain · 3D Visualization
Developed by @Gsnchez — bquantfinance.com

Features:
- GEX + DEX (Delta Exposure) dual analysis
- IV-weighted Gamma Profile
- Vanna & Charm exposure
- Gamma Heatmap (Strike × Expiry)
- 3D Three.js terrain visualization
- Enhanced pinning probability model
- Multi-ticker comparison
- Historical level tracking
- Professional glassmorphism dark UI
"""

import streamlit as st
import json, os, math
from datetime import timedelta, datetime
from typing import Tuple, Optional, Dict, List
import warnings
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GEX PRO | bquantfinance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# THEME & CSS — Glassmorphism dark theme with cyan/magenta accents
# ═══════════════════════════════════════════════════════════════════════════════
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;600;700;900&display=swap');

/* ── Global ── */
:root {
    --bg-primary: #06090F;
    --bg-secondary: #0B1120;
    --bg-card: rgba(11, 17, 32, 0.7);
    --border-subtle: rgba(0, 217, 255, 0.08);
    --border-glow: rgba(0, 217, 255, 0.25);
    --accent-cyan: #00D9FF;
    --accent-magenta: #FE53BB;
    --accent-gold: #FFD700;
    --accent-green: #00FF88;
    --accent-red: #FF4466;
    --text-primary: #E8ECF4;
    --text-secondary: #7A8BA8;
    --text-muted: #4A5568;
    --font-display: 'Outfit', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

.stApp {
    background: var(--bg-primary);
    font-family: var(--font-display);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080D18 0%, #0B1120 50%, #0D0A1A 100%);
    border-right: 1px solid var(--border-subtle);
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stTextInput label {
    font-family: var(--font-display);
    font-weight: 600;
    color: var(--text-secondary) !important;
    font-size: 13px;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

/* ── Cards / Glass panels ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-magenta), transparent);
    opacity: 0.5;
}

/* ── Headers ── */
h1, h2, h3 {
    font-family: var(--font-display) !important;
    font-weight: 900 !important;
    letter-spacing: -0.5px;
}

.hero-title {
    font-family: var(--font-display);
    font-size: 42px;
    font-weight: 900;
    letter-spacing: -1.5px;
    background: linear-gradient(135deg, #00D9FF 0%, #FFFFFF 40%, #FE53BB 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin: 0;
}

.hero-sub {
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 8px;
}

/* ── Cheat Sheet / Key Levels ── */
.levels-bar {
    display: flex;
    gap: 2px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 4px;
    margin: 16px 0;
    overflow: hidden;
}

.level-chip {
    flex: 1;
    padding: 12px 8px;
    text-align: center;
    border-radius: 10px;
    transition: all 0.2s ease;
    cursor: default;
}

.level-chip:hover {
    background: rgba(255,255,255,0.03);
}

.level-chip .label {
    font-family: var(--font-mono);
    font-size: 9px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
}

.level-chip .value {
    font-family: var(--font-display);
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.level-chip .delta {
    font-family: var(--font-mono);
    font-size: 11px;
    margin-top: 2px;
}

/* ── Regime Badge ── */
.regime-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 50px;
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.5px;
}

.regime-positive {
    background: rgba(0, 255, 136, 0.08);
    border: 1px solid rgba(0, 255, 136, 0.2);
    color: var(--accent-green);
}

.regime-negative {
    background: rgba(255, 68, 102, 0.08);
    border: 1px solid rgba(255, 68, 102, 0.2);
    color: var(--accent-red);
}

/* ── Metric Boxes ── */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
}

[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted) !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--border-subtle);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 13px;
    color: var(--text-secondary);
    padding: 8px 16px;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,217,255,0.12), rgba(254,83,187,0.12));
    color: var(--text-primary);
}

.stTabs [data-baseweb="tab-highlight"] {
    background: transparent !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: var(--font-display);
    font-weight: 700;
    letter-spacing: 0.5px;
    border: none;
    border-radius: 50px;
    padding: 12px 28px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-magenta));
    color: #fff;
    transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
    box-shadow: 0 4px 20px rgba(0, 217, 255, 0.15);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(254, 83, 187, 0.25);
}

/* ── DataFrames ── */
.stDataFrame {
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    overflow: hidden;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-magenta));
    border-radius: 8px;
}

/* ── Alerts ── */
.stAlert {
    border-radius: 12px !important;
    font-family: var(--font-display) !important;
    border: 1px solid var(--border-subtle) !important;
}

/* ── Dividers ── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-glow), transparent);
    margin: 24px 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

/* ── Animations ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 15px rgba(0,217,255,0.1); }
    50% { box-shadow: 0 0 25px rgba(0,217,255,0.2); }
}

.animate-in { animation: fadeIn 0.5s ease forwards; }
.glow-pulse { animation: pulse-glow 3s ease-in-out infinite; }

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    color: var(--accent-cyan) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}

/* ── Info/Warning/Success boxes ── */
div[data-testid="stAlert"] > div {
    font-family: var(--font-display) !important;
}

/* ── Hide full-screen button on charts ── */
button[title="View fullscreen"] { display: none; }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
CONTRACT_SIZE = 100
API_BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options"

# Plotly dark template config
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(6,9,15,0.8)",
    font=dict(family="Outfit, sans-serif", color="#7A8BA8"),
    title_font=dict(family="Outfit, sans-serif", color="#E8ECF4"),
    xaxis=dict(
        gridcolor="rgba(0,217,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.1)",
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor="rgba(0,217,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.1)",
        showgrid=True,
    ),
    hoverlabel=dict(
        bgcolor="#0B1120",
        bordercolor="rgba(0,217,255,0.3)",
        font=dict(family="JetBrains Mono", size=12, color="#E8ECF4"),
    ),
    margin=dict(l=60, r=30, t=60, b=50),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,217,255,0.08)",
        borderwidth=1,
        font=dict(size=11),
    ),
    hovermode="x unified",
)

COLORS = {
    "cyan": "#00D9FF",
    "magenta": "#FE53BB",
    "gold": "#FFD700",
    "green": "#00FF88",
    "red": "#FF4466",
    "blue": "#4488FF",
    "purple": "#A855F7",
    "orange": "#FF8C42",
    "teal": "#00BFA6",
    "white": "#E8ECF4",
    "muted": "#4A5568",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "ticker_data" not in st.session_state:
    st.session_state.ticker_data = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "compare_tickers" not in st.session_state:
    st.session_state.compare_tickers = []


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def fetch_option_data(ticker: str) -> Optional[dict]:
    """Fetch options data from CBOE API"""
    urls = [
        f"{API_BASE_URL}/_{ticker}.json",
        f"{API_BASE_URL}/{ticker}.json",
    ]
    for url in urls:
        try:
            response = requests.get(url, timeout=12)
            if response.status_code == 200:
                return response.json()
        except Exception:
            continue
    return None


def parse_option_data(raw_data: dict) -> Tuple[float, pd.DataFrame]:
    """Parse raw CBOE options data"""
    try:
        data = pd.DataFrame.from_dict(raw_data)
        spot_price = float(data.loc["current_price", "data"])
        option_data = pd.DataFrame(data.loc["options", "data"])
        return spot_price, option_data
    except Exception as e:
        st.error(f"Parse error: {e}")
        return 0, pd.DataFrame()


def process_option_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and process raw options data — fully vectorized"""
    df = data.copy()
    df["type"] = df.option.str.extract(r"\d([CP])\d")
    df["strike_raw"] = df.option.str.extract(r"[CP](\d+)").astype(float)
    df["strike"] = df["strike_raw"] / 1000
    df["expiration_str"] = df.option.str.extract(r"[A-Z]+(\d{6})")
    df["expiration"] = pd.to_datetime(df["expiration_str"], format="%y%m%d", errors="coerce")

    numeric_cols = ["gamma", "open_interest", "volume", "delta", "vega", "theta", "iv"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = (
        df["type"].notna()
        & df["strike"].notna()
        & df["expiration"].notna()
        & df["gamma"].notna()
        & df["open_interest"].notna()
        & (df["open_interest"] > 0)
        & (df["gamma"] > 0)
    )
    return df[mask].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# CALCULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_gex(spot: float, data: pd.DataFrame, dealer: str = "standard") -> pd.DataFrame:
    """Vectorized GEX calculation"""
    df = data.copy()
    df["GEX"] = df["gamma"] * df["open_interest"] * CONTRACT_SIZE * (spot**2) * 0.01
    if dealer == "standard":
        df["GEX"] = np.where(df["type"] == "P", -df["GEX"], df["GEX"])
    elif dealer == "inverse":
        df["GEX"] = np.where(df["type"] == "P", df["GEX"], -df["GEX"])
    df["days_to_expiry"] = (df["expiration"] - pd.Timestamp.now()).dt.days
    return df


def calculate_dex(spot: float, data: pd.DataFrame, dealer: str = "standard") -> pd.DataFrame:
    """Delta Exposure (DEX) calculation — directional hedging pressure"""
    df = data.copy()
    if "delta" not in df.columns or df["delta"].isna().all():
        df["DEX"] = 0.0
        return df
    df["DEX"] = df["delta"].abs() * df["open_interest"] * CONTRACT_SIZE * spot * 0.01
    if dealer == "standard":
        df["DEX"] = np.where(df["type"] == "P", -df["DEX"], df["DEX"])
    elif dealer == "inverse":
        df["DEX"] = np.where(df["type"] == "P", df["DEX"], -df["DEX"])
    return df


def calculate_vanna_exposure(data: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Approximate Vanna exposure from available greeks"""
    df = data.copy()
    if "vega" not in df.columns or "delta" not in df.columns:
        df["vanna_approx"] = 0
        return df
    # Vanna ≈ dDelta/dVol. We approximate via vega * delta / spot
    df["vanna_approx"] = (df["vega"] * df["delta"].abs() * df["open_interest"] * CONTRACT_SIZE / spot)
    df.loc[df["type"] == "P", "vanna_approx"] *= -1
    return df


def calculate_charm_exposure(data: pd.DataFrame) -> pd.DataFrame:
    """Approximate Charm (delta decay) from available greeks"""
    df = data.copy()
    if "delta" not in df.columns or "theta" not in df.columns:
        df["charm_approx"] = 0
        return df
    # Charm ≈ dDelta/dTime. Approximate via theta * delta
    df["charm_approx"] = df["theta"].abs() * df["delta"].abs() * df["open_interest"] * CONTRACT_SIZE
    df.loc[df["type"] == "P", "charm_approx"] *= -1
    return df


def calculate_gamma_profile(data: pd.DataFrame, spot: float, dealer: str = "standard") -> dict:
    """IV-weighted gamma profile with full vectorization"""
    strikes = np.sort(data["strike"].unique())
    profile_strikes = np.linspace(strikes.min(), strikes.max(), 250)

    option_strikes = data["strike"].values
    option_gammas = data["gamma"].values
    option_oi = data["open_interest"].values
    option_types = data["type"].values

    # Use IV for kernel width if available, else 5% of strike
    if "iv" in data.columns and data["iv"].notna().any():
        option_ivs = data["iv"].fillna(0.20).values
        widths = option_strikes * np.clip(option_ivs, 0.05, 1.0) * 0.5
    else:
        widths = option_strikes * 0.05

    distances = np.abs(profile_strikes[:, np.newaxis] - option_strikes[np.newaxis, :])
    gamma_contributions = option_gammas * np.exp(-(distances**2) / (2 * widths**2))
    gamma_values = gamma_contributions * option_oi * CONTRACT_SIZE * 0.01 / 1e9

    is_call = option_types == "C"
    is_put = option_types == "P"

    if dealer == "standard":
        gamma_values[:, is_put] *= -1
    elif dealer == "inverse":
        gamma_values[:, is_call] *= -1

    aggregate_gamma = np.sum(gamma_values, axis=1)
    call_gamma = np.sum(gamma_values[:, is_call], axis=1) if np.any(is_call) else np.zeros_like(profile_strikes)
    put_gamma = np.sum(gamma_values[:, is_put], axis=1) if np.any(is_put) else np.zeros_like(profile_strikes)

    # Gamma flip
    zero_crossings = np.where(np.diff(np.sign(aggregate_gamma)))[0]
    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        x1, x2 = profile_strikes[idx], profile_strikes[idx + 1]
        y1, y2 = aggregate_gamma[idx], aggregate_gamma[idx + 1]
        gamma_flip = x1 - y1 * (x2 - x1) / (y2 - y1) if (y2 - y1) != 0 else spot
    else:
        gamma_flip = spot

    # By expiry (top 4)
    unique_expiries = sorted(data["expiration"].unique())[:4]
    expiry_profiles = {}
    for expiry in unique_expiries:
        emask = data["expiration"] == expiry
        if not np.any(emask):
            continue
        es = data.loc[emask, "strike"].values
        eg = data.loc[emask, "gamma"].values
        eoi = data.loc[emask, "open_interest"].values
        et = data.loc[emask, "type"].values
        ed = np.abs(profile_strikes[:, np.newaxis] - es[np.newaxis, :])
        ew = es * 0.05
        ev = eg * np.exp(-(ed**2) / (2 * ew**2)) * eoi * CONTRACT_SIZE * 0.01 / 1e9
        if dealer == "standard":
            ev[:, et == "P"] *= -1
        elif dealer == "inverse":
            ev[:, et == "C"] *= -1
        expiry_profiles[expiry] = np.sum(ev, axis=1)

    return {
        "strikes": profile_strikes,
        "aggregate_gamma": aggregate_gamma,
        "call_gamma": call_gamma,
        "put_gamma": put_gamma,
        "gamma_flip": gamma_flip,
        "by_expiry": expiry_profiles,
    }


def calculate_max_pain(data: pd.DataFrame, spot: float) -> tuple:
    """Vectorized max pain calculation"""
    strikes = np.sort(data["strike"].unique())
    calls = data[data["type"] == "C"][["strike", "open_interest"]].values
    puts = data[data["type"] == "P"][["strike", "open_interest"]].values

    pain = {}
    for exp_price in strikes:
        c_mask = calls[:, 0] < exp_price
        c_pain = np.sum((exp_price - calls[c_mask, 0]) * calls[c_mask, 1] * CONTRACT_SIZE)
        p_mask = puts[:, 0] > exp_price
        p_pain = np.sum((puts[p_mask, 0] - exp_price) * puts[p_mask, 1] * CONTRACT_SIZE)
        pain[exp_price] = c_pain + p_pain

    if pain:
        mp_strike = min(pain, key=pain.get)
        return mp_strike, pain, pain[mp_strike]
    return spot, {}, 0


def calculate_pinning_probability(max_pain, spot, total_gex, days_to_exp, iv_mean=0.20):
    """Enhanced pinning probability using IV and GEX regime"""
    distance_pct = abs(max_pain - spot) / spot * 100

    # Expected move based on IV
    expected_move_pct = iv_mean * 100 * math.sqrt(max(days_to_exp, 1) / 365)

    # Base: how close is max pain relative to expected move
    if expected_move_pct > 0:
        closeness = 1 - min(distance_pct / expected_move_pct, 2) / 2
    else:
        closeness = 0.5

    base_prob = closeness * 50

    # Expiry boost
    if days_to_exp == 0:
        exp_boost = 30
    elif days_to_exp <= 1:
        exp_boost = 22
    elif days_to_exp <= 3:
        exp_boost = 15
    elif days_to_exp <= 7:
        exp_boost = 10
    else:
        exp_boost = 3

    # GEX regime boost: positive GEX = more pinning
    gex_boost = 12 if total_gex > 0 else -5

    # Distance penalty
    if distance_pct > 5:
        dist_penalty = -15
    elif distance_pct > 3:
        dist_penalty = -8
    elif distance_pct > 1:
        dist_penalty = 0
    else:
        dist_penalty = 10

    prob = min(95, max(5, base_prob + exp_boost + gex_boost + dist_penalty))

    return {
        "probability": round(prob),
        "direction": "BULLISH" if max_pain > spot else "BEARISH" if max_pain < spot else "NEUTRAL",
        "distance_pct": distance_pct,
        "expected_move": expected_move_pct,
    }


def compute_all_metrics(data: pd.DataFrame, spot: float) -> dict:
    """Compute all metrics in a single pass"""
    gex_by_strike = data.groupby("strike")["GEX"].sum()
    gex_by_type = data.groupby("type")["GEX"].sum()
    oi_by_type = data.groupby("type")["open_interest"].sum()
    vol_by_type = data.groupby("type")["volume"].sum() if "volume" in data.columns else pd.Series(dtype=float)

    total_gex = data["GEX"].sum() / 1e9
    call_gex = gex_by_type.get("C", 0) / 1e9
    put_gex = gex_by_type.get("P", 0) / 1e9

    # DEX metrics
    has_dex = "DEX" in data.columns and data["DEX"].notna().any()
    total_dex = (data["DEX"].sum() / 1e9) if has_dex else 0
    call_dex = (data[data["type"] == "C"]["DEX"].sum() / 1e9) if has_dex else 0
    put_dex = (data[data["type"] == "P"]["DEX"].sum() / 1e9) if has_dex else 0

    # IV metrics
    has_iv = "iv" in data.columns and data["iv"].notna().any()
    iv_mean = data["iv"].mean() if has_iv else 0.20
    iv_calls = data[data["type"] == "C"]["iv"].mean() if has_iv else 0
    iv_puts = data[data["type"] == "P"]["iv"].mean() if has_iv else 0

    # Volume metrics
    total_volume = data["volume"].sum() if "volume" in data.columns else 0
    call_vol = vol_by_type.get("C", 0) if not vol_by_type.empty else 0
    put_vol = vol_by_type.get("P", 0) if not vol_by_type.empty else 0

    metrics = {
        "total_gex": total_gex,
        "call_gex": call_gex,
        "put_gex": put_gex,
        "total_dex": total_dex,
        "call_dex": call_dex,
        "put_dex": put_dex,
        "call_oi": oi_by_type.get("C", 0),
        "put_oi": oi_by_type.get("P", 0),
        "total_oi": data["open_interest"].sum(),
        "total_volume": total_volume,
        "call_volume": call_vol,
        "put_volume": put_vol,
        "iv_mean": iv_mean,
        "iv_calls": iv_calls,
        "iv_puts": iv_puts,
        "iv_skew": iv_puts - iv_calls if has_iv else 0,
        "max_gex_strike": gex_by_strike.abs().idxmax() if len(gex_by_strike) > 0 else spot,
        "top_strikes": gex_by_strike.abs().nlargest(5).to_dict(),
        "nearest_expiry": data["expiration"].min(),
        "put_call_ratio": abs(put_gex / call_gex) if call_gex != 0 else 0,
        "vol_put_call": put_vol / call_vol if call_vol > 0 else 0,
        "days_to_expiry": max(0, (data["expiration"].min() - pd.Timestamp.now()).days)
            if pd.notna(data["expiration"].min()) else 0,
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CHART FACTORY
# ═══════════════════════════════════════════════════════════════════════════════
def _base_layout(**kwargs) -> dict:
    """Merge base chart layout with overrides"""
    layout = {**CHART_LAYOUT}
    layout.update(kwargs)
    return layout


def chart_gamma_profile(profiles: dict, spot: float, ticker: str) -> go.Figure:
    """Gamma exposure profile — aggregate + calls/puts"""
    fig = go.Figure()

    # Fill between zero and aggregate
    pos_mask = profiles["aggregate_gamma"] >= 0
    neg_mask = profiles["aggregate_gamma"] < 0

    fig.add_trace(go.Scatter(
        x=profiles["strikes"], y=profiles["aggregate_gamma"],
        mode="lines", name="Net Gamma",
        line=dict(color=COLORS["cyan"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,217,255,0.08)",
        hovertemplate="Strike: $%{x:.1f}<br>GEX: %{y:.3f}B<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=profiles["strikes"], y=profiles["call_gamma"],
        mode="lines", name="Call Gamma",
        line=dict(color=COLORS["green"], width=1.5, dash="dot"),
        opacity=0.6,
        hovertemplate="Strike: $%{x:.1f}<br>Call: %{y:.3f}B<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=profiles["strikes"], y=profiles["put_gamma"],
        mode="lines", name="Put Gamma",
        line=dict(color=COLORS["red"], width=1.5, dash="dot"),
        opacity=0.6,
        hovertemplate="Strike: $%{x:.1f}<br>Put: %{y:.3f}B<extra></extra>",
    ))

    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["gold"], line_width=2,
                  annotation_text=f"Spot ${spot:.2f}", annotation_font=dict(size=11, color=COLORS["gold"]))

    if profiles["gamma_flip"] != spot:
        fig.add_vline(x=profiles["gamma_flip"], line_dash="dot", line_color=COLORS["magenta"], line_width=2,
                      annotation_text=f"Flip ${profiles['gamma_flip']:.2f}",
                      annotation_position="bottom right",
                      annotation_font=dict(size=11, color=COLORS["magenta"]))

    fig.update_layout(**_base_layout(
        title=dict(text=f"Gamma Exposure Profile — {ticker}", font=dict(size=18)),
        yaxis_title="Gamma Exposure ($Bn / 1% move)",
        xaxis_title="Strike",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    ))
    return fig


def chart_gamma_by_expiry(profiles: dict, spot: float, ticker: str) -> go.Figure:
    """Gamma profile decomposed by expiration"""
    fig = go.Figure()
    exp_colors = [COLORS["cyan"], COLORS["magenta"], COLORS["green"], COLORS["gold"]]

    fig.add_trace(go.Scatter(
        x=profiles["strikes"], y=profiles["aggregate_gamma"],
        mode="lines", name="All Expiries",
        line=dict(color="rgba(255,255,255,0.5)", width=2),
    ))

    for i, (expiry, vals) in enumerate(profiles["by_expiry"].items()):
        fig.add_trace(go.Scatter(
            x=profiles["strikes"], y=vals,
            mode="lines", name=f"{expiry.strftime('%d %b')}",
            line=dict(color=exp_colors[i % len(exp_colors)], width=2),
            opacity=0.85,
        ))

    fig.add_hline(y=0, line_color="rgba(255,255,255,0.1)", line_width=1)
    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["gold"], line_width=1.5)

    fig.update_layout(**_base_layout(
        title=dict(text=f"Gamma by Expiration — {ticker}", font=dict(size=18)),
        yaxis_title="GEX ($Bn)", xaxis_title="Strike", height=480,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    ))
    return fig


def chart_max_pain(pain_by_strike: dict, max_pain: float, spot: float) -> go.Figure:
    """Max pain curve with clean design"""
    if not pain_by_strike:
        return go.Figure()
    strikes = sorted(pain_by_strike.keys())
    if len(strikes) > 120:
        step = len(strikes) // 120
        strikes = strikes[::step]
        if max_pain not in strikes:
            strikes.append(max_pain)
        strikes.sort()

    values = [pain_by_strike.get(s, 0) / 1e9 for s in strikes]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=strikes, y=values, mode="lines", fill="tozeroy",
        line=dict(color=COLORS["red"], width=2),
        fillcolor="rgba(255,68,102,0.1)",
        name="Pain Curve",
        hovertemplate="Strike: $%{x:.1f}<br>Pain: $%{y:.2f}B<extra></extra>",
    ))

    if max_pain in pain_by_strike:
        fig.add_trace(go.Scatter(
            x=[max_pain], y=[pain_by_strike[max_pain] / 1e9],
            mode="markers+text",
            marker=dict(size=14, color=COLORS["green"], symbol="diamond", line=dict(width=2, color="#fff")),
            text=["MAX PAIN"], textposition="top center",
            textfont=dict(size=13, color=COLORS["green"], family="Outfit"),
            showlegend=False,
        ))

    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["gold"], line_width=2,
                  annotation_text=f"Spot ${spot:.2f}")
    fig.add_vline(x=max_pain, line_dash="dot", line_color=COLORS["green"], line_width=1.5, opacity=0.6)

    dist = (max_pain - spot) / spot * 100
    arrow = "↑" if dist > 0 else "↓"

    fig.update_layout(**_base_layout(
        title=dict(text=f"Max Pain ${max_pain:.2f}  |  Spot ${spot:.2f}  |  {arrow} {abs(dist):.2f}%",
                   font=dict(size=18)),
        xaxis_title="Strike ($)", yaxis_title="Total Pain ($Bn)", height=460, showlegend=False,
    ))
    return fig


def chart_gex_by_strike(spot: float, data: pd.DataFrame, strike_range: float) -> go.Figure:
    """GEX bar chart by strike"""
    gex = data.groupby("strike")["GEX"].sum() / 1e9
    lo, hi = spot * (1 - strike_range / 100), spot * (1 + strike_range / 100)
    gex = gex[(gex.index >= lo) & (gex.index <= hi)]

    fig = go.Figure()
    colors = [COLORS["cyan"] if v > 0 else COLORS["magenta"] for v in gex.values]

    fig.add_trace(go.Bar(
        x=gex.index, y=gex.values, marker_color=colors, opacity=0.85,
        hovertemplate="Strike: $%{x:.1f}<br>GEX: %{y:.3f}B<extra></extra>",
    ))

    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["gold"], line_width=2)
    if not gex.empty:
        mg = gex.abs().idxmax()
        fig.add_vline(x=mg, line_dash="dot", line_color=COLORS["red"], line_width=1.5,
                      opacity=0.6, annotation_text=f"Max ${mg:.0f}")

    fig.update_layout(**_base_layout(
        title=dict(text="GEX by Strike", font=dict(size=18)),
        xaxis_title="Strike ($)", yaxis_title="GEX ($Bn)", height=450, showlegend=False,
    ))
    return fig


def chart_dex_by_strike(spot: float, data: pd.DataFrame, strike_range: float) -> go.Figure:
    """DEX bar chart by strike"""
    if "DEX" not in data.columns:
        return go.Figure()
    dex = data.groupby("strike")["DEX"].sum() / 1e9
    lo, hi = spot * (1 - strike_range / 100), spot * (1 + strike_range / 100)
    dex = dex[(dex.index >= lo) & (dex.index <= hi)]

    fig = go.Figure()
    colors = [COLORS["blue"] if v > 0 else COLORS["orange"] for v in dex.values]

    fig.add_trace(go.Bar(
        x=dex.index, y=dex.values, marker_color=colors, opacity=0.85,
        hovertemplate="Strike: $%{x:.1f}<br>DEX: %{y:.3f}B<extra></extra>",
    ))
    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["gold"], line_width=2)

    fig.update_layout(**_base_layout(
        title=dict(text="Delta Exposure (DEX) by Strike", font=dict(size=18)),
        xaxis_title="Strike ($)", yaxis_title="DEX ($Bn)", height=450, showlegend=False,
    ))
    return fig


def chart_gex_by_expiration(data: pd.DataFrame, max_days: int) -> go.Figure:
    """GEX by expiration date"""
    cutoff = datetime.now() + timedelta(days=max_days)
    filtered = data[data["expiration"] <= cutoff]
    gex = filtered.groupby("expiration")["GEX"].sum() / 1e9

    fig = go.Figure()
    colors = [COLORS["cyan"] if v > 0 else COLORS["magenta"] for v in gex.values]
    fig.add_trace(go.Bar(
        x=gex.index, y=gex.values, marker_color=colors, opacity=0.85,
        hovertemplate="%{x|%b %d}<br>GEX: %{y:.3f}B<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="GEX by Expiration", font=dict(size=18)),
        xaxis_title="Expiration", yaxis_title="GEX ($Bn)", height=450,
        xaxis=dict(tickformat="%b %d"), showlegend=False,
    ))
    return fig


def chart_calls_vs_puts(spot: float, data: pd.DataFrame) -> go.Figure:
    """Overlaid call/put GEX distribution"""
    calls = data[data["type"] == "C"].groupby("strike")["GEX"].sum() / 1e9
    puts = data[data["type"] == "P"].groupby("strike")["GEX"].sum() / 1e9

    fig = go.Figure()
    fig.add_trace(go.Bar(x=calls.index, y=calls.values, name="Calls",
                         marker_color=COLORS["cyan"], opacity=0.65))
    fig.add_trace(go.Bar(x=puts.index, y=puts.values, name="Puts",
                         marker_color=COLORS["magenta"], opacity=0.65))
    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["gold"], line_width=2)

    fig.update_layout(**_base_layout(
        title=dict(text="GEX Distribution — Calls vs Puts", font=dict(size=18)),
        xaxis_title="Strike ($)", yaxis_title="GEX ($Bn)", height=450, barmode="overlay",
    ))
    return fig


def chart_cumulative_gex(data: pd.DataFrame) -> go.Figure:
    """Cumulative GEX by days to expiry"""
    df = data.copy()
    df["dte"] = (df["expiration"] - datetime.now()).dt.days
    df = df[df["dte"] >= 0]
    gex_days = df.groupby("dte")["GEX"].sum().sort_index() / 1e9
    cumulative = gex_days.cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative.index, y=cumulative.values, mode="lines", fill="tozeroy",
        line=dict(color=COLORS["cyan"], width=2.5),
        fillcolor="rgba(0,217,255,0.08)",
    ))
    for d in [7, 30, 60, 90]:
        if d <= cumulative.index.max():
            fig.add_vline(x=d, line_dash="dot", line_color="rgba(255,255,255,0.15)",
                          annotation_text=f"{d}d", annotation_font=dict(size=10, color=COLORS["muted"]))

    fig.update_layout(**_base_layout(
        title=dict(text="Cumulative GEX by Time to Expiry", font=dict(size=18)),
        xaxis_title="Days to Expiry", yaxis_title="Cumulative GEX ($Bn)", height=450, showlegend=False,
    ))
    return fig


def chart_gamma_heatmap(data: pd.DataFrame, spot: float, strike_range: float) -> go.Figure:
    """2D Heatmap: Strike × Expiry colored by GEX intensity"""
    lo, hi = spot * (1 - strike_range / 100), spot * (1 + strike_range / 100)
    df = data[(data["strike"] >= lo) & (data["strike"] <= hi)].copy()

    # Round strikes to reduce resolution for readability
    n_unique = df["strike"].nunique()
    if n_unique > 50:
        # Bin strikes
        bins = np.linspace(lo, hi, 50)
        df["strike_bin"] = pd.cut(df["strike"], bins=bins, labels=bins[:-1]).astype(float)
    else:
        df["strike_bin"] = df["strike"]

    pivot = df.pivot_table(values="GEX", index="strike_bin", columns="expiration", aggfunc="sum", fill_value=0) / 1e6

    # Limit expiration columns
    if pivot.shape[1] > 20:
        pivot = pivot.iloc[:, :20]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[d.strftime("%b %d") for d in pivot.columns],
        y=[f"${s:.0f}" for s in pivot.index],
        colorscale=[
            [0.0, "#1a0033"],
            [0.25, "#4400aa"],
            [0.45, "#FE53BB"],
            [0.55, "#222"],
            [0.65, "#006688"],
            [0.75, "#00aacc"],
            [1.0, "#00D9FF"],
        ],
        zmid=0,
        colorbar=dict(title="GEX ($M)", titlefont=dict(size=11)),
        hovertemplate="Strike: %{y}<br>Expiry: %{x}<br>GEX: $%{z:.1f}M<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        title=dict(text="Gamma Heatmap — Strike × Expiry", font=dict(size=18)),
        xaxis_title="Expiration", yaxis_title="Strike", height=550,
        yaxis=dict(autorange=True),
    ))
    return fig


def chart_iv_smile(data: pd.DataFrame, spot: float) -> go.Figure:
    """IV smile for nearest expirations"""
    if "iv" not in data.columns or data["iv"].isna().all():
        fig = go.Figure()
        fig.update_layout(**_base_layout(title="IV data not available", height=400))
        return fig

    expiries = sorted(data["expiration"].unique())[:4]
    exp_colors = [COLORS["cyan"], COLORS["magenta"], COLORS["green"], COLORS["gold"]]

    fig = go.Figure()
    for i, exp in enumerate(expiries):
        edf = data[data["expiration"] == exp].sort_values("strike")
        calls = edf[edf["type"] == "C"]
        puts = edf[edf["type"] == "P"]

        # Merge IV from puts (below spot) and calls (above spot)
        merged = pd.concat([
            puts[puts["strike"] <= spot][["strike", "iv"]],
            calls[calls["strike"] > spot][["strike", "iv"]],
        ]).sort_values("strike").dropna(subset=["iv"])

        if merged.empty:
            continue

        fig.add_trace(go.Scatter(
            x=merged["strike"], y=merged["iv"] * 100,
            mode="lines+markers", name=f"{exp.strftime('%d %b')}",
            line=dict(color=exp_colors[i % len(exp_colors)], width=2),
            marker=dict(size=4),
        ))

    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["gold"], line_width=1.5)
    fig.update_layout(**_base_layout(
        title=dict(text="Implied Volatility Smile", font=dict(size=18)),
        xaxis_title="Strike", yaxis_title="IV (%)", height=450,
    ))
    return fig


def chart_vanna_charm(data: pd.DataFrame, spot: float, strike_range: float) -> go.Figure:
    """Vanna and Charm exposure combined chart"""
    lo, hi = spot * (1 - strike_range / 100), spot * (1 + strike_range / 100)
    df = data[(data["strike"] >= lo) & (data["strike"] <= hi)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Vanna Exposure", "Charm Exposure"))

    if "vanna_approx" in df.columns:
        vanna = df.groupby("strike")["vanna_approx"].sum() / 1e6
        vc = [COLORS["purple"] if v > 0 else COLORS["orange"] for v in vanna.values]
        fig.add_trace(go.Bar(x=vanna.index, y=vanna.values, marker_color=vc, opacity=0.8,
                             name="Vanna", showlegend=False), row=1, col=1)

    if "charm_approx" in df.columns:
        charm = df.groupby("strike")["charm_approx"].sum() / 1e6
        cc = [COLORS["teal"] if v > 0 else COLORS["red"] for v in charm.values]
        fig.add_trace(go.Bar(x=charm.index, y=charm.values, marker_color=cc, opacity=0.8,
                             name="Charm", showlegend=False), row=2, col=1)

    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["gold"], line_width=1.5)

    fig.update_layout(**_base_layout(
        title=dict(text="Vanna & Charm Exposure", font=dict(size=18)),
        height=600, showlegend=False,
    ))
    fig.update_yaxes(title_text="Vanna ($M)", row=1, col=1)
    fig.update_yaxes(title_text="Charm ($M)", row=2, col=1)
    fig.update_xaxes(title_text="Strike ($)", row=2, col=1)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3D VISUALIZATION — Embedded Three.js
# ═══════════════════════════════════════════════════════════════════════════════
def render_3d_terrain(data: pd.DataFrame, spot: float, strike_range: float):
    """Render 3D gamma terrain using Three.js embedded in Streamlit"""
    lo, hi = spot * (1 - strike_range / 100), spot * (1 + strike_range / 100)
    df = data[(data["strike"] >= lo) & (data["strike"] <= hi)].copy()

    # Build grid data
    df["dte"] = (df["expiration"] - pd.Timestamp.now()).dt.days
    df = df[df["dte"] >= 0]

    # Aggregate by strike bin and DTE
    strike_bins = np.linspace(lo, hi, 40)
    df["s_bin"] = pd.cut(df["strike"], bins=strike_bins, labels=strike_bins[:-1]).astype(float)
    dte_bins = np.arange(0, min(df["dte"].max() + 5, 95), 3)
    if len(dte_bins) < 2:
        dte_bins = np.arange(0, 30, 3)
    df["d_bin"] = pd.cut(df["dte"], bins=dte_bins, labels=dte_bins[:-1]).astype(float)

    pivot = df.pivot_table(values="GEX", index="s_bin", columns="d_bin", aggfunc="sum", fill_value=0) / 1e9

    grid_data = []
    for si, strike_val in enumerate(pivot.index):
        for di, dte_val in enumerate(pivot.columns):
            gex_val = float(pivot.iloc[si, di])
            grid_data.append({"s": float(strike_val), "d": float(dte_val), "g": round(gex_val, 4)})

    json_data = json.dumps({
        "grid": grid_data,
        "spot": spot,
        "nStrikes": len(pivot.index),
        "nDtes": len(pivot.columns),
        "strikes": [float(s) for s in pivot.index],
        "dtes": [float(d) for d in pivot.columns],
    })

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ margin:0; overflow:hidden; background:#06090F; font-family:'Outfit',sans-serif; }}
        canvas {{ display:block; }}
        #info {{
            position:absolute; top:12px; left:16px; color:#7A8BA8; font-size:11px;
            background:rgba(6,9,15,0.85); padding:10px 16px; border-radius:10px;
            border:1px solid rgba(0,217,255,0.1); backdrop-filter:blur(10px);
            pointer-events:none;
        }}
        #info .title {{ color:#E8ECF4; font-weight:700; font-size:14px; margin-bottom:4px;
            background:linear-gradient(90deg,#00D9FF,#FE53BB);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        }}
        #legend {{
            position:absolute; bottom:12px; left:16px; color:#4A5568; font-size:10px;
            background:rgba(6,9,15,0.8); padding:8px 14px; border-radius:8px;
            border:1px solid rgba(0,217,255,0.06);
        }}
        #legend span {{ margin-right:12px; }}
        #controls {{
            position:absolute; top:12px; right:16px; display:flex; gap:6px;
        }}
        #controls button {{
            background:rgba(11,17,32,0.8); border:1px solid rgba(0,217,255,0.15);
            color:#7A8BA8; padding:6px 12px; border-radius:16px; font-size:11px;
            cursor:pointer; font-family:'Outfit',sans-serif;
        }}
        #controls button:hover {{ border-color:rgba(0,217,255,0.4); color:#E8ECF4; }}
        #controls button.active {{ background:rgba(0,217,255,0.12); color:#00D9FF; }}
    </style>
    </head>
    <body>
    <div id="info">
        <div class="title">3D Gamma Terrain</div>
        <div>X: Strike · Z: Days to Expiry · Y: GEX</div>
        <div style="margin-top:4px;color:#FFD700;">Spot: ${spot:.2f}</div>
    </div>
    <div id="legend">
        <span style="color:#00D9FF;">■ Positive GEX</span>
        <span style="color:#FE53BB;">■ Negative GEX</span>
        <span style="color:#FFD700;">● Spot</span>
    </div>
    <div id="controls">
        <button id="btnRotate" class="active" onclick="toggleRotate()">⟳ Auto-Rotate</button>
        <button onclick="resetCamera()">↺ Reset View</button>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    const DATA = {json_data};
    let autoRotate = true;

    // Scene setup
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x06090F, 0.015);
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.1, 200);
    const renderer = new THREE.WebGLRenderer({{ antialias:true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x06090F);
    document.body.appendChild(renderer.domElement);

    // Lights
    scene.add(new THREE.AmbientLight(0x334466, 0.7));
    const dl = new THREE.DirectionalLight(0xffffff, 0.7);
    dl.position.set(10, 20, 10);
    scene.add(dl);
    const pl1 = new THREE.PointLight(0x00D9FF, 1.2, 40);
    pl1.position.set(-10, 8, -10);
    scene.add(pl1);
    const pl2 = new THREE.PointLight(0xFE53BB, 1.2, 40);
    pl2.position.set(10, 8, 10);
    scene.add(pl2);

    // Grid
    const grid = new THREE.GridHelper(30, 30, 0x111825, 0x0D1220);
    grid.position.y = -0.05;
    scene.add(grid);

    // Build terrain
    const nS = DATA.nStrikes, nD = DATA.nDtes;
    const W = 20, D = 14;
    const geo = new THREE.PlaneGeometry(W, D, nS-1, nD-1);
    const cols = new Float32Array(geo.attributes.position.count * 3);

    // Build lookup
    const lookup = {{}};
    DATA.grid.forEach(p => {{ lookup[p.s + ',' + p.d] = p.g; }});
    const maxG = Math.max(...DATA.grid.map(p => Math.abs(p.g)), 0.001);

    for (let j = 0; j < nD; j++) {{
        for (let i = 0; i < nS; i++) {{
            const idx = j * nS + i;
            const gex = lookup[DATA.strikes[i] + ',' + DATA.dtes[j]] || 0;
            const h = (gex / maxG) * 5;
            const pi3 = idx * 3;
            geo.attributes.position.array[pi3 + 2] = -h;

            // Color
            const n = gex / maxG;
            if (n > 0) {{
                cols[pi3] = 0; cols[pi3+1] = 0.3 + Math.min(n,1)*0.7; cols[pi3+2] = 0.7 + Math.min(n,1)*0.3;
            }} else {{
                const a = Math.min(-n, 1);
                cols[pi3] = 0.7 + a*0.3; cols[pi3+1] = 0.1; cols[pi3+2] = 0.4 + a*0.35;
            }}
        }}
    }}

    geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    geo.computeVertexNormals();

    const mat = new THREE.MeshPhongMaterial({{
        vertexColors: true, side: THREE.DoubleSide,
        shininess: 70, transparent: true, opacity: 0.92
    }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    scene.add(mesh);

    // Wireframe
    const wm = new THREE.MeshBasicMaterial({{ color:0xffffff, wireframe:true, transparent:true, opacity:0.04 }});
    const wMesh = new THREE.Mesh(geo.clone(), wm);
    wMesh.rotation.x = -Math.PI / 2;
    scene.add(wMesh);

    // Spot marker
    const spotI = DATA.strikes.findIndex(s => s >= DATA.spot);
    const spotX = -W/2 + (spotI / (nS-1)) * W;
    const sg = new THREE.SphereGeometry(0.25, 32, 32);
    const sm = new THREE.MeshBasicMaterial({{ color:0xFFD700, transparent:true, opacity:0.9 }});
    const sMesh = new THREE.Mesh(sg, sm);
    sMesh.position.set(spotX, 3, 0);
    scene.add(sMesh);

    // Spot glow
    const gg = new THREE.SphereGeometry(0.5, 32, 32);
    const gm = new THREE.MeshBasicMaterial({{ color:0xFFD700, transparent:true, opacity:0.12 }});
    const gMesh = new THREE.Mesh(gg, gm);
    gMesh.position.set(spotX, 3, 0);
    scene.add(gMesh);

    // Spot beam
    const bg = new THREE.CylinderGeometry(0.015, 0.015, 7, 8);
    const bm = new THREE.MeshBasicMaterial({{ color:0xFFD700, transparent:true, opacity:0.2 }});
    const bMesh = new THREE.Mesh(bg, bm);
    bMesh.position.set(spotX, 0, 0);
    scene.add(bMesh);

    // Camera orbit
    let theta = 0.7, phi = 0.55, radius = 22;
    let mouseDown = false, lastX = 0, lastY = 0;

    camera.position.set(15, 12, 15);
    camera.lookAt(0, 1, 0);

    renderer.domElement.addEventListener('mousedown', e => {{ mouseDown=true; lastX=e.clientX; lastY=e.clientY; }});
    renderer.domElement.addEventListener('mousemove', e => {{
        if (!mouseDown) return;
        theta -= (e.clientX - lastX) * 0.005;
        phi = Math.max(0.15, Math.min(1.4, phi + (e.clientY - lastY) * 0.005));
        lastX = e.clientX; lastY = e.clientY;
    }});
    renderer.domElement.addEventListener('mouseup', () => mouseDown=false);
    renderer.domElement.addEventListener('mouseleave', () => mouseDown=false);
    renderer.domElement.addEventListener('wheel', e => {{
        radius = Math.max(8, Math.min(40, radius + e.deltaY * 0.015));
    }});

    // Touch support
    renderer.domElement.addEventListener('touchstart', e => {{
        if (e.touches.length === 1) {{ mouseDown=true; lastX=e.touches[0].clientX; lastY=e.touches[0].clientY; }}
    }});
    renderer.domElement.addEventListener('touchmove', e => {{
        if (!mouseDown || e.touches.length !== 1) return;
        theta -= (e.touches[0].clientX - lastX) * 0.005;
        phi = Math.max(0.15, Math.min(1.4, phi + (e.touches[0].clientY - lastY) * 0.005));
        lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
    }});
    renderer.domElement.addEventListener('touchend', () => mouseDown=false);

    function toggleRotate() {{
        autoRotate = !autoRotate;
        document.getElementById('btnRotate').classList.toggle('active', autoRotate);
    }}
    function resetCamera() {{ theta=0.7; phi=0.55; radius=22; }}

    let t = 0;
    function animate() {{
        requestAnimationFrame(animate);
        t += 0.01;
        if (autoRotate) theta += 0.0015;

        camera.position.set(
            radius * Math.sin(phi) * Math.cos(theta),
            Math.max(2, radius * Math.cos(phi)),
            radius * Math.sin(phi) * Math.sin(theta)
        );
        camera.lookAt(0, 1, 0);

        // Pulse spot
        sMesh.position.y = 3 + Math.sin(t * 2) * 0.15;
        gMesh.position.y = sMesh.position.y;
        gMesh.scale.setScalar(1 + Math.sin(t * 3) * 0.1);

        renderer.render(scene, camera);
    }}
    animate();

    window.addEventListener('resize', () => {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }});
    </script>
    </body>
    </html>
    """
    components.html(html, height=550, scrolling=False)


# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════
def render_hero():
    """App header"""
    st.markdown("""
    <div style="text-align:center; padding:20px 0 10px;">
        <div class="hero-title">⚡ GEX PRO</div>
        <div class="hero-sub">Gamma · Delta · Vanna · Max Pain · 3D</div>
        <div style="margin-top:8px;">
            <span style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">
                @Gsnchez — bquantfinance.com
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_key_levels(spot, max_pain, gamma_flip, metrics):
    """Top-level cheat sheet bar with all key levels"""
    gex = metrics["total_gex"]
    dex = metrics["total_dex"]

    mp_delta = (max_pain - spot) / spot * 100
    gf_delta = (gamma_flip - spot) / spot * 100
    mg_strike = metrics["max_gex_strike"]
    mg_delta = (mg_strike - spot) / spot * 100

    # Regime
    if gex > 0:
        regime_html = '<span class="regime-badge regime-positive">⬢ POSITIVE GAMMA</span>'
    else:
        regime_html = '<span class="regime-badge regime-negative">⬡ NEGATIVE GAMMA</span>'

    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; margin-bottom:8px;">
        <div>{regime_html}</div>
        <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">
            IV Mean: {metrics['iv_mean']*100:.1f}% &nbsp;|&nbsp;
            IV Skew: {metrics['iv_skew']*100:+.1f}% &nbsp;|&nbsp;
            P/C Vol: {metrics['vol_put_call']:.2f} &nbsp;|&nbsp;
            OI: {metrics['total_oi']:,.0f}
        </div>
    </div>

    <div class="levels-bar glow-pulse">
        <div class="level-chip">
            <div class="label">Spot</div>
            <div class="value" style="color:#FFD700;">${spot:.2f}</div>
        </div>
        <div class="level-chip">
            <div class="label">Max Pain</div>
            <div class="value" style="color:#00FF88;">${max_pain:.2f}</div>
            <div class="delta" style="color:{'#00FF88' if mp_delta >= 0 else '#FF4466'};">{mp_delta:+.2f}%</div>
        </div>
        <div class="level-chip">
            <div class="label">Gamma Flip</div>
            <div class="value" style="color:#FE53BB;">${gamma_flip:.2f}</div>
            <div class="delta" style="color:{'#00FF88' if gf_delta >= 0 else '#FF4466'};">{gf_delta:+.2f}%</div>
        </div>
        <div class="level-chip">
            <div class="label">Max GEX Strike</div>
            <div class="value" style="color:#00D9FF;">${mg_strike:.2f}</div>
            <div class="delta" style="color:{'#00FF88' if mg_delta >= 0 else '#FF4466'};">{mg_delta:+.2f}%</div>
        </div>
        <div class="level-chip">
            <div class="label">Net GEX</div>
            <div class="value" style="color:{'#00FF88' if gex > 0 else '#FF4466'};">${gex:.3f}B</div>
        </div>
        <div class="level-chip">
            <div class="label">Net DEX</div>
            <div class="value" style="color:{'#4488FF' if dex > 0 else '#FF8C42'};">${dex:.3f}B</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_probability_panel(prob_data, max_pain, spot, metrics):
    """Pin probability analysis panel"""
    prob = prob_data["probability"]
    if prob >= 70:
        color, status = "#00FF88", "STRONG SIGNAL"
    elif prob >= 50:
        color, status = "#FFD700", "MODERATE"
    else:
        color, status = "#FF4466", "WEAK"

    direction = prob_data["direction"]
    dist = prob_data["distance_pct"]
    exp_move = prob_data["expected_move"]

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:48px; font-weight:900; color:{color}; font-family:var(--font-display);">{prob}%</div>
            <div style="color:{color}; font-size:12px; font-weight:600; letter-spacing:1px;">{status}</div>
            <div style="color:var(--text-muted); font-size:11px; margin-top:8px;">Pin Probability</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Progress bar
        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                <div>
                    <span style="color:var(--text-muted); font-size:11px; font-family:var(--font-mono); text-transform:uppercase; letter-spacing:1px;">Direction</span><br/>
                    <span style="color:{color}; font-size:20px; font-weight:700;">{direction}</span>
                </div>
                <div style="text-align:right;">
                    <span style="color:var(--text-muted); font-size:11px; font-family:var(--font-mono); text-transform:uppercase; letter-spacing:1px;">Distance</span><br/>
                    <span style="color:var(--text-primary); font-size:20px; font-weight:700;">{dist:.2f}%</span>
                </div>
            </div>
            <div style="width:100%; height:8px; background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden;">
                <div style="width:{prob}%; height:100%; background:linear-gradient(90deg, {color}88, {color}); border-radius:4px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:10px; color:var(--text-muted); font-family:var(--font-mono);">
                <span>0%</span><span>50%</span><span>100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        speed = "SLOW (Mean Reversion)" if metrics["total_gex"] > 0 else "FAST (Momentum)"
        dte = metrics["days_to_expiry"]
        st.markdown(f"""
        <div class="glass-card">
            <div style="margin-bottom:8px;">
                <span style="color:var(--text-muted); font-size:10px; font-family:var(--font-mono); text-transform:uppercase; letter-spacing:1px;">Expected Move</span><br/>
                <span style="color:var(--text-primary); font-size:18px; font-weight:700;">±{exp_move:.2f}%</span>
            </div>
            <div style="margin-bottom:8px;">
                <span style="color:var(--text-muted); font-size:10px; font-family:var(--font-mono); text-transform:uppercase; letter-spacing:1px;">Speed</span><br/>
                <span style="color:var(--text-secondary); font-size:12px;">{speed}</span>
            </div>
            <div>
                <span style="color:var(--text-muted); font-size:10px; font-family:var(--font-mono); text-transform:uppercase; letter-spacing:1px;">Nearest Expiry</span><br/>
                <span style="color:{'#00FF88' if dte <= 1 else 'var(--text-primary)'}; font-size:18px; font-weight:700;">
                    {'0DTE ⚡' if dte == 0 else f'{dte}d'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Sidebar with configuration"""
    with st.sidebar:
        st.markdown("""
        <div style="padding:10px 0;">
            <div style="font-family:var(--font-display); font-size:22px; font-weight:900;
                        background:linear-gradient(135deg,#00D9FF,#FE53BB);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                ⚡ GEX PRO
            </div>
            <div style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted);
                        letter-spacing:2px; text-transform:uppercase; margin-top:2px;">
                Configuration
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        ticker = st.text_input("TICKER", value="SPY", help="e.g. SPY, QQQ, AAPL, TSLA, NVDA").upper()

        st.markdown("##### Analysis Parameters")
        strike_range = st.slider("Strike Range (%)", 5, 50, 10, 5)
        max_exp_days = st.slider("Max Expiry (days)", 7, 180, 60, 7)
        min_oi = st.number_input("Min Open Interest", 0, 10000, 500, 100)

        st.markdown("##### Dealer Positioning")
        dealer = st.selectbox("Assumption", ["standard", "inverse", "neutral"],
            format_func=lambda x: {
                "standard": "Standard (Short Puts, Long Calls)",
                "inverse": "Inverse (Long Puts, Short Calls)",
                "neutral": "Neutral",
            }[x])

        st.markdown("---")

        # Compare tickers
        st.markdown("##### Multi-Ticker Compare")
        compare_input = st.text_input("Add tickers (comma-separated)", placeholder="QQQ, AAPL, TSLA")

        analyze = st.button("⚡ ANALYZE", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; padding:10px 0;">
            <a href="https://bquantfinance.com" style="color:var(--accent-cyan); text-decoration:none;
                     font-family:var(--font-mono); font-size:11px;">bquantfinance.com</a><br/>
            <a href="https://twitter.com/Gsnchez" style="color:var(--accent-magenta); text-decoration:none;
                     font-family:var(--font-mono); font-size:11px;">@Gsnchez</a>
        </div>
        """, unsafe_allow_html=True)

        return ticker, strike_range, max_exp_days, min_oi, dealer, analyze, compare_input


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
def display_results(ticker, spot, data, strike_range, max_exp_days, dealer):
    """Main results display"""

    # ── Calculations ──
    data = calculate_gex(spot, data, dealer)
    data = calculate_dex(spot, data, dealer)
    data = calculate_vanna_exposure(data, spot)
    data = calculate_charm_exposure(data)

    metrics = compute_all_metrics(data, spot)
    max_pain, pain_by_strike, _ = calculate_max_pain(data, spot)
    profiles = calculate_gamma_profile(data, spot, dealer)
    prob = calculate_pinning_probability(max_pain, spot, metrics["total_gex"],
                                         metrics["days_to_expiry"], metrics["iv_mean"])

    # ── Key Levels Bar ──
    render_key_levels(spot, max_pain, profiles["gamma_flip"], metrics)

    st.markdown("---")

    # ── Interpretation Panel ──
    col_l, col_r = st.columns(2)
    with col_l:
        if metrics["total_gex"] > 0:
            st.markdown(f"""
            <div class="glass-card">
                <div style="color:#00FF88; font-weight:700; font-size:15px; margin-bottom:8px;">⬢ POSITIVE GAMMA REGIME</div>
                <div style="color:var(--text-secondary); font-size:13px; line-height:1.7;">
                    Dealers are <b style="color:#E8ECF4;">long gamma</b> — they sell into rallies & buy dips.<br/>
                    Expect <b style="color:#00FF88;">compressed volatility</b> and mean-reverting price action.<br/>
                    Support/resistance levels are more likely to hold.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="glass-card">
                <div style="color:#FF4466; font-weight:700; font-size:15px; margin-bottom:8px;">⬡ NEGATIVE GAMMA REGIME</div>
                <div style="color:var(--text-secondary); font-size:13px; line-height:1.7;">
                    Dealers are <b style="color:#E8ECF4;">short gamma</b> — they chase momentum in both directions.<br/>
                    Expect <b style="color:#FF4466;">amplified volatility</b> and trending moves.<br/>
                    Breakouts are more likely to sustain.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        direction = "BULLISH" if max_pain > spot else "BEARISH" if max_pain < spot else "AT EQUILIBRIUM"
        d_color = "#00FF88" if max_pain > spot else "#FF4466" if max_pain < spot else "#FFD700"
        speed = "slow (mean-reversion)" if metrics["total_gex"] > 0 else "fast (momentum-driven)"
        st.markdown(f"""
        <div class="glass-card">
            <div style="color:{d_color}; font-weight:700; font-size:15px; margin-bottom:8px;">
                {'↑' if max_pain > spot else '↓' if max_pain < spot else '='} MAX PAIN: {direction}
            </div>
            <div style="color:var(--text-secondary); font-size:13px; line-height:1.7;">
                Target: <b style="color:#E8ECF4;">${max_pain:.2f}</b> ({(max_pain-spot)/spot*100:+.2f}%)<br/>
                Expected move speed: <b style="color:#E8ECF4;">{speed}</b><br/>
                IV-implied range: <b style="color:#E8ECF4;">±{prob['expected_move']:.2f}%</b> ({metrics['days_to_expiry']}d)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Tabs — 7 focused tabs preserving original core ──
    tabs = st.tabs([
        "🎯 Max Pain",
        "📊 GEX Analysis",
        "📈 Gamma Profile",
        "🔥 Heatmap & 3D",
        "⚡ DEX & Greeks",
        "🌊 IV Surface",
        "📋 Data",
    ])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1 — MAX PAIN (original core + enhanced probability)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[0]:
        st.plotly_chart(chart_max_pain(pain_by_strike, max_pain, spot), use_container_width=True)
        st.markdown("#### Pin Probability Analysis")
        render_probability_panel(prob, max_pain, spot, metrics)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2 — GEX ANALYSIS (merges original: by strike, by expiry, calls/puts, cumulative)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[1]:
        # Primary chart: GEX by Strike
        st.plotly_chart(chart_gex_by_strike(spot, data, strike_range), use_container_width=True)

        # Top strikes table
        st.markdown("##### Top GEX Strikes")
        top_cols = st.columns(5)
        for i, (strike_val, gex_val) in enumerate(metrics["top_strikes"].items()):
            if i >= 5:
                break
            dist = (strike_val - spot) / spot * 100
            with top_cols[i]:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; padding:12px;">
                    <div style="color:var(--accent-cyan); font-size:18px; font-weight:700;">${strike_val:.0f}</div>
                    <div style="color:var(--text-secondary); font-size:12px;">${gex_val/1e9:.3f}B</div>
                    <div style="color:{'#00FF88' if dist >= 0 else '#FF4466'}; font-size:11px; font-family:var(--font-mono);">{dist:+.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # Secondary charts: Expiry + Calls vs Puts side by side
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_gex_by_expiration(data, max_exp_days), use_container_width=True)
        with c2:
            st.plotly_chart(chart_calls_vs_puts(spot, data), use_container_width=True)

        # Cumulative GEX + Period breakdown
        st.plotly_chart(chart_cumulative_gex(data), use_container_width=True)

        st.markdown("##### GEX by Period")
        dtemp = data.copy()
        dtemp["d"] = (dtemp["expiration"] - datetime.now()).dt.days
        periods = {"0-7d": (0, 7), "7-30d": (7, 30), "30-60d": (30, 60), "60-90d": (60, 90), "90d+": (90, 999)}
        pcols = st.columns(5)
        for i, (label, (p_lo, p_hi)) in enumerate(periods.items()):
            mask = (dtemp["d"] >= p_lo) & (dtemp["d"] < p_hi)
            val = dtemp[mask]["GEX"].sum() / 1e9
            with pcols[i]:
                st.metric(label, f"${val:.3f}B")

        # OI summary from original
        st.markdown("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="glass-card">
                <div style="color:var(--accent-cyan); font-weight:700; margin-bottom:6px;">📈 CALLS</div>
                <div style="color:var(--text-secondary); font-size:13px; line-height:1.8;">
                    GEX: <b style="color:var(--text-primary);">${metrics['call_gex']:.3f}B</b><br/>
                    OI: <b style="color:var(--text-primary);">{metrics['call_oi']:,.0f}</b><br/>
                    Volume: <b style="color:var(--text-primary);">{metrics['call_volume']:,.0f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="glass-card">
                <div style="color:var(--accent-magenta); font-weight:700; margin-bottom:6px;">📉 PUTS</div>
                <div style="color:var(--text-secondary); font-size:13px; line-height:1.8;">
                    GEX: <b style="color:var(--text-primary);">${metrics['put_gex']:.3f}B</b><br/>
                    OI: <b style="color:var(--text-primary);">{metrics['put_oi']:,.0f}</b><br/>
                    Volume: <b style="color:var(--text-primary);">{metrics['put_volume']:,.0f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3 — GAMMA PROFILE (original + IV-weighted improvement)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[2]:
        st.plotly_chart(chart_gamma_profile(profiles, spot, ticker), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Support / Resistance from Gamma Peaks")
            gamma_arr = profiles["aggregate_gamma"]
            strikes_arr = profiles["strikes"]
            peaks = []
            for i in range(1, len(gamma_arr) - 1):
                if gamma_arr[i] > gamma_arr[i - 1] and gamma_arr[i] > gamma_arr[i + 1] and abs(gamma_arr[i]) > 0.05:
                    peaks.append((strikes_arr[i], gamma_arr[i]))
            peaks.sort(key=lambda x: abs(x[1]), reverse=True)
            if peaks:
                for s_val, g_val in peaks[:5]:
                    icon = "🟢 Support" if g_val > 0 else "🔴 Resistance"
                    st.markdown(f"**{icon}** @ ${s_val:.2f} — {abs(g_val):.3f}B")
            else:
                st.markdown("*No significant gamma peaks detected in range.*")

            # Gamma profile interpretation (from original)
            max_gamma = max(gamma_arr)
            min_gamma = min(gamma_arr)
            st.markdown("")
            if max_gamma > abs(min_gamma):
                st.markdown(f"""
                <div class="glass-card">
                    <div style="color:#00FF88; font-weight:600; font-size:13px;">Positive Gamma Dominant</div>
                    <div style="color:var(--text-secondary); font-size:12px; margin-top:4px;">
                        Market tends toward stability. Mean reversion expected. Compressed volatility.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="color:#FF4466; font-weight:600; font-size:13px;">Negative Gamma Dominant</div>
                    <div style="color:var(--text-secondary); font-size:12px; margin-top:4px;">
                        Market prone to sharp moves. Range breakouts likely. Expanded volatility.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with c2:
            st.plotly_chart(chart_gamma_by_expiry(profiles, spot, ticker), use_container_width=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 4 — HEATMAP & 3D (new features combined)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[3]:
        st.plotly_chart(chart_gamma_heatmap(data, spot, strike_range), use_container_width=True)
        st.markdown("""
        <div style="color:var(--text-muted); font-size:12px; font-family:var(--font-mono); margin-bottom:16px;">
            Gamma walls shown as bright zones. Cyan = positive (support). Magenta = negative (vulnerability).
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 🏔️ 3D Gamma Terrain")
        st.markdown("""
        <div style="color:var(--text-muted); font-size:12px; margin-bottom:8px; font-family:var(--font-mono);">
            🖱️ Drag to orbit · Scroll to zoom · Touch supported
        </div>
        """, unsafe_allow_html=True)
        render_3d_terrain(data, spot, strike_range)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 5 — DEX & GREEKS (new: delta exposure + vanna + charm)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[4]:
        # DEX by strike
        st.plotly_chart(chart_dex_by_strike(spot, data, strike_range), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Net DEX", f"${metrics['total_dex']:.3f}B")
        with c2:
            st.metric("Call DEX", f"${metrics['call_dex']:.3f}B")
        with c3:
            st.metric("Put DEX", f"${metrics['put_dex']:.3f}B")

        st.markdown("""
        <div class="glass-card" style="margin:16px 0;">
            <div style="color:var(--text-secondary); font-size:13px; line-height:1.7;">
                <b style="color:var(--text-primary);">Delta Exposure (DEX)</b> — directional hedging pressure from dealers.<br/>
                <b style="color:#4488FF;">Positive DEX</b> → upward pressure (dealers buy as price rises) &nbsp;|&nbsp;
                <b style="color:#FF8C42;">Negative DEX</b> → downward pressure (dealers sell as price falls)<br/>
                Combine with GEX: <b>GEX</b> = volatility regime, <b>DEX</b> = directional bias.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Vanna & Charm
        st.markdown("---")
        st.plotly_chart(chart_vanna_charm(data, spot, strike_range), use_container_width=True)
        st.markdown("""
        <div class="glass-card">
            <div style="color:var(--text-secondary); font-size:13px; line-height:1.7;">
                <b style="color:#A855F7;">Vanna</b> = sensitivity of delta to IV changes.
                When IV drops, vanna flows push dealers to buy (positive) or sell (negative).<br/>
                <b style="color:#00BFA6;">Charm</b> = delta decay over time.
                Shows how hedging pressure shifts as time passes — critical for OPEX dynamics.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 6 — IV SURFACE (new)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[5]:
        st.plotly_chart(chart_iv_smile(data, spot), use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("IV Mean", f"{metrics['iv_mean']*100:.1f}%")
        with c2:
            st.metric("Call IV", f"{metrics['iv_calls']*100:.1f}%")
        with c3:
            st.metric("Put IV", f"{metrics['iv_puts']*100:.1f}%")
        with c4:
            skew = metrics["iv_skew"] * 100
            st.metric("IV Skew", f"{skew:+.1f}%",
                      delta="Put Premium" if skew > 0 else "Call Premium",
                      delta_color="inverse" if skew > 0 else "normal")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 7 — DATA (original: table + CSV download)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[6]:
        st.markdown("##### Options Data")
        c1, c2, c3 = st.columns(3)
        with c1:
            type_f = st.selectbox("Type", ["All", "Calls", "Puts"])
        with c2:
            sort_f = st.selectbox("Sort by", ["GEX", "DEX", "open_interest", "volume", "strike", "iv"])
        with c3:
            n_rows = st.number_input("Rows", 10, 100, 25)

        ddf = data.copy()
        if type_f == "Calls":
            ddf = ddf[ddf["type"] == "C"]
        elif type_f == "Puts":
            ddf = ddf[ddf["type"] == "P"]
        ddf = ddf.sort_values(sort_f, ascending=False, key=abs).head(n_rows)

        display_cols = ["option", "type", "strike", "expiration", "GEX", "DEX", "open_interest", "volume", "iv", "delta", "gamma"]
        display_cols = [c for c in display_cols if c in ddf.columns]
        show_df = ddf[display_cols].copy()
        show_df["GEX"] = show_df["GEX"].apply(lambda x: f"${x/1e6:.1f}M")
        if "DEX" in show_df.columns:
            show_df["DEX"] = show_df["DEX"].apply(lambda x: f"${x/1e6:.1f}M")
        show_df["expiration"] = show_df["expiration"].dt.strftime("%Y-%m-%d")
        if "iv" in show_df.columns:
            show_df["iv"] = show_df["iv"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")

        st.dataframe(show_df, use_container_width=True, height=500)

        csv = ddf.to_csv(index=False)
        st.download_button("⬇ Download CSV", data=csv,
                          file_name=f"{ticker}_gex_pro_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                          mime="text/csv")


def show_landing():
    """Landing page / educational content"""
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:40px;">
        <div style="font-size:64px; margin-bottom:16px;">⚡</div>
        <div class="hero-title" style="font-size:36px;">Professional Options Flow Analysis</div>
        <div style="color:var(--text-secondary); font-size:15px; margin-top:12px; max-width:600px; margin-left:auto; margin-right:auto; line-height:1.7;">
            Analyze Gamma Exposure, Delta Exposure, Max Pain, Vanna & Charm
            across the entire options chain — powered by free CBOE data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    cols = st.columns(3)
    cards = [
        ("GEX", "#00D9FF", "Gamma Exposure reveals how dealer hedging creates support/resistance levels and determines the volatility regime.",
         "GEX > 0 → Stability  ·  GEX < 0 → Volatility"),
        ("Max Pain", "#00FF88", "The strike where most options expire worthless. Acts as a price magnet, especially near OPEX.",
         "0DTE → Strong Pin  ·  30d+ → Weak Signal"),
        ("DEX + Vanna", "#FE53BB", "Delta Exposure shows directional pressure. Vanna shows how IV changes affect hedging flows.",
         "DEX → Direction  ·  Vanna → IV Sensitivity"),
    ]

    for i, (title, color, desc, hint) in enumerate(cards):
        with cols[i]:
            st.markdown(f"""
            <div class="glass-card" style="min-height:220px;">
                <div style="color:{color}; font-weight:900; font-size:18px; margin-bottom:10px; font-family:var(--font-display);">
                    {title}
                </div>
                <div style="color:var(--text-secondary); font-size:13px; line-height:1.7; margin-bottom:12px;">
                    {desc}
                </div>
                <div style="color:var(--text-muted); font-size:11px; font-family:var(--font-mono); padding-top:8px; border-top:1px solid var(--border-subtle);">
                    {hint}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    with st.expander("📖 How to Use This Tool", expanded=False):
        st.markdown("""
        **1. Enter a ticker** — SPY, QQQ, AAPL, TSLA, etc.

        **2. Configure parameters** — Adjust strike range, expiry window, and minimum OI.

        **3. Analyze the dashboard:**
        - **Key Levels Bar** — Spot, Max Pain, Gamma Flip, Max GEX Strike at a glance
        - **Regime Badge** — Instantly see if we're in positive or negative gamma
        - **Max Pain Tab** — Direction and pin probability with IV-enhanced model
        - **Gamma Profile** — Full exposure curve with support/resistance identification
        - **3D Terrain** — Interactive Three.js visualization of the gamma surface
        - **Heatmap** — Strike × Expiry concentration map
        - **DEX** — Directional hedging pressure from delta exposure
        - **IV Surface** — Volatility smile across expirations
        - **Vanna & Charm** — Second-order greek exposures for OPEX dynamics

        **4. Best Practices:**
        - 0DTE/weekly expiries: Max pain is most effective
        - Gamma Flip: Critical level where market behavior changes
        - GEX + DEX combined: Full picture of dealer positioning
        - IV Skew > 0: Put premium = downside hedging demand
        """)

    st.markdown("""
    <div style="text-align:center; margin-top:40px; padding:20px;">
        <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted); letter-spacing:2px;">
            BUILT FOR THE QUANTITATIVE TRADING COMMUNITY
        </div>
        <div style="margin-top:8px;">
            <a href="https://bquantfinance.com" style="color:var(--accent-cyan); text-decoration:none;
                     font-family:var(--font-mono); font-size:12px;">bquantfinance.com</a>
            <span style="color:var(--text-muted); margin:0 8px;">|</span>
            <a href="https://twitter.com/Gsnchez" style="color:var(--accent-magenta); text-decoration:none;
                     font-family:var(--font-mono); font-size:12px;">@Gsnchez</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    render_hero()
    ticker, strike_range, max_exp_days, min_oi, dealer, analyze, compare_input = render_sidebar()

    if analyze:
        progress = st.progress(0)
        status = st.empty()

        try:
            status.markdown(f'<div style="color:var(--text-muted); font-family:var(--font-mono); font-size:12px;">⟳ Fetching {ticker} from CBOE...</div>', unsafe_allow_html=True)
            progress.progress(15)
            raw = fetch_option_data(ticker)

            if raw:
                status.markdown('<div style="color:var(--text-muted); font-family:var(--font-mono); font-size:12px;">⟳ Processing options chain...</div>', unsafe_allow_html=True)
                progress.progress(40)
                spot, odata = parse_option_data(raw)

                if not odata.empty:
                    odata = process_option_data(odata)
                    odata = odata[odata["open_interest"] >= min_oi]

                    if not odata.empty:
                        status.markdown('<div style="color:var(--text-muted); font-family:var(--font-mono); font-size:12px;">⟳ Computing exposures...</div>', unsafe_allow_html=True)
                        progress.progress(70)

                        st.session_state.data_loaded = True
                        st.session_state.ticker_data = {
                            "ticker": ticker, "spot": spot, "data": odata, "dealer": dealer,
                        }

                        progress.progress(100)
                        progress.empty()
                        status.empty()

                        display_results(ticker, spot, odata, strike_range, max_exp_days, dealer)

                        # Multi-ticker compare
                        if compare_input and compare_input.strip():
                            st.markdown("---")
                            st.markdown("### ⚡ Multi-Ticker Comparison")
                            compare_tickers = [t.strip().upper() for t in compare_input.split(",") if t.strip()]
                            compare_data = []
                            for ct in compare_tickers[:3]:
                                craw = fetch_option_data(ct)
                                if craw:
                                    cspot, codata = parse_option_data(craw)
                                    if not codata.empty:
                                        codata = process_option_data(codata)
                                        codata = codata[codata["open_interest"] >= min_oi]
                                        if not codata.empty:
                                            codata = calculate_gex(cspot, codata, dealer)
                                            codata = calculate_dex(cspot, codata, dealer)
                                            cmetrics = compute_all_metrics(codata, cspot)
                                            cmax_pain, _, _ = calculate_max_pain(codata, cspot)
                                            cprofiles = calculate_gamma_profile(codata, cspot, dealer)
                                            compare_data.append({
                                                "ticker": ct, "spot": cspot,
                                                "max_pain": cmax_pain,
                                                "gamma_flip": cprofiles["gamma_flip"],
                                                **cmetrics,
                                            })

                            if compare_data:
                                # Add main ticker
                                main_metrics = compute_all_metrics(
                                    calculate_gex(spot, odata, dealer), spot)
                                main_mp, _, _ = calculate_max_pain(odata, spot)
                                main_prof = calculate_gamma_profile(odata, spot, dealer)

                                all_tickers = [{"ticker": ticker, "spot": spot,
                                               "max_pain": main_mp,
                                               "gamma_flip": main_prof["gamma_flip"],
                                               **main_metrics}] + compare_data

                                comp_df = pd.DataFrame(all_tickers)
                                display_cols = ["ticker", "spot", "max_pain", "gamma_flip",
                                               "total_gex", "total_dex", "iv_mean", "put_call_ratio"]
                                display_cols = [c for c in display_cols if c in comp_df.columns]
                                fmt_df = comp_df[display_cols].copy()
                                for c in ["spot", "max_pain", "gamma_flip"]:
                                    if c in fmt_df.columns:
                                        fmt_df[c] = fmt_df[c].apply(lambda x: f"${x:.2f}")
                                for c in ["total_gex", "total_dex"]:
                                    if c in fmt_df.columns:
                                        fmt_df[c] = fmt_df[c].apply(lambda x: f"${x:.3f}B")
                                if "iv_mean" in fmt_df.columns:
                                    fmt_df["iv_mean"] = fmt_df["iv_mean"].apply(lambda x: f"{x*100:.1f}%")
                                if "put_call_ratio" in fmt_df.columns:
                                    fmt_df["put_call_ratio"] = fmt_df["put_call_ratio"].apply(lambda x: f"{x:.2f}")

                                st.dataframe(fmt_df, use_container_width=True, hide_index=True)
                    else:
                        progress.empty(); status.empty()
                        st.error("No valid data after filtering. Try reducing minimum OI.")
                else:
                    progress.empty(); status.empty()
                    st.error("No options data found.")
            else:
                progress.empty(); status.empty()
                st.error(f"Could not fetch data for {ticker}")

        except Exception as e:
            progress.empty(); status.empty()
            st.error(f"Error: {str(e)}")

    elif st.session_state.data_loaded:
        td = st.session_state.ticker_data
        display_results(td["ticker"], td["spot"], td["data"], strike_range, max_exp_days,
                       td.get("dealer", "standard"))
    else:
        show_landing()


if __name__ == "__main__":
    main()
