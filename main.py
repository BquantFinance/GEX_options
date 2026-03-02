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

/* Hide default streamlit chrome — but keep sidebar toggle */
#MainMenu, footer {visibility: hidden;}
.stDeployButton {display: none;}
header[data-testid="stHeader"] {
    background: transparent !important;
}
/* Ensure sidebar collapse/expand button stays visible */
button[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    visibility: visible !important;
    color: var(--text-secondary) !important;
}

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
        st.error(f"Error de formato: {e}")
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


def calculate_max_pain(data: pd.DataFrame, spot: float, target_expiry=None) -> tuple:
    """
    Max Pain calculated PER EXPIRATION (Barchart methodology).
    
    Key: Max pain only uses options from ONE expiration date, because only
    those options expire/get exercised on that date. Mixing expirations
    produces incorrect results.
    
    Args:
        data: Options DataFrame with 'type', 'strike', 'open_interest', 'expiration'
        spot: Current spot price
        target_expiry: Specific expiration to calculate for. If None, uses nearest.
    
    Returns:
        (max_pain_strike, pain_by_strike, min_pain_value, selected_expiry)
    """
    # If no target expiry, use the nearest expiration
    if target_expiry is None:
        available_expiries = sorted(data["expiration"].unique())
        if not available_expiries:
            return spot, {}, 0, None, {}, {}
        target_expiry = available_expiries[0]  # nearest expiration
    
    # Filter to ONLY this expiration date
    exp_data = data[data["expiration"] == target_expiry]
    if exp_data.empty:
        return spot, {}, 0, target_expiry, {}, {}
    
    strikes = np.sort(exp_data["strike"].unique())
    
    # Aggregate OI per strike per type (in case of duplicates)
    call_oi = exp_data[exp_data["type"] == "C"].groupby("strike")["open_interest"].sum()
    put_oi = exp_data[exp_data["type"] == "P"].groupby("strike")["open_interest"].sum()
    
    pain = {}
    call_pain_curve = {}
    put_pain_curve = {}
    
    for exp_price in strikes:
        # Call holders' pain: calls ITM when strike < exp_price
        # Pain = (exp_price - strike) × OI × 100
        c_pain = 0.0
        for c_strike, c_oi_val in call_oi.items():
            if c_strike < exp_price:
                c_pain += (exp_price - c_strike) * c_oi_val * CONTRACT_SIZE
        
        # Put holders' pain: puts ITM when strike > exp_price
        # Pain = (strike - exp_price) × OI × 100
        p_pain = 0.0
        for p_strike, p_oi_val in put_oi.items():
            if p_strike > exp_price:
                p_pain += (p_strike - exp_price) * p_oi_val * CONTRACT_SIZE
        
        call_pain_curve[exp_price] = c_pain
        put_pain_curve[exp_price] = p_pain
        pain[exp_price] = c_pain + p_pain
    
    if pain:
        mp_strike = min(pain, key=pain.get)
        return mp_strike, pain, pain[mp_strike], target_expiry, call_pain_curve, put_pain_curve
    return spot, {}, 0, target_expiry, {}, {}


def calculate_max_pain_term_structure(data: pd.DataFrame, spot: float, n_expiries: int = 8) -> list:
    """
    Calculate max pain for multiple upcoming expirations.
    Returns a list of (expiry_date, max_pain_strike, days_to_expiry).
    """
    expiries = sorted(data["expiration"].unique())[:n_expiries]
    term_structure = []
    
    for expiry in expiries:
        mp_strike, _, _, _, _, _ = calculate_max_pain(data, spot, target_expiry=expiry)
        dte = max(0, (expiry - pd.Timestamp.now()).days)
        term_structure.append({
            "expiry": expiry,
            "max_pain": mp_strike,
            "dte": dte,
            "distance_pct": (mp_strike - spot) / spot * 100,
        })
    
    return term_structure


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
        "direction": "ALCISTA" if max_pain > spot else "BAJISTA" if max_pain < spot else "NEUTRAL",
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
# NEW: PRICE DATA, WALL DETECTION, EXPECTED MOVE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_price_data(ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
    """Fetch OHLCV price data from Yahoo Finance (stock price only, NOT options)"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"range": f"{days}d", "interval": "1d", "includePrePost": "false"}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()

        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "date": pd.to_datetime(timestamps, unit="s"),
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        }).dropna()
        return df
    except Exception:
        return None


def detect_walls(data: pd.DataFrame, spot: float, strike_range: float = 10, n_walls: int = 5) -> dict:
    """Detect major call/put OI walls — largest OI clusters"""
    lo = spot * (1 - strike_range / 100)
    hi = spot * (1 + strike_range / 100)
    df = data[(data["strike"] >= lo) & (data["strike"] <= hi)]

    calls = df[df["type"] == "C"].groupby("strike").agg(
        oi=("open_interest", "sum"), vol=("volume", "sum"), gex=("GEX", "sum")
    ).sort_values("oi", ascending=False)

    puts = df[df["type"] == "P"].groupby("strike").agg(
        oi=("open_interest", "sum"), vol=("volume", "sum"), gex=("GEX", "sum")
    ).sort_values("oi", ascending=False)

    call_walls = []
    for strike, row in calls.head(n_walls).iterrows():
        dist = (strike - spot) / spot * 100
        call_walls.append({
            "strike": strike, "oi": int(row["oi"]), "vol": int(row["vol"]),
            "gex": row["gex"], "dist_pct": dist,
        })

    put_walls = []
    for strike, row in puts.head(n_walls).iterrows():
        dist = (strike - spot) / spot * 100
        put_walls.append({
            "strike": strike, "oi": int(row["oi"]), "vol": int(row["vol"]),
            "gex": row["gex"], "dist_pct": dist,
        })

    # Biggest single wall overall
    biggest_call = call_walls[0] if call_walls else None
    biggest_put = put_walls[0] if put_walls else None

    return {
        "call_walls": call_walls,
        "put_walls": put_walls,
        "biggest_call": biggest_call,
        "biggest_put": biggest_put,
    }


def calculate_expected_move(data: pd.DataFrame, spot: float) -> dict:
    """Calculate expected move from ATM IV for nearest expiry"""
    nearest_exp = data["expiration"].min()
    dte = max(1, (nearest_exp - pd.Timestamp.now()).days)

    # Get ATM IV — options closest to spot
    atm = data[data["expiration"] == nearest_exp].copy()
    atm["dist"] = (atm["strike"] - spot).abs()
    atm = atm.nsmallest(6, "dist")
    atm_iv = atm["iv"].dropna().mean()

    if pd.isna(atm_iv) or atm_iv <= 0:
        atm_iv = data["iv"].dropna().median()
    if pd.isna(atm_iv) or atm_iv <= 0:
        atm_iv = 0.20  # fallback

    # Expected move = spot × IV × sqrt(DTE/365)
    sqrt_t = math.sqrt(dte / 365)
    em_1sigma = spot * atm_iv * sqrt_t
    em_2sigma = em_1sigma * 2

    return {
        "atm_iv": atm_iv,
        "dte": dte,
        "expiry": nearest_exp,
        "em_1sigma": em_1sigma,
        "em_2sigma": em_2sigma,
        "em_1sigma_pct": (em_1sigma / spot) * 100,
        "em_2sigma_pct": (em_2sigma / spot) * 100,
        "upper_1s": spot + em_1sigma,
        "lower_1s": spot - em_1sigma,
        "upper_2s": spot + em_2sigma,
        "lower_2s": spot - em_2sigma,
    }


def compute_0dte_metrics(data: pd.DataFrame, spot: float, dealer: str = "standard") -> Optional[dict]:
    """Compute separate GEX/DEX metrics for 0DTE options only"""
    today = pd.Timestamp.now().normalize()
    dte0 = data[data["expiration"].dt.normalize() == today]

    if dte0.empty:
        return None

    dte0_gex = calculate_gex(spot, dte0, dealer)
    dte0_dex = calculate_dex(spot, dte0, dealer)

    return {
        "count": len(dte0),
        "gex": dte0_gex["GEX"].sum() / 1e9,
        "dex": dte0_dex["DEX"].sum() / 1e9 if "DEX" in dte0_dex.columns else 0,
        "call_oi": int(dte0[dte0["type"] == "C"]["open_interest"].sum()),
        "put_oi": int(dte0[dte0["type"] == "P"]["open_interest"].sum()),
        "call_vol": int(dte0[dte0["type"] == "C"]["volume"].sum()),
        "put_vol": int(dte0[dte0["type"] == "P"]["volume"].sum()),
        "pct_of_total_gex": 0,  # filled below
        "top_strike": float(dte0_gex.groupby("strike")["GEX"].sum().abs().idxmax()) if not dte0_gex.empty else spot,
    }


def chart_price_with_levels(price_df: pd.DataFrame, spot: float, max_pain: float,
                            gamma_flip: float, metrics: dict, walls: dict,
                            expected_move: dict, ticker: str) -> go.Figure:
    """Candlestick chart with GEX levels + expected move cone overlay"""
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=price_df["date"], open=price_df["open"], high=price_df["high"],
        low=price_df["low"], close=price_df["close"],
        increasing_line_color="#00FF88", decreasing_line_color="#FF4466",
        increasing_fillcolor="rgba(0,255,136,0.25)", decreasing_fillcolor="rgba(255,68,102,0.25)",
        name="Precio",
    ))

    # Volume bars on secondary y
    vol_colors = ["rgba(0,255,136,0.2)" if c >= o else "rgba(255,68,102,0.2)"
                  for c, o in zip(price_df["close"], price_df["open"])]
    fig.add_trace(go.Bar(
        x=price_df["date"], y=price_df["volume"],
        marker_color=vol_colors, name="Volume",
        yaxis="y2", opacity=0.3, showlegend=False,
    ))

    last_date = price_df["date"].iloc[-1]
    first_date = price_df["date"].iloc[0]

    # ── Y-axis range: based on price action + EM bands, NOT outlier levels ──
    em = expected_move
    price_lo = min(price_df["low"].min(), em["lower_2s"])
    price_hi = max(price_df["high"].max(), em["upper_2s"])
    pad = (price_hi - price_lo) * 0.08
    y_lo = price_lo - pad
    y_hi = price_hi + pad

    def in_range(price):
        return y_lo <= price <= y_hi

    # ── Level Lines (only if within visible range) ──
    def add_level(price, color, name, dash="dash", width=1.5):
        if in_range(price):
            fig.add_hline(y=price, line_dash=dash, line_color=color, line_width=width, opacity=0.7)
            fig.add_annotation(
                x=last_date, y=price, text=f" {name}: ${price:.2f}",
                showarrow=False, xanchor="left", font=dict(size=10, color=color),
                bgcolor="rgba(6,9,15,0.8)", bordercolor=color, borderwidth=1, borderpad=3,
            )
        else:
            # Off-screen: show at edge with arrow
            edge_y = y_hi - pad * 0.3 if price > y_hi else y_lo + pad * 0.3
            arrow = "↑" if price > y_hi else "↓"
            dist_pct = (price - spot) / spot * 100
            fig.add_annotation(
                x=last_date, y=edge_y,
                text=f" {arrow} {name}: ${price:.2f} ({dist_pct:+.1f}%)",
                showarrow=False, xanchor="left", font=dict(size=10, color=color),
                bgcolor="rgba(6,9,15,0.85)", bordercolor=color, borderwidth=1, borderpad=3,
            )

    add_level(max_pain, "#00FF88", "Max Pain", "dash", 2)
    add_level(gamma_flip, "#FE53BB", "γ Flip", "dash", 2)
    add_level(metrics["max_gex_strike"], "#00D9FF", "Max GEX", "dot", 1.5)

    # ── Expected Move Cone ──
    for sigma, opacity, label in [(1, 0.12, "±1σ"), (2, 0.06, "±2σ")]:
        upper = em[f"upper_{sigma}s"]
        lower = em[f"lower_{sigma}s"]
        fig.add_hrect(y0=lower, y1=upper,
                      fillcolor=f"rgba(0,217,255,{opacity})",
                      line_width=0, layer="below")
        fig.add_annotation(
            x=last_date, y=upper,
            text=f" {label} ${upper:.2f}", showarrow=False,
            xanchor="left", font=dict(size=9, color="rgba(0,217,255,0.53)"),
        )
        fig.add_annotation(
            x=last_date, y=lower,
            text=f" {label} ${lower:.2f}", showarrow=False,
            xanchor="left", font=dict(size=9, color="rgba(0,217,255,0.53)"),
        )

    # ── Call/Put Walls (only if visible) ──
    if walls["biggest_call"] and in_range(walls["biggest_call"]["strike"]):
        cw = walls["biggest_call"]["strike"]
        fig.add_hline(y=cw, line_dash="dot", line_color="rgba(0,255,136,0.53)", line_width=1)
        fig.add_annotation(
            x=first_date, y=cw, text=f"CALL WALL ${cw:.0f} ({walls['biggest_call']['oi']:,} OI)",
            showarrow=False, xanchor="left",
            font=dict(size=9, color="#00FF88"), bgcolor="rgba(0,255,136,0.08)",
        )

    if walls["biggest_put"] and in_range(walls["biggest_put"]["strike"]):
        pw = walls["biggest_put"]["strike"]
        fig.add_hline(y=pw, line_dash="dot", line_color="rgba(255,68,102,0.53)", line_width=1)
        fig.add_annotation(
            x=first_date, y=pw, text=f"PUT WALL ${pw:.0f} ({walls['biggest_put']['oi']:,} OI)",
            showarrow=False, xanchor="left",
            font=dict(size=9, color="#FF4466"), bgcolor="rgba(255,68,102,0.08)",
        )

    fig.update_layout(**_base_layout(
        title=dict(text=f"{ticker} — Price + GEX Levels", font=dict(size=18)),
        height=520, showlegend=False,
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
        yaxis=dict(title="Price ($)", side="right", range=[y_lo, y_hi]),
        yaxis2=dict(overlaying="y", side="left", showgrid=False, showticklabels=False,
                    range=[0, price_df["volume"].max() * 4]),
    ))
    return fig


def chart_levels_map(spot: float, max_pain: float, gamma_flip: float,
                     metrics: dict, walls: dict, expected_move: dict,
                     ticker: str) -> go.Figure:
    """Visual levels map — all key strikes on a vertical price axis (CBOE data only)"""
    fig = go.Figure()

    em = expected_move
    lo = em["lower_2s"] * 0.995
    hi = em["upper_2s"] * 1.005

    # ── Expected Move bands ──
    fig.add_shape(type="rect", x0=0, x1=1, y0=em["lower_2s"], y1=em["upper_2s"],
                  fillcolor="rgba(0,217,255,0.04)", line_width=0, xref="paper")
    fig.add_shape(type="rect", x0=0, x1=1, y0=em["lower_1s"], y1=em["upper_1s"],
                  fillcolor="rgba(0,217,255,0.08)", line_width=0, xref="paper")

    # ── Build scatter of all levels ──
    levels = []

    # Spot
    levels.append(dict(y=spot, x=0.5, text=f"SPOT ${spot:.2f}", color="#FFD700", size=16, symbol="diamond"))
    # Max Pain
    levels.append(dict(y=max_pain, x=0.35, text=f"Max Pain ${max_pain:.2f}", color="#00FF88", size=13, symbol="triangle-up"))
    # Gamma Flip
    if gamma_flip != spot:
        levels.append(dict(y=gamma_flip, x=0.65, text=f"γ Flip ${gamma_flip:.2f}", color="#FE53BB", size=13, symbol="triangle-down"))
    # Max GEX
    levels.append(dict(y=metrics["max_gex_strike"], x=0.5, text=f"Max GEX ${metrics['max_gex_strike']:.2f}", color="#00D9FF", size=12, symbol="star"))

    # EM bounds
    for sigma, alpha in [(1, 0.7), (2, 0.4)]:
        u, l = em[f"upper_{sigma}s"], em[f"lower_{sigma}s"]
        fig.add_hline(y=u, line_dash="dot", line_color=f"rgba(0,217,255,{alpha})", line_width=1)
        fig.add_hline(y=l, line_dash="dot", line_color=f"rgba(0,217,255,{alpha})", line_width=1)
        fig.add_annotation(x=1, y=u, text=f" +{sigma}σ ${u:.2f}", xref="paper",
                          showarrow=False, xanchor="left", font=dict(size=9, color=f"rgba(0,217,255,{alpha})"))
        fig.add_annotation(x=1, y=l, text=f" -{sigma}σ ${l:.2f}", xref="paper",
                          showarrow=False, xanchor="left", font=dict(size=9, color=f"rgba(0,217,255,{alpha})"))

    # Call walls
    for i, w in enumerate(walls["call_walls"][:3]):
        oi_scale = w["oi"] / max(walls["call_walls"][0]["oi"], 1)
        levels.append(dict(y=w["strike"], x=0.75 + i * 0.05,
                          text=f"C Wall ${w['strike']:.0f} ({w['oi']:,})",
                          color="#00FF88", size=8 + oi_scale * 8, symbol="triangle-right"))

    # Put walls
    for i, w in enumerate(walls["put_walls"][:3]):
        oi_scale = w["oi"] / max(walls["put_walls"][0]["oi"], 1)
        levels.append(dict(y=w["strike"], x=0.25 - i * 0.05,
                          text=f"P Wall ${w['strike']:.0f} ({w['oi']:,})",
                          color="#FF4466", size=8 + oi_scale * 8, symbol="triangle-left"))

    # Plot all levels
    for lv in levels:
        fig.add_trace(go.Scatter(
            x=[lv["x"]], y=[lv["y"]], mode="markers+text",
            marker=dict(color=lv["color"], size=lv["size"], symbol=lv["symbol"],
                       line=dict(width=1, color="rgba(255,255,255,0.3)")),
            text=[lv["text"]], textposition="middle right" if lv["x"] < 0.5 else "middle left",
            textfont=dict(size=10, color=lv["color"]),
            hovertemplate=f"{lv['text']}<extra></extra>",
            showlegend=False,
        ))

    # Spot horizontal line
    fig.add_hline(y=spot, line_dash="solid", line_color="rgba(255,215,0,0.4)", line_width=2)
    fig.add_hline(y=max_pain, line_dash="dash", line_color="rgba(0,255,136,0.27)", line_width=1.5)
    if gamma_flip != spot:
        fig.add_hline(y=gamma_flip, line_dash="dash", line_color="rgba(254,83,187,0.27)", line_width=1.5)

    fig.update_layout(**_base_layout(
        title=dict(text=f"{ticker} — Key Levels Map", font=dict(size=18)),
        height=520, showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.1, 1.1]),
        yaxis=dict(title="Price ($)", side="right", range=[lo, hi]),
    ))
    return fig


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
        mode="lines", name="Gamma Neta",
        line=dict(color=COLORS["cyan"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,217,255,0.08)",
        hovertemplate="Strike: $%{x:.1f}<br>GEX: %{y:.3f}B<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=profiles["strikes"], y=profiles["call_gamma"],
        mode="lines", name="Gamma Calls",
        line=dict(color=COLORS["green"], width=1.5, dash="dot"),
        opacity=0.6,
        hovertemplate="Strike: $%{x:.1f}<br>Call: %{y:.3f}B<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=profiles["strikes"], y=profiles["put_gamma"],
        mode="lines", name="Gamma Puts",
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
        title=dict(text=f"Perfil de Exposición Gamma — {ticker}", font=dict(size=18)),
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


def chart_max_pain(pain_by_strike: dict, max_pain: float, spot: float,
                   call_pain: dict = None, put_pain: dict = None, expiry_label: str = "") -> go.Figure:
    """Max pain curve with separate call/put pain (Barchart style)"""
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

    # Separate call and put pain curves (Barchart style)
    if call_pain and put_pain:
        call_vals = [call_pain.get(s, 0) / 1e9 for s in strikes]
        put_vals = [put_pain.get(s, 0) / 1e9 for s in strikes]

        fig.add_trace(go.Scatter(
            x=strikes, y=call_vals, mode="lines", fill="tozeroy",
            line=dict(color=COLORS["cyan"], width=1.5),
            fillcolor="rgba(0,217,255,0.12)",
            name="Dolor Calls",
            hovertemplate="Strike: $%{x:.1f}<br>Call Pain: $%{y:.2f}B<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=strikes, y=put_vals, mode="lines", fill="tozeroy",
            line=dict(color=COLORS["magenta"], width=1.5),
            fillcolor="rgba(254,83,187,0.12)",
            name="Dolor Puts",
            hovertemplate="Strike: $%{x:.1f}<br>Put Pain: $%{y:.2f}B<extra></extra>",
        ))

    # Total pain curve on top
    fig.add_trace(go.Scatter(
        x=strikes, y=values, mode="lines",
        line=dict(color=COLORS["red"], width=2.5),
        name="Dolor Total",
        hovertemplate="Strike: $%{x:.1f}<br>Total Pain: $%{y:.2f}B<extra></extra>",
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
    title_suffix = f" — {expiry_label}" if expiry_label else ""

    fig.update_layout(**_base_layout(
        title=dict(text=f"Max Pain ${max_pain:.2f}  |  Spot ${spot:.2f}  |  {arrow} {abs(dist):.2f}%{title_suffix}",
                   font=dict(size=18)),
        xaxis_title="Strike ($)", yaxis_title="Total Pain ($Bn)", height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    ))
    return fig


def chart_max_pain_term_structure(term_structure: list, spot: float) -> go.Figure:
    """Max pain across multiple expirations — term structure"""
    if not term_structure:
        return go.Figure()

    dates = [ts["expiry"] for ts in term_structure]
    mp_vals = [ts["max_pain"] for ts in term_structure]
    dists = [ts["distance_pct"] for ts in term_structure]
    colors = [COLORS["green"] if d >= 0 else COLORS["red"] for d in dists]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=mp_vals, mode="lines+markers",
        line=dict(color=COLORS["cyan"], width=2),
        marker=dict(size=10, color=colors, line=dict(width=2, color="#fff")),
        name="Max Pain",
        hovertemplate="%{x|%b %d}<br>Max Pain: $%{y:.2f}<extra></extra>",
    ))

    # Spot reference line
    fig.add_hline(y=spot, line_dash="dash", line_color=COLORS["gold"], line_width=1.5,
                  annotation_text=f"Spot ${spot:.2f}",
                  annotation_font=dict(size=11, color=COLORS["gold"]))

    fig.update_layout(**_base_layout(
        title=dict(text="Max Pain Term Structure", font=dict(size=18)),
        xaxis_title="Expiration", yaxis_title="Max Pain ($)",
        height=350, showlegend=False,
        xaxis=dict(tickformat="%b %d"),
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


def render_3d_iv_surface(data: pd.DataFrame, spot: float, strike_range: float, metrics: dict):
    """Render 3D IV surface — Three.js, direct XYZ geometry, cinematic"""
    if "iv" not in data.columns or data["iv"].isna().all():
        st.warning("Datos de IV no disponibles para la superficie 3D.")
        return

    lo, hi = spot * (1 - strike_range / 100), spot * (1 + strike_range / 100)
    df = data[(data["strike"] >= lo) & (data["strike"] <= hi)].copy()
    df["dte"] = (df["expiration"] - pd.Timestamp.now()).dt.days
    df = df[(df["dte"] >= 0) & (df["iv"].notna()) & (df["iv"] > 0)]

    if df.empty:
        st.warning("Datos de IV insuficientes para la superficie 3D.")
        return

    rows = []
    for exp in df["expiration"].unique():
        edf = df[df["expiration"] == exp]
        dte = int(edf["dte"].iloc[0])
        puts = edf[(edf["type"] == "P") & (edf["strike"] <= spot)].sort_values("strike")
        calls = edf[(edf["type"] == "C") & (edf["strike"] > spot)].sort_values("strike")
        merged = pd.concat([puts, calls])
        for _, r in merged.iterrows():
            rows.append({"strike": float(r["strike"]), "dte": dte, "iv": float(r["iv"]) * 100})

    if not rows:
        st.warning("Datos de IV insuficientes.")
        return

    iv_df = pd.DataFrame(rows)
    n_strike_bins = min(40, iv_df["strike"].nunique())
    n_dte_bins = min(25, iv_df["dte"].nunique())
    strike_bins = np.linspace(iv_df["strike"].min(), iv_df["strike"].max(), n_strike_bins + 1)
    dte_max = min(iv_df["dte"].max(), 120)
    dte_bins = np.linspace(0, dte_max, n_dte_bins + 1)
    iv_df["s_bin"] = pd.cut(iv_df["strike"], bins=strike_bins, labels=strike_bins[:-1], include_lowest=True).astype(float)
    iv_df["d_bin"] = pd.cut(iv_df["dte"], bins=dte_bins, labels=dte_bins[:-1], include_lowest=True).astype(float)
    pivot = iv_df.pivot_table(values="iv", index="s_bin", columns="d_bin", aggfunc="mean")
    pivot = pivot.loc[pivot.index.notna(), pivot.columns.notna()]
    pivot = pivot.interpolate(axis=0, limit=3).interpolate(axis=1, limit=3).bfill().ffill().fillna(0)

    strikes_list = [float(s) for s in pivot.index]
    dtes_list = [float(d) for d in pivot.columns]
    nS = len(strikes_list)
    nD = len(dtes_list)

    grid = []
    for si, sv in enumerate(strikes_list):
        for di, dv in enumerate(dtes_list):
            grid.append({"s": sv, "d": dv, "iv": round(float(pivot.iloc[si, di]), 2)})

    n_stk = min(6, nS)
    stk_idx = [int(i * (nS - 1) / max(n_stk - 1, 1)) for i in range(n_stk)]
    stk_labels = [{"i": i, "v": round(strikes_list[i], 1)} for i in stk_idx]
    n_dt = min(5, nD)
    dt_idx = [int(i * (nD - 1) / max(n_dt - 1, 1)) for i in range(n_dt)]
    dt_labels = [{"i": i, "v": int(dtes_list[i])} for i in dt_idx]

    iv_min_v = float(pivot[pivot > 0].min().min()) if (pivot > 0).any().any() else 10
    iv_max_v = float(pivot.max().max())

    json_data = json.dumps({
        "grid": grid, "spot": spot, "nS": nS, "nD": nD,
        "strikes": strikes_list, "dtes": dtes_list,
        "stkLabels": stk_labels, "dtLabels": dt_labels,
        "ivMin": round(iv_min_v, 1), "ivMax": round(iv_max_v, 1),
        "ivMean": round(metrics["iv_mean"] * 100, 1),
        "ivSkew": round(metrics["iv_skew"] * 100, 1),
    })

    html = f"""
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{overflow:hidden;background:#06090F;font-family:'JetBrains Mono',monospace}}
    canvas{{display:block;filter:contrast(1.04) saturate(1.12)}}

    .vig{{position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:5;
      background:radial-gradient(ellipse at 50% 50%,transparent 55%,rgba(6,9,15,0.55) 100%)}}

    .hud{{position:absolute;top:14px;left:16px;z-index:10;pointer-events:none}}
    .hud-c{{
      background:rgba(6,9,15,0.82);backdrop-filter:blur(20px);
      border:1px solid rgba(0,217,255,0.1);border-radius:14px;
      padding:14px 20px;color:#7A8BA8;font-size:11px;min-width:180px;
      box-shadow:0 8px 32px rgba(0,0,0,0.4),0 0 60px rgba(0,217,255,0.03),inset 0 1px 0 rgba(255,255,255,0.03)}}
    .hud-t{{font-size:18px;font-weight:700;letter-spacing:-0.5px;margin-bottom:8px;
      background:linear-gradient(135deg,#00D9FF,#FE53BB 60%,#FFD700);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
    .hud-r{{display:flex;justify-content:space-between;margin-top:5px}}
    .hud-l{{color:#4A5568;font-size:9px;text-transform:uppercase;letter-spacing:1.2px}}
    .hud-v{{color:#E8ECF4;font-weight:600;font-size:12px}}

    .ctr{{position:absolute;top:14px;right:16px;display:flex;gap:6px;z-index:10}}
    .ctr button{{
      background:rgba(11,17,32,0.8);backdrop-filter:blur(12px);
      border:1px solid rgba(0,217,255,0.12);color:#7A8BA8;
      padding:7px 16px;border-radius:20px;font-size:10px;cursor:pointer;
      font-family:inherit;transition:all .25s;letter-spacing:0.3px}}
    .ctr button:hover{{border-color:rgba(0,217,255,0.35);color:#E8ECF4;
      box-shadow:0 0 15px rgba(0,217,255,0.08)}}
    .ctr button.on{{background:rgba(0,217,255,0.1);color:#00D9FF;border-color:rgba(0,217,255,0.25)}}

    #tip{{position:absolute;display:none;z-index:20;pointer-events:none;
      background:rgba(6,9,15,0.92);backdrop-filter:blur(16px);
      border:1px solid rgba(0,217,255,0.2);border-radius:10px;
      padding:10px 14px;font-size:11px;color:#E8ECF4;
      box-shadow:0 8px 30px rgba(0,0,0,0.5),0 0 40px rgba(0,217,255,0.05)}}
    #tip b{{color:#00D9FF}}
    .tip-iv{{font-size:18px;font-weight:700;margin:2px 0 4px;
      background:linear-gradient(90deg,#00D9FF,#FFD700);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent}}

    .leg{{position:absolute;bottom:14px;left:16px;z-index:10;pointer-events:none;
      background:rgba(6,9,15,0.8);backdrop-filter:blur(12px);
      border:1px solid rgba(0,217,255,0.06);border-radius:10px;
      padding:10px 16px;font-size:9px;color:#4A5568}}
    .leg .bar{{width:150px;height:6px;border-radius:3px;margin:5px 0;
      background:linear-gradient(90deg,#0a1628,#003366,#0088cc,#00D9FF,#88CC44,#FFD700,#FF6633,#FF2200)}}

    .axi{{position:absolute;bottom:14px;right:16px;z-index:10;pointer-events:none;
      background:rgba(6,9,15,0.8);backdrop-filter:blur(12px);
      border:1px solid rgba(0,217,255,0.06);border-radius:10px;
      padding:8px 14px;font-size:9px;color:#4A5568;line-height:2}}
    </style></head><body>
    <div class="vig"></div>

    <div class="hud"><div class="hud-c">
      <div class="hud-t">Vol Surface</div>
      <div class="hud-r"><span class="hud-l">Spot</span><span class="hud-v" style="color:#FFD700">${spot:.2f}</span></div>
      <div class="hud-r"><span class="hud-l">IV Media</span><span class="hud-v">{metrics['iv_mean']*100:.1f}%</span></div>
      <div class="hud-r"><span class="hud-l">Sesgo IV</span><span class="hud-v" style="color:{'#FF4466' if metrics['iv_skew']>0 else '#00FF88'}">{metrics['iv_skew']*100:+.1f}%</span></div>
      <div class="hud-r"><span class="hud-l">Rango</span><span class="hud-v" style="font-size:10px">{iv_min_v:.0f}% — {iv_max_v:.0f}%</span></div>
    </div></div>

    <div class="ctr">
      <button id="bRot" class="on" onclick="autoRot=!autoRot;this.classList.toggle('on')">⟳ Rotate</button>
      <button onclick="th=0.6;ph=0.7;rad=26">↺ Reset</button>
      <button id="bW" class="on" onclick="wire.visible=!wire.visible;this.classList.toggle('on')">◻ Wire</button>
      <button id="bR" class="on" onclick="refs.visible=!refs.visible;this.classList.toggle('on')">▧ Labels</button>
    </div>
    <div id="tip"></div>

    <div class="leg">
      <div style="color:#7A8BA8;font-weight:600;letter-spacing:1px">IV SCALE</div>
      <div class="bar"></div>
      <div style="display:flex;justify-content:space-between"><span>{iv_min_v:.0f}%</span><span>{iv_max_v:.0f}%</span></div>
      <div style="margin-top:6px;line-height:1.8"><span style="color:#FFD700">◆ ATM / Spot</span><br>
        <span style="color:rgba(0,217,255,0.5)">─ IV Planes</span></div>
    </div>

    <div class="axi">
      <span style="color:#00D9FF;font-weight:600">X →</span> Strike ($)<br>
      <span style="color:#FFD700;font-weight:600">Y ↑</span> IV (%)<br>
      <span style="color:#FE53BB;font-weight:600">Z →</span> DTE
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    const D={json_data};
    const nS=D.nS,nD=D.nD,W=20,Dep=14,H=10;
    const lk={{}};D.grid.forEach(p=>{{lk[p.s+','+p.d]=p.iv}});
    const ivMin=D.ivMin,ivMax=D.ivMax,ivR=ivMax-ivMin||1;
    let autoRot=true,th=0.6,ph=0.7,rad=26,md=false,lx=0,ly=0;

    /* ═══ Scene ═══ */
    const sc=new THREE.Scene();
    sc.fog=new THREE.FogExp2(0x06090F,0.003);
    const cam=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,0.1,300);
    const R=new THREE.WebGLRenderer({{antialias:true,alpha:true,powerPreference:'high-performance'}});
    R.setSize(innerWidth,innerHeight);
    R.setPixelRatio(Math.min(devicePixelRatio,2));
    R.setClearColor(0x06090F);
    R.shadowMap.enabled=true;
    R.shadowMap.type=THREE.PCFSoftShadowMap;
    document.body.appendChild(R.domElement);

    /* ═══ Lights ═══ */
    sc.add(new THREE.AmbientLight(0x1a2840,0.6));
    const dL=new THREE.DirectionalLight(0xffffff,0.8);
    dL.position.set(10,25,10);dL.castShadow=true;
    dL.shadow.mapSize.set(1024,1024);sc.add(dL);
    const pA=new THREE.PointLight(0x00D9FF,1.5,60);pA.position.set(-15,12,-10);sc.add(pA);
    const pB=new THREE.PointLight(0xFE53BB,1.0,60);pB.position.set(15,12,12);sc.add(pB);
    const pC=new THREE.PointLight(0xFFD700,0.6,40);pC.position.set(0,18,0);sc.add(pC);
    const pD=new THREE.PointLight(0x00D9FF,0.3,30);pD.position.set(0,-2,0);sc.add(pD);

    /* ═══ Floor ═══ */
    const flG=new THREE.PlaneGeometry(50,50);
    const flM=new THREE.MeshPhongMaterial({{color:0x080c14,specular:0x111828,shininess:80,transparent:true,opacity:0.85}});
    const fl=new THREE.Mesh(flG,flM);
    fl.rotation.x=-Math.PI/2;fl.position.y=-0.1;fl.receiveShadow=true;sc.add(fl);
    sc.add(new THREE.GridHelper(40,40,0x0d1525,0x0a0f1c));

    /* ═══ Color function ═══ */
    function ivCol(t){{
      let r,g,b;
      if(t<0.2){{const s=t/0.2;r=0.04;g=0.06+s*0.15;b=0.15+s*0.35}}
      else if(t<0.4){{const s=(t-0.2)/0.2;r=0;g=0.21+s*0.5;b=0.5+s*0.5}}
      else if(t<0.55){{const s=(t-0.4)/0.15;r=s*0.2;g=0.71+s*0.17;b=1.0-s*0.15}}
      else if(t<0.7){{const s=(t-0.55)/0.15;r=0.2+s*0.8;g=0.88-s*0.05;b=0.85-s*0.65}}
      else if(t<0.85){{const s=(t-0.7)/0.15;r=1.0;g=0.83-s*0.45;b=0.2-s*0.1}}
      else{{const s=(t-0.85)/0.15;r=1.0-s*0.15;g=0.38-s*0.25;b=0.1-s*0.05}}
      return[r,g,b]
    }}

    /* ═══ Build surface (direct XYZ, no rotation) ═══ */
    const vs=[],cs=[],ix=[];
    for(let j=0;j<nD;j++){{
      for(let i=0;i<nS;i++){{
        const x=-W/2+(i/(nS-1||1))*W;
        const z=-Dep/2+(j/(nD-1||1))*Dep;
        const iv=lk[D.strikes[i]+','+D.dtes[j]]||0;
        const t=Math.max(0,Math.min(1,(iv-ivMin)/ivR));
        vs.push(x,t*H,z);
        const[cr,cg,cb]=ivCol(t);
        cs.push(cr,cg,cb);
      }}
    }}
    for(let j=0;j<nD-1;j++)
      for(let i=0;i<nS-1;i++){{
        const a=j*nS+i,b=a+1,c=(j+1)*nS+i,d=c+1;
        ix.push(a,b,d, a,d,c);
      }}

    const sG=new THREE.BufferGeometry();
    sG.setAttribute('position',new THREE.Float32BufferAttribute(vs,3));
    sG.setAttribute('color',new THREE.Float32BufferAttribute(cs,3));
    sG.setIndex(ix);
    sG.computeVertexNormals();

    const sM=new THREE.MeshPhongMaterial({{
      vertexColors:true,side:THREE.DoubleSide,
      shininess:120,transparent:true,opacity:0.90,
      specular:0x334466
    }});
    const surf=new THREE.Mesh(sG,sM);
    surf.castShadow=true;surf.receiveShadow=true;sc.add(surf);

    /* Edge glow */
    const glM=new THREE.MeshBasicMaterial({{vertexColors:true,transparent:true,opacity:0.05,side:THREE.BackSide,depthWrite:false}});
    const glow=new THREE.Mesh(sG.clone(),glM);
    glow.scale.set(1.01,1.05,1.01);sc.add(glow);

    /* Wireframe */
    const wM=new THREE.MeshBasicMaterial({{color:0xffffff,wireframe:true,transparent:true,opacity:0.035}});
    const wire=new THREE.Mesh(sG.clone(),wM);
    sc.add(wire);

    /* ═══ ATM line + spot ═══ */
    const sI=D.strikes.reduce((b,s,i)=>Math.abs(s-D.spot)<Math.abs(D.strikes[b]-D.spot)?i:b,0);
    const sX=-W/2+(sI/(nS-1||1))*W;

    // ATM curve on surface
    const ap=[];
    for(let j=0;j<nD;j++){{
      const iv=lk[D.strikes[sI]+','+D.dtes[j]]||0;
      const t=Math.max(0,Math.min(1,(iv-ivMin)/ivR));
      ap.push(new THREE.Vector3(sX,t*H+0.06,-Dep/2+(j/(nD-1||1))*Dep));
    }}
    sc.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(ap),
      new THREE.LineBasicMaterial({{color:0xFFD700,transparent:true,opacity:0.75}})
    ));

    // Beam
    const bmG=new THREE.CylinderGeometry(0.01,0.01,H+4,8);
    const bmM=new THREE.MeshBasicMaterial({{color:0xFFD700,transparent:true,opacity:0.06}});
    const bm=new THREE.Mesh(bmG,bmM);bm.position.set(sX,(H+4)/2,0);sc.add(bm);

    // Diamond
    const diG=new THREE.OctahedronGeometry(0.25);
    const diM=new THREE.MeshPhongMaterial({{color:0xFFD700,emissive:0xFFD700,emissiveIntensity:0.4,shininess:200}});
    const dia=new THREE.Mesh(diG,diM);dia.position.set(sX,H+1.5,0);sc.add(dia);
    const dgM=new THREE.MeshBasicMaterial({{color:0xFFD700,transparent:true,opacity:0.06}});
    const dgl=new THREE.Mesh(new THREE.SphereGeometry(0.6,16,16),dgM);
    dgl.position.copy(dia.position);sc.add(dgl);

    /* ═══ References group ═══ */
    const refs=new THREE.Group();

    function mkT(text,color,sz){{
      const c=document.createElement('canvas');c.width=256;c.height=64;
      const x=c.getContext('2d');
      x.font=(sz||30)+'px JetBrains Mono,monospace';
      x.fillStyle=color||'#7A8BA8';
      x.textAlign='center';x.textBaseline='middle';
      x.fillText(text,128,32);
      const t=new THREE.CanvasTexture(c);t.minFilter=THREE.LinearFilter;
      const sp=new THREE.Sprite(new THREE.SpriteMaterial({{map:t,transparent:true,opacity:0.7,depthWrite:false}}));
      sp.scale.set(2.2,0.55,1);return sp
    }}

    // Strike labels (X front)
    D.stkLabels.forEach(s=>{{
      const x=-W/2+(s.i/(nS-1||1))*W;
      const lb=mkT('$'+s.v.toFixed(0),'#00D9FF',26);
      lb.position.set(x,-0.5,Dep/2+1.2);refs.add(lb);
      const tg=new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x,0,Dep/2+0.2),new THREE.Vector3(x,0,Dep/2+0.6)]);
      refs.add(new THREE.Line(tg,new THREE.LineBasicMaterial({{color:0x00D9FF,transparent:true,opacity:0.2}})));
    }});

    // DTE labels (Z left)
    D.dtLabels.forEach(d=>{{
      const z=-Dep/2+(d.i/(nD-1||1))*Dep;
      const lb=mkT(d.v+'d','#FE53BB',26);
      lb.position.set(-W/2-1.6,-0.5,z);refs.add(lb);
      const tg=new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-W/2-0.2,0,z),new THREE.Vector3(-W/2-0.6,0,z)]);
      refs.add(new THREE.Line(tg,new THREE.LineBasicMaterial({{color:0xFE53BB,transparent:true,opacity:0.2}})));
    }});

    // IV% labels (Y left) + ref planes
    const ivSt=[];
    for(let p=Math.ceil(ivMin/5)*5;p<=ivMax;p+=Math.max(5,Math.round((ivMax-ivMin)/4/5)*5))ivSt.push(p);
    if(ivSt.length>5){{const st=Math.ceil(ivSt.length/4);ivSt.splice(0,ivSt.length,...ivSt.filter((_,i)=>i%st===0))}}

    ivSt.forEach(iv=>{{
      const y=((iv-ivMin)/ivR)*H;
      const lb=mkT(iv+'%','#00D9FF',24);
      lb.position.set(-W/2-2.3,y,-Dep/2);refs.add(lb);
      // Glass ref plane
      const pg=new THREE.PlaneGeometry(W,Dep);
      const pm=new THREE.MeshBasicMaterial({{color:0x00D9FF,transparent:true,opacity:0.012,side:THREE.DoubleSide,depthWrite:false}});
      const p=new THREE.Mesh(pg,pm);p.rotation.x=-Math.PI/2;p.position.set(0,y,0);refs.add(p);
      // Edge border
      const pts=[new THREE.Vector3(-W/2,y,-Dep/2),new THREE.Vector3(W/2,y,-Dep/2),
        new THREE.Vector3(W/2,y,Dep/2),new THREE.Vector3(-W/2,y,Dep/2),new THREE.Vector3(-W/2,y,-Dep/2)];
      refs.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({{color:0x00D9FF,transparent:true,opacity:0.05}})));
    }});

    // Axis titles
    const t1=mkT('STRIKE →','#00D9FF',22);t1.position.set(0,-0.5,Dep/2+2.5);refs.add(t1);
    const t2=mkT('← DTE','#FE53BB',22);t2.position.set(-W/2-1.6,-0.5,-Dep/2-1.2);refs.add(t2);
    const t3=mkT('IV% ↑','#FFD700',22);t3.position.set(-W/2-2.3,H*0.5,Dep/2);refs.add(t3);

    // Spot label above diamond
    const spL=mkT('SPOT $'+D.spot.toFixed(1),'#FFD700',28);
    spL.position.set(sX,H+2.8,0);refs.add(spL);

    // Bounding box edges for structure
    const bxPts=[
      [-W/2,0,-Dep/2],[-W/2,0,Dep/2],[-W/2,H,-Dep/2],[-W/2,H,Dep/2],
      [W/2,0,-Dep/2],[W/2,0,Dep/2]];
    // Left wall vertical edges
    [[0,2],[1,3]].forEach(([a,b])=>{{
      const pts=[new THREE.Vector3(...bxPts[a]),new THREE.Vector3(...bxPts[b])];
      refs.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({{color:0x0d1525,transparent:true,opacity:0.3}})));
    }});
    // Bottom edges
    [[0,1],[0,4],[1,5],[4,5]].forEach(([a,b])=>{{
      const pts=[new THREE.Vector3(...bxPts[a]),new THREE.Vector3(...bxPts[b])];
      refs.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({{color:0x0d1525,transparent:true,opacity:0.15}})));
    }});

    sc.add(refs);

    /* ═══ Raycasting tooltip ═══ */
    const rc=new THREE.Raycaster();
    const mv=new THREE.Vector2();
    const tipE=document.getElementById('tip');

    R.domElement.addEventListener('mousemove',e=>{{
      mv.x=(e.clientX/innerWidth)*2-1;
      mv.y=-(e.clientY/innerHeight)*2+1;
      rc.setFromCamera(mv,cam);
      const h=rc.intersectObject(surf);
      if(h.length>0){{
        const p=h[0].point;
        const si=Math.round(((p.x+W/2)/W)*(nS-1));
        const di=Math.round(((p.z+Dep/2)/Dep)*(nD-1));
        if(si>=0&&si<nS&&di>=0&&di<nD){{
          const iv=lk[D.strikes[si]+','+D.dtes[di]]||0;
          tipE.style.display='block';
          tipE.style.left=(e.clientX+16)+'px';
          tipE.style.top=(e.clientY-10)+'px';
          tipE.innerHTML=`<div class="tip-iv">${{iv.toFixed(1)}}%</div>
            <b>Strike:</b> $${{D.strikes[si].toFixed(1)}}<br>
            <b>DTE:</b> ${{D.dtes[di].toFixed(0)}}d`;
        }}
      }}else tipE.style.display='none';
    }});

    /* ═══ Camera ═══ */
    const cv=R.domElement;
    cv.addEventListener('mousedown',e=>{{md=true;lx=e.clientX;ly=e.clientY}});
    cv.addEventListener('mousemove',e=>{{if(!md)return;
      th-=(e.clientX-lx)*0.005;
      ph=Math.max(0.1,Math.min(1.45,ph+(e.clientY-ly)*0.005));
      lx=e.clientX;ly=e.clientY}});
    cv.addEventListener('mouseup',()=>md=false);
    cv.addEventListener('mouseleave',()=>md=false);
    cv.addEventListener('wheel',e=>{{rad=Math.max(10,Math.min(50,rad+e.deltaY*0.015))}});
    cv.addEventListener('touchstart',e=>{{if(e.touches.length===1){{md=true;lx=e.touches[0].clientX;ly=e.touches[0].clientY}}}});
    cv.addEventListener('touchmove',e=>{{if(!md||e.touches.length!==1)return;
      th-=(e.touches[0].clientX-lx)*0.005;
      ph=Math.max(0.1,Math.min(1.45,ph+(e.touches[0].clientY-ly)*0.005));
      lx=e.touches[0].clientX;ly=e.touches[0].clientY}});
    cv.addEventListener('touchend',()=>md=false);

    /* ═══ Animate ═══ */
    let t=0;
    function anim(){{
      requestAnimationFrame(anim);
      t+=0.006;
      if(autoRot)th+=0.001;
      cam.position.set(rad*Math.sin(ph)*Math.cos(th),Math.max(2,rad*Math.cos(ph)),rad*Math.sin(ph)*Math.sin(th));
      cam.lookAt(0,H*0.35,0);
      dia.position.y=H+1.5+Math.sin(t*2.5)*0.12;
      dia.rotation.y=t*1.5;
      dgl.position.y=dia.position.y;
      dgl.scale.setScalar(1+Math.sin(t*3)*0.15);
      pA.position.x=-15+Math.sin(t*0.4)*4;
      pB.position.z=12+Math.cos(t*0.3)*4;
      pC.intensity=0.6+Math.sin(t*0.8)*0.1;
      R.render(sc,cam)}}
    anim();

    addEventListener('resize',()=>{{
      cam.aspect=innerWidth/innerHeight;
      cam.updateProjectionMatrix();
      R.setSize(innerWidth,innerHeight)}});
    </script></body></html>"""

    components.html(html, height=650, scrolling=False)







def render_gex_scenario(data: pd.DataFrame, spot: float, strike_range: float, profiles: dict, dealer: str = "standard"):
    """GEX Scenario Simulator — What happens to gamma if price moves to $X?"""
    st.markdown("##### 🎮 Simulador de Escenarios GEX")
    st.markdown("""
    <div style="color:var(--text-muted); font-size:12px; margin-bottom:12px;">
        Simula cómo cambia la exposición gamma de los dealers cuando se mueve el subyacente.
        Arrastra el slider para ver cómo se recalcula el perfil gamma en cada precio simulado.
    </div>
    """, unsafe_allow_html=True)

    lo_bound = spot * (1 - strike_range / 100)
    hi_bound = spot * (1 + strike_range / 100)

    # Helper: get net GEX at a specific price from a profile
    def gex_at_price(prof, price):
        idx = np.abs(prof["strikes"] - price).argmin()
        return prof["aggregate_gamma"][idx]

    c1, c2 = st.columns([3, 1])
    with c1:
        sim_spot = st.slider(
            "Precio Spot Simulado",
            min_value=float(round(lo_bound, 2)),
            max_value=float(round(hi_bound, 2)),
            value=float(round(spot, 2)),
            step=float(round((hi_bound - lo_bound) / 200, 2)),
            format="$%.2f",
        )
    with c2:
        pct_move = (sim_spot - spot) / spot * 100
        color = "#00FF88" if pct_move >= 0 else "#FF4466"
        st.markdown(f"""
        <div style="text-align:center; padding-top:8px;">
            <div style="font-size:10px; color:#4A5568; text-transform:uppercase; letter-spacing:1px;">Movimiento</div>
            <div style="font-size:24px; font-weight:700; color:{color};">{pct_move:+.2f}%</div>
            <div style="font-size:11px; color:#4A5568;">${sim_spot - spot:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Current profile (already computed)
    profiles_now = profiles

    # Simulated profile (recalculated at new spot)
    profiles_sim = calculate_gamma_profile(data, sim_spot, dealer)

    # Key metrics
    flip_now = profiles_now["gamma_flip"]
    flip_sim = profiles_sim["gamma_flip"]
    gex_now = gex_at_price(profiles_now, spot)
    gex_sim = gex_at_price(profiles_sim, sim_spot)

    regime_now = "γ Positiva" if gex_now >= 0 else "γ Negativa"
    regime_sim = "γ Positiva" if gex_sim >= 0 else "γ Negativa"
    regime_color_sim = "#00FF88" if gex_sim >= 0 else "#FF4466"
    regime_change = (gex_now >= 0) != (gex_sim >= 0)

    cols = st.columns(4)
    with cols[0]:
        st.metric("GEX Neto (actual)", f"${gex_now:.3f}B")
    with cols[1]:
        delta_gex = gex_sim - gex_now
        st.metric("GEX Neto (sim)", f"${gex_sim:.3f}B",
                  delta=f"{delta_gex:+.3f}B")
    with cols[2]:
        st.metric("γ Flip (actual)", f"${flip_now:.2f}")
    with cols[3]:
        st.metric("γ Flip (sim)", f"${flip_sim:.2f}",
                  delta=f"${flip_sim - flip_now:+.2f}")

    if regime_change:
        st.error(f"⚠️ **CAMBIO DE RÉGIMEN** — Pasando de **{regime_now}** a **{regime_sim}**. "
                 f"Cruzar el gamma flip cambia la cobertura de los dealers de "
                 f"{'estabilizador → amplificador' if gex_sim < 0 else 'amplificador → estabilizador'}.")

    # ── Overlay chart: current vs simulated gamma profile ──
    strikes_now = profiles_now["strikes"]
    gamma_now = profiles_now["aggregate_gamma"]
    strikes_sim = profiles_sim["strikes"]
    gamma_sim = profiles_sim["aggregate_gamma"]

    fig = go.Figure()

    # Positive/negative fill for CURRENT
    pos_now = np.where(gamma_now >= 0, gamma_now, 0)
    neg_now = np.where(gamma_now < 0, gamma_now, 0)

    fig.add_trace(go.Scatter(
        x=strikes_now, y=pos_now,
        mode="lines", name="Current (+)", showlegend=False,
        line=dict(width=0), fill="tozeroy",
        fillcolor="rgba(0,255,136,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=strikes_now, y=neg_now,
        mode="lines", name="Current (−)", showlegend=False,
        line=dict(width=0), fill="tozeroy",
        fillcolor="rgba(255,68,102,0.08)",
    ))

    # Current line
    fig.add_trace(go.Scatter(
        x=strikes_now, y=gamma_now,
        mode="lines", name=f"Actual (${spot:.0f})",
        line=dict(color="#00D9FF", width=2.5),
    ))

    # Simulated line
    fig.add_trace(go.Scatter(
        x=strikes_sim, y=gamma_sim,
        mode="lines", name=f"Simulado (${sim_spot:.0f})",
        line=dict(color="#FE53BB", width=2.5, dash="dash"),
    ))

    # Current spot vline
    fig.add_vline(x=spot, line_dash="dash", line_color="#FFD700", line_width=1.5, opacity=0.6)
    fig.add_annotation(x=spot, y=max(gamma_now) * 0.95, text=f"Actual ${spot:.0f}",
                       showarrow=False, font=dict(color="#FFD700", size=10))

    # Simulated spot vline
    if abs(sim_spot - spot) > 0.5:
        fig.add_vline(x=sim_spot, line_dash="dot", line_color="#FE53BB", line_width=1.5, opacity=0.6)
        fig.add_annotation(x=sim_spot, y=max(gamma_sim) * 0.88, text=f"Sim ${sim_spot:.0f}",
                           showarrow=False, font=dict(color="#FE53BB", size=10))

    # Gamma flip markers
    fig.add_vline(x=flip_now, line_dash="dash", line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.add_annotation(x=flip_now, y=min(gamma_now) * 0.8, text=f"Flip ${flip_now:.0f}",
                       showarrow=False, font=dict(color="rgba(255,255,255,0.4)", size=9))

    # Zero line
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)

    fig.update_layout(**_base_layout(
        title=dict(text="Perfil Gamma — Actual vs Simulado", font=dict(size=16)),
        height=440, showlegend=True,
        legend=dict(x=0.01, y=0.99, font=dict(size=10)),
        xaxis=dict(title="Precio ($)"),
        yaxis=dict(title="Gamma Neto ($Bn / 1% mov.)"),
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ── Dealer flow interpretation ──
    if abs(pct_move) > 0.05:
        direction = "sube" if pct_move > 0 else "cae"
        if gex_sim >= 0:
            flow_desc = "los dealers <b>venderán en subidas</b> y <b>comprarán en caídas</b> → estabilizador, reversión a la media"
        else:
            flow_desc = "los dealers <b>comprarán en subidas</b> y <b>venderán en caídas</b> → amplificador, tendencial"

        st.markdown(f"""
        <div style="background:rgba(0,217,255,0.04); border:1px solid rgba(0,217,255,0.1);
                    border-radius:10px; padding:14px 18px; font-size:12px; color:var(--text-secondary); line-height:1.7;">
            <b style="color:#E8ECF4;">Escenario:</b> Si el precio {direction}
            <b>{pct_move:+.1f}%</b> to <b>${sim_spot:.2f}</b>:<br/>
            El gamma neto pasa de <b>${gex_now:.3f}B</b> → <b style="color:{regime_color_sim}">${gex_sim:.3f}B</b>.
            Al precio simulado, {flow_desc}.
            {'<br/><span style="color:#FF4466; font-weight:600;">⚠️ Se cruza el gamma flip — esperar cambio de régimen de volatilidad.</span>' if regime_change else ''}
        </div>
        """, unsafe_allow_html=True)


def chart_vanna_charm(data: pd.DataFrame, spot: float, strike_range: float) -> go.Figure:
    """Vanna and Charm exposure combined chart"""
    lo, hi = spot * (1 - strike_range / 100), spot * (1 + strike_range / 100)
    df = data[(data["strike"] >= lo) & (data["strike"] <= hi)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Exposición Vanna", "Exposición Charm"))

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
        regime_html = '<span class="regime-badge regime-positive">⬢ GAMMA POSITIVA</span>'
    else:
        regime_html = '<span class="regime-badge regime-negative">⬡ GAMMA NEGATIVA</span>'

    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; margin-bottom:8px;">
        <div>{regime_html}</div>
        <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">
            IV Media: {metrics['iv_mean']*100:.1f}% &nbsp;|&nbsp;
            Sesgo IV: {metrics['iv_skew']*100:+.1f}% &nbsp;|&nbsp;
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
            <div class="label">GEX Neto</div>
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
        color, status = "#00FF88", "SEÑAL FUERTE"
    elif prob >= 50:
        color, status = "#FFD700", "MODERADA"
    else:
        color, status = "#FF4466", "DÉBIL"

    direction = prob_data["direction"]
    dist = prob_data["distance_pct"]
    exp_move = prob_data["expected_move"]

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:48px; font-weight:900; color:{color}; font-family:var(--font-display);">{prob}%</div>
            <div style="color:{color}; font-size:12px; font-weight:600; letter-spacing:1px;">{status}</div>
            <div style="color:var(--text-muted); font-size:11px; margin-top:8px;">Prob. de Pinning</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Progress bar
        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                <div>
                    <span style="color:var(--text-muted); font-size:11px; font-family:var(--font-mono); text-transform:uppercase; letter-spacing:1px;">Dirección</span><br/>
                    <span style="color:{color}; font-size:20px; font-weight:700;">{direction}</span>
                </div>
                <div style="text-align:right;">
                    <span style="color:var(--text-muted); font-size:11px; font-family:var(--font-mono); text-transform:uppercase; letter-spacing:1px;">Distancia</span><br/>
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
        speed = "LENTO (Reversión Media)" if metrics["total_gex"] > 0 else "RÁPIDO (Momentum)"
        dte = metrics["days_to_expiry"]
        st.markdown(f"""
        <div class="glass-card">
            <div style="margin-bottom:8px;">
                <span style="color:var(--text-muted); font-size:10px; font-family:var(--font-mono); text-transform:uppercase; letter-spacing:1px;">Movimiento Esperado</span><br/>
                <span style="color:var(--text-primary); font-size:18px; font-weight:700;">±{exp_move:.2f}%</span>
            </div>
            <div style="margin-bottom:8px;">
                <span style="color:var(--text-muted); font-size:10px; font-family:var(--font-mono); text-transform:uppercase; letter-spacing:1px;">Velocidad</span><br/>
                <span style="color:var(--text-secondary); font-size:12px;">{speed}</span>
            </div>
            <div>
                <span style="color:var(--text-muted); font-size:10px; font-family:var(--font-mono); text-transform:uppercase; letter-spacing:1px;">Venc. Próximo</span><br/>
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
                Configuración
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        ticker = st.text_input("TICKER", value="SPY", help="Ej: SPY, QQQ, AAPL, TSLA, NVDA").upper()

        st.markdown("##### Parámetros de Análisis")
        strike_range = st.slider("Rango de Strikes (%)", 5, 50, 10, 5)
        max_exp_days = st.slider("Vencimiento Máx. (días)", 7, 180, 60, 7)
        min_oi = st.number_input("OI Mínimo", 0, 10000, 500, 100)

        st.markdown("##### Filtro de Vencimiento")
        expiry_filter = st.selectbox("Enfoque", ["Todos los Vencimientos", "Solo 0DTE", "≤ 7 DTE", "≤ 30 DTE"],
            help="Filtra opciones por tiempo al vencimiento. 0DTE = solo expiraciones del día.")

        st.markdown("##### Posicionamiento del Dealer")
        dealer = st.selectbox("Modelo", ["standard", "inverse", "neutral"],
            format_func=lambda x: {
                "standard": "Standard (Short Puts, Long Calls)",
                "inverse": "Inverse (Long Puts, Short Calls)",
                "neutral": "Neutral",
            }[x])

        st.markdown("---")

        st.markdown("##### Gráfico de Precio")
        price_days = st.slider("Histórico (días)", 5, 90, 30, 5,
            help="Datos históricos de precio de Yahoo Finance (solo OHLCV del subyacente)")

        # Compare tickers
        st.markdown("##### Comparar Tickers")
        compare_input = st.text_input("Añadir tickers (separados por coma)", placeholder="QQQ, AAPL, TSLA")

        analyze = st.button("⚡ ANALIZAR", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; padding:10px 0;">
            <a href="https://bquantfinance.com" style="color:var(--accent-cyan); text-decoration:none;
                     font-family:var(--font-mono); font-size:11px;">bquantfinance.com</a><br/>
            <a href="https://twitter.com/Gsnchez" style="color:var(--accent-magenta); text-decoration:none;
                     font-family:var(--font-mono); font-size:11px;">@Gsnchez</a>
        </div>
        """, unsafe_allow_html=True)

        return ticker, strike_range, max_exp_days, min_oi, dealer, analyze, compare_input, expiry_filter, price_days


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
def display_results(ticker, spot, data, strike_range, max_exp_days, dealer, price_days=30):
    """Main results display"""

    # ── Calculations ──
    data = calculate_gex(spot, data, dealer)
    data = calculate_dex(spot, data, dealer)
    data = calculate_vanna_exposure(data, spot)
    data = calculate_charm_exposure(data)

    metrics = compute_all_metrics(data, spot)
    max_pain, pain_by_strike, _, mp_expiry, call_pain_curve, put_pain_curve = calculate_max_pain(data, spot)
    mp_term_structure = calculate_max_pain_term_structure(data, spot)
    profiles = calculate_gamma_profile(data, spot, dealer)
    prob = calculate_pinning_probability(max_pain, spot, metrics["total_gex"],
                                         metrics["days_to_expiry"], metrics["iv_mean"])

    # NEW: Walls, Expected Move, 0DTE, Price data (Yahoo = stock OHLCV only, options = CBOE)
    walls = detect_walls(data, spot, strike_range)
    expected_move = calculate_expected_move(data, spot)
    dte0_metrics = compute_0dte_metrics(data, spot, dealer)
    price_df = fetch_price_data(ticker, price_days)
    gamma_flip = profiles["gamma_flip"]

    # ── Key Levels Bar ──
    render_key_levels(spot, max_pain, profiles["gamma_flip"], metrics)

    st.markdown("---")

    # ── Interpretation Panel ──
    col_l, col_r = st.columns(2)
    with col_l:
        if metrics["total_gex"] > 0:
            st.markdown(f"""
            <div class="glass-card">
                <div style="color:#00FF88; font-weight:700; font-size:15px; margin-bottom:8px;">⬢ RÉGIMEN GAMMA POSITIVA</div>
                <div style="color:var(--text-secondary); font-size:13px; line-height:1.7;">
                    Los dealers están <b style="color:#E8ECF4;">largos de gamma</b> — venden en subidas y compran en caídas.<br/>
                    Esperar <b style="color:#00FF88;">volatilidad comprimida</b> y acción de precio con reversión a la media.<br/>
                    Los niveles de soporte/resistencia tienden a mantenerse.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="glass-card">
                <div style="color:#FF4466; font-weight:700; font-size:15px; margin-bottom:8px;">⬡ RÉGIMEN GAMMA NEGATIVA</div>
                <div style="color:var(--text-secondary); font-size:13px; line-height:1.7;">
                    Los dealers están <b style="color:#E8ECF4;">cortos de gamma</b> — persiguen el momentum en ambas direcciones.<br/>
                    Esperar <b style="color:#FF4466;">volatilidad amplificada</b> y movimientos tendenciales.<br/>
                    Las rupturas de rango tienden a persistir.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        direction = "ALCISTA" if max_pain > spot else "BAJISTA" if max_pain < spot else "EN EQUILIBRIO"
        d_color = "#00FF88" if max_pain > spot else "#FF4466" if max_pain < spot else "#FFD700"
        speed = "lento (reversión media)" if metrics["total_gex"] > 0 else "rápido (momentum)"
        st.markdown(f"""
        <div class="glass-card">
            <div style="color:{d_color}; font-weight:700; font-size:15px; margin-bottom:8px;">
                {'↑' if max_pain > spot else '↓' if max_pain < spot else '='} MAX PAIN: {direction}
            </div>
            <div style="color:var(--text-secondary); font-size:13px; line-height:1.7;">
                Objetivo: <b style="color:#E8ECF4;">${max_pain:.2f}</b> ({(max_pain-spot)/spot*100:+.2f}%)<br/>
                Velocidad movimiento esperado: <b style="color:#E8ECF4;">{speed}</b><br/>
                Rango implícito por IV: <b style="color:#E8ECF4;">±{prob['expected_move']:.2f}%</b> ({metrics['days_to_expiry']}d)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Tabs ──
    tabs = st.tabs([
        "📉 Precio y Niveles",
        "🎯 Max Pain",
        "📊 Análisis GEX",
        "📈 Perfil Gamma",
        "🎮 Simulador",
        "⚡ DEX y Griegas",
        "🌊 Superficie 3D",
        "📋 Datos",
    ])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1 — PRICE & LEVELS (NEW: candlestick + GEX levels + EM cone + walls)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[0]:
        if price_df is not None and not price_df.empty:
            st.plotly_chart(
                chart_price_with_levels(price_df, spot, max_pain, gamma_flip,
                                        metrics, walls, expected_move, ticker),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                chart_levels_map(spot, max_pain, gamma_flip, metrics, walls, expected_move, ticker),
                use_container_width=True,
            )

        # ── Expected Move Panel ──
        em = expected_move
        st.markdown("##### Movimiento Esperado")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("IV ATM", f"{em['atm_iv']*100:.1f}%")
        with c2:
            st.metric(f"±1σ ({em['dte']}d)", f"${em['em_1sigma']:.2f}",
                      delta=f"±{em['em_1sigma_pct']:.2f}%")
        with c3:
            st.metric(f"±2σ ({em['dte']}d)", f"${em['em_2sigma']:.2f}",
                      delta=f"±{em['em_2sigma_pct']:.2f}%")
        with c4:
            st.metric("Rango", f"${em['lower_1s']:.2f} — ${em['upper_1s']:.2f}")

        # ── Call & Put Walls ──
        st.markdown("---")
        st.markdown("##### Muros de Calls y Puts")
        wc1, wc2 = st.columns(2)
        with wc1:
            st.markdown("""
            <div style="color:#00FF88; font-weight:700; font-size:13px; margin-bottom:8px;">📗 MUROS DE CALLS (Resistencia)</div>
            """, unsafe_allow_html=True)
            for w in walls["call_walls"][:5]:
                bar_w = min(100, w["oi"] / max(walls["call_walls"][0]["oi"], 1) * 100)
                st.markdown(f"""
                <div style="margin-bottom:6px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; font-family:var(--font-mono);">
                        <span style="color:#E8ECF4; font-weight:600;">${w['strike']:.0f}</span>
                        <span style="color:var(--text-secondary);">{w['oi']:,} OI</span>
                        <span style="color:{'#00FF88' if w['dist_pct'] >= 0 else '#FF4466'};">{w['dist_pct']:+.1f}%</span>
                    </div>
                    <div style="width:100%; height:4px; background:rgba(255,255,255,0.04); border-radius:2px; margin-top:3px;">
                        <div style="width:{bar_w}%; height:100%; background:linear-gradient(90deg, #00FF8800, #00FF88); border-radius:2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with wc2:
            st.markdown("""
            <div style="color:#FF4466; font-weight:700; font-size:13px; margin-bottom:8px;">📕 MUROS DE PUTS (Soporte)</div>
            """, unsafe_allow_html=True)
            for w in walls["put_walls"][:5]:
                bar_w = min(100, w["oi"] / max(walls["put_walls"][0]["oi"], 1) * 100)
                st.markdown(f"""
                <div style="margin-bottom:6px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; font-family:var(--font-mono);">
                        <span style="color:#E8ECF4; font-weight:600;">${w['strike']:.0f}</span>
                        <span style="color:var(--text-secondary);">{w['oi']:,} OI</span>
                        <span style="color:{'#00FF88' if w['dist_pct'] >= 0 else '#FF4466'};">{w['dist_pct']:+.1f}%</span>
                    </div>
                    <div style="width:100%; height:4px; background:rgba(255,255,255,0.04); border-radius:2px; margin-top:3px;">
                        <div style="width:{bar_w}%; height:100%; background:linear-gradient(90deg, #FF446600, #FF4466); border-radius:2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── 0DTE Panel ──
        if dte0_metrics:
            st.markdown("---")
            st.markdown("##### ⚡ Dashboard 0DTE")
            z = dte0_metrics
            total_gex_abs = abs(metrics["total_gex"])
            pct_0dte = abs(z["gex"]) / total_gex_abs * 100 if total_gex_abs > 0 else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("GEX 0DTE", f"${z['gex']:.3f}B")
            with c2:
                st.metric("DEX 0DTE", f"${z['dex']:.3f}B")
            with c3:
                st.metric("% del Total", f"{pct_0dte:.1f}%")
            with c4:
                st.metric("OI Calls 0DTE", f"{z['call_oi']:,}")
            with c5:
                st.metric("OI Puts 0DTE", f"{z['put_oi']:,}")

            st.markdown(f"""
            <div class="glass-card" style="margin-top:8px;">
                <div style="color:var(--text-secondary); font-size:12px; line-height:1.7;">
                    {'<b style="color:#FFD700;">⚡ Alta concentración 0DTE</b> — efectos gamma intradía amplificados. Vigilar pin hacia <b style="color:#E8ECF4;">$' + f"{z['top_strike']:.0f}" + '</b> (strike de máximo GEX 0DTE).' if pct_0dte > 30 else '<b style="color:var(--text-muted);">Gamma 0DTE moderada</b> — vencimientos de varios días dominan el posicionamiento.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2 — MAX PAIN
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[1]:
        # Expiration selector
        available_expiries = sorted(data["expiration"].unique())
        expiry_labels = [f"{e.strftime('%b %d, %Y')} ({max(0,(e - pd.Timestamp.now()).days)}d)"
                         for e in available_expiries]

        if available_expiries:
            selected_idx = st.selectbox(
                "Seleccionar Vencimiento",
                range(len(available_expiries)),
                format_func=lambda i: expiry_labels[i],
                index=0,
                key="mp_expiry_select",
            )
            sel_expiry = available_expiries[selected_idx]

            # Recalculate max pain for selected expiry
            mp_sel, pain_sel, _, _, cp_sel, pp_sel = calculate_max_pain(data, spot, target_expiry=sel_expiry)
            expiry_lbl = sel_expiry.strftime("%b %d, %Y")

            st.plotly_chart(
                chart_max_pain(pain_sel, mp_sel, spot, cp_sel, pp_sel, expiry_lbl),
                use_container_width=True,
            )

            st.markdown("#### Análisis de Probabilidad de Pinning")
            dte_sel = max(0, (sel_expiry - pd.Timestamp.now()).days)
            prob_sel = calculate_pinning_probability(mp_sel, spot, metrics["total_gex"],
                                                     dte_sel, metrics["iv_mean"])
            render_probability_panel(prob_sel, mp_sel, spot, {**metrics, "days_to_expiry": dte_sel})
        else:
            st.warning("No hay fechas de vencimiento disponibles.")

        # Term structure
        if mp_term_structure:
            st.markdown("---")
            st.markdown("##### Estructura Temporal de Max Pain")
            st.plotly_chart(chart_max_pain_term_structure(mp_term_structure, spot), use_container_width=True)

            # Term structure table
            ts_cols = st.columns(min(len(mp_term_structure), 6))
            for i, ts in enumerate(mp_term_structure[:6]):
                with ts_cols[i]:
                    d_pct = ts["distance_pct"]
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center; padding:10px;">
                        <div style="color:var(--text-muted); font-size:10px; font-family:var(--font-mono);">
                            {ts['expiry'].strftime('%b %d')} ({ts['dte']}d)
                        </div>
                        <div style="color:var(--text-primary); font-size:17px; font-weight:700;">
                            ${ts['max_pain']:.2f}
                        </div>
                        <div style="color:{'#00FF88' if d_pct >= 0 else '#FF4466'}; font-size:11px; font-family:var(--font-mono);">
                            {d_pct:+.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2 — GEX ANALYSIS (merges original: by strike, by expiry, calls/puts, cumulative)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[2]:
        # Primary chart: GEX by Strike
        st.plotly_chart(chart_gex_by_strike(spot, data, strike_range), use_container_width=True)

        # Top strikes table
        st.markdown("##### Strikes con Mayor GEX")
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

        st.markdown("##### GEX por Período")
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
    with tabs[3]:
        st.plotly_chart(chart_gamma_profile(profiles, spot, ticker), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Soporte / Resistencia por Picos de Gamma")
            gamma_arr = profiles["aggregate_gamma"]
            strikes_arr = profiles["strikes"]
            peaks = []
            for i in range(1, len(gamma_arr) - 1):
                if gamma_arr[i] > gamma_arr[i - 1] and gamma_arr[i] > gamma_arr[i + 1] and abs(gamma_arr[i]) > 0.05:
                    peaks.append((strikes_arr[i], gamma_arr[i]))
            peaks.sort(key=lambda x: abs(x[1]), reverse=True)
            if peaks:
                for s_val, g_val in peaks[:5]:
                    icon = "🟢 Soporte" if g_val > 0 else "🔴 Resistencia"
                    st.markdown(f"**{icon}** @ ${s_val:.2f} — {abs(g_val):.3f}B")
            else:
                st.markdown("*No se detectaron picos gamma significativos en el rango.*")

            # Gamma profile interpretation (from original)
            max_gamma = max(gamma_arr)
            min_gamma = min(gamma_arr)
            st.markdown("")
            if max_gamma > abs(min_gamma):
                st.markdown(f"""
                <div class="glass-card">
                    <div style="color:#00FF88; font-weight:600; font-size:13px;">Gamma Positiva Dominante</div>
                    <div style="color:var(--text-secondary); font-size:12px; margin-top:4px;">
                        Mercado tiende a la estabilidad. Se espera reversión a la media. Volatilidad comprimida.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="color:#FF4466; font-weight:600; font-size:13px;">Gamma Negativa Dominante</div>
                    <div style="color:var(--text-secondary); font-size:12px; margin-top:4px;">
                        Mercado propenso a movimientos bruscos. Rupturas de rango probables. Volatilidad expandida.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with c2:
            st.plotly_chart(chart_gamma_by_expiry(profiles, spot, ticker), use_container_width=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 4 — GEX SCENARIO SIMULATOR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[4]:
        render_gex_scenario(data, spot, strike_range, profiles, dealer)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 5 — DEX & GREEKS (delta exposure + vanna + charm)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[5]:
        # DEX by strike
        st.plotly_chart(chart_dex_by_strike(spot, data, strike_range), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("DEX Neto", f"${metrics['total_dex']:.3f}B")
        with c2:
            st.metric("DEX Calls", f"${metrics['call_dex']:.3f}B")
        with c3:
            st.metric("DEX Puts", f"${metrics['put_dex']:.3f}B")

        st.markdown("""
        <div class="glass-card" style="margin:16px 0;">
            <div style="color:var(--text-secondary); font-size:13px; line-height:1.7;">
                <b style="color:var(--text-primary);">Delta Exposure (DEX)</b> — presión direccional de cobertura de los dealers.<br/>
                <b style="color:#4488FF;">DEX Positivo</b> → presión alcista (dealers compran al subir el precio) &nbsp;|&nbsp;
                <b style="color:#FF8C42;">DEX Negativo</b> → presión bajista (dealers venden al caer el precio)<br/>
                Combinar con GEX: <b>GEX</b> = régimen de volatilidad, <b>DEX</b> = sesgo direccional.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Vanna & Charm
        st.markdown("---")
        st.plotly_chart(chart_vanna_charm(data, spot, strike_range), use_container_width=True)
        st.markdown("""
        <div class="glass-card">
            <div style="color:var(--text-secondary); font-size:13px; line-height:1.7;">
                <b style="color:#A855F7;">Vanna</b> = sensibilidad del delta a cambios de IV.
                Cuando la IV cae, los flujos de vanna empujan a los dealers a comprar (+) o vender (−).<br/>
                <b style="color:#00BFA6;">Charm</b> = decaimiento del delta en el tiempo.
                Muestra cómo cambia la presión de cobertura con el tiempo — crítico para dinámicas de OPEX.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 6 — 3D IV SURFACE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[6]:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("IV Media", f"{metrics['iv_mean']*100:.1f}%")
        with c2:
            st.metric("IV Calls", f"{metrics['iv_calls']*100:.1f}%")
        with c3:
            st.metric("IV Puts", f"{metrics['iv_puts']*100:.1f}%")
        with c4:
            skew = metrics["iv_skew"] * 100
            st.metric("Sesgo IV", f"{skew:+.1f}%",
                      delta="Prima Put" if skew > 0 else "Prima Call",
                      delta_color="inverse" if skew > 0 else "normal")

        st.markdown("""
        <div style="color:var(--text-muted); font-size:12px; margin-bottom:4px; font-family:var(--font-mono);">
            🖱️ Arrastra para orbitar · Scroll para zoom · Hover para ver IV
        </div>
        """, unsafe_allow_html=True)
        render_3d_iv_surface(data, spot, strike_range, metrics)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 7 — DATA (table + CSV download)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[7]:
        st.markdown("##### Datos de Opciones")
        c1, c2, c3 = st.columns(3)
        with c1:
            type_f = st.selectbox("Tipo", ["Todos", "Calls", "Puts"])
        with c2:
            sort_f = st.selectbox("Ordenar por", ["GEX", "DEX", "open_interest", "volume", "strike", "iv"])
        with c3:
            n_rows = st.number_input("Filas", 10, 100, 25)

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
        st.download_button("⬇ Descargar CSV", data=csv,
                          file_name=f"{ticker}_gex_pro_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                          mime="text/csv")


def show_landing():
    """Landing page / educational content"""
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:40px;">
        <div style="font-size:64px; margin-bottom:16px;">⚡</div>
        <div class="hero-title" style="font-size:36px;">Análisis Profesional de Flujo de Opciones</div>
        <div style="color:var(--text-secondary); font-size:15px; margin-top:12px; max-width:600px; margin-left:auto; margin-right:auto; line-height:1.7;">
            Analiza Exposición Gamma, Delta, Max Pain, Vanna y Charm
            en toda la cadena de opciones — con datos CBOE gratuitos.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    cols = st.columns(3)
    cards = [
        ("GEX", "#00D9FF", "La Exposición Gamma revela cómo la cobertura de dealers crea niveles de soporte/resistencia y determina el régimen de volatilidad.",
         "GEX > 0 → Estabilidad  ·  GEX < 0 → Volatilidad"),
        ("Max Pain", "#00FF88", "El strike donde más opciones expiran sin valor. Actúa como imán de precio, especialmente cerca de OPEX.",
         "0DTE → Pin fuerte  ·  30d+ → Señal débil"),
        ("DEX + Vanna", "#FE53BB", "Delta Exposure muestra presión direccional. Vanna muestra cómo los cambios de IV afectan los flujos de cobertura.",
         "DEX → Dirección  ·  Vanna → Sensibilidad IV"),
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

    with st.expander("📖 Cómo Usar Esta Herramienta", expanded=False):
        st.markdown("""
        **1. Introduce un ticker** — SPY, QQQ, AAPL, TSLA, etc.

        **2. Configura parámetros** — Ajusta rango de strikes, ventana de vencimiento y OI mínimo.

        **3. Analiza el dashboard:**
        - **Barra de Niveles Clave** — Spot, Max Pain, Gamma Flip, Strike Max GEX de un vistazo
        - **Indicador de Régimen** — Ver al instante si estamos en gamma positiva o negativa
        - **Pestaña Max Pain** — Dirección y probabilidad de pinning con modelo mejorado por IV
        - **Perfil Gamma** — Curva completa de exposición con identificación de soporte/resistencia
        - **Terreno 3D** — Visualización interactiva Three.js de la superficie gamma
        - **Mapa de Calor** — Mapa de concentración Strike × Vencimiento
        - **DEX** — Presión direccional de cobertura por exposición delta
        - **Superficie IV** — Smile de volatilidad a lo largo de los vencimientos
        - **Vanna y Charm** — Exposición de griegas de segundo orden para dinámicas de OPEX

        **4. Buenas Prácticas:**
        - Vencimientos 0DTE/semanales: Max pain es más efectivo
        - Gamma Flip: Nivel crítico donde cambia el comportamiento del mercado
        - GEX + DEX combinados: Visión completa del posicionamiento de dealers
        - IV Skew > 0: Prima de puts = demanda de cobertura bajista
        """)

    st.markdown("""
    <div style="text-align:center; margin-top:40px; padding:20px;">
        <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted); letter-spacing:2px;">
            CREADO PARA LA COMUNIDAD DE TRADING CUANTITATIVO
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
def apply_expiry_filter(data: pd.DataFrame, expiry_filter: str) -> pd.DataFrame:
    """Apply expiry filter to options data"""
    if expiry_filter == "Todos los Vencimientos":
        return data
    today = pd.Timestamp.now().normalize()
    if expiry_filter == "Solo 0DTE":
        return data[data["expiration"].dt.normalize() == today]
    elif expiry_filter == "≤ 7 DTE":
        cutoff = today + timedelta(days=7)
        return data[data["expiration"] <= cutoff]
    elif expiry_filter == "≤ 30 DTE":
        cutoff = today + timedelta(days=30)
        return data[data["expiration"] <= cutoff]
    return data


def main():
    render_hero()
    ticker, strike_range, max_exp_days, min_oi, dealer, analyze, compare_input, expiry_filter, price_days = render_sidebar()

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

                        # Apply expiry filter
                        filtered = apply_expiry_filter(odata, expiry_filter)
                        if filtered.empty:
                            st.warning(f"No hay opciones con el filtro '{expiry_filter}'. Mostrando todos los vencimientos.")
                            filtered = odata

                        if expiry_filter != "Todos los Vencimientos":
                            st.info(f"🔍 Mostrando: **{expiry_filter}** ({len(filtered):,} contratos)")

                        display_results(ticker, spot, filtered, strike_range, max_exp_days, dealer, price_days)

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
                                            cmax_pain, _, _, _, _, _ = calculate_max_pain(codata, cspot)
                                            cprofiles = calculate_gamma_profile(codata, cspot, dealer)
                                            compare_data.append({
                                                "ticker": ct, "spot": cspot,
                                                "max_pain": cmax_pain,
                                                "gamma_flip": cprofiles["gamma_flip"],
                                                **cmetrics,
                                            })

                            if compare_data:
                                main_metrics = compute_all_metrics(
                                    calculate_gex(spot, odata, dealer), spot)
                                main_mp, _, _, _, _, _ = calculate_max_pain(odata, spot)
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
                        st.error("No hay datos válidos tras filtrar. Prueba reduciendo el OI mínimo.")
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
        filtered = apply_expiry_filter(td["data"], expiry_filter)
        if filtered.empty:
            filtered = td["data"]
        if expiry_filter != "Todos los Vencimientos":
            st.info(f"🔍 Mostrando: **{expiry_filter}** ({len(filtered):,} contratos)")
        display_results(td["ticker"], td["spot"], filtered, strike_range, max_exp_days,
                       td.get("dealer", "standard"), price_days)
    else:
        show_landing()


if __name__ == "__main__":
    main()
