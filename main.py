"""
GEX Analyzer + Max Pain + Gamma Profiles - Versión Avanzada Completa
Desarrollado por @Gsnchez - bquantfinance.com
"""

import streamlit as st
import json
import os
from datetime import timedelta, datetime
from typing import Tuple, Optional
import warnings
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
from plotly.subplots import make_subplots
from scipy.stats import norm

warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="GEX Analyzer Pro | bquantfinance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para tema oscuro mejorado
st.markdown("""
<style>
    /* Tema oscuro personalizado */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Headers con gradiente */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00D9FF, #FE53BB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* Métricas personalizadas */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(0,217,255,0.1), rgba(254,83,187,0.1));
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(0,217,255,0.05), rgba(254,83,187,0.05));
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 20px 0;
        backdrop-filter: blur(5px);
    }
    
    /* Botones personalizados */
    .stButton > button {
        background: linear-gradient(90deg, #00D9FF, #FE53BB);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(254,83,187,0.4);
    }
    
    /* Tabs mejorados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, rgba(0,217,255,0.05), rgba(254,83,187,0.05));
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(90deg, rgba(0,217,255,0.2), rgba(254,83,187,0.2));
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00D9FF, #FE53BB);
    }
</style>
""", unsafe_allow_html=True)

# Constantes
CONTRACT_SIZE = 100
CACHE_DIR = "data"
API_BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options"

# Inicializar estado de la sesión
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'ticker_data' not in st.session_state:
    st.session_state.ticker_data = {}
if 'gamma_profile_cache' not in st.session_state:
    st.session_state.gamma_profile_cache = {}

def ensure_cache_dir():
    """Crear directorio de caché si no existe"""
    os.makedirs(CACHE_DIR, exist_ok=True)

@st.cache_data(ttl=300)
def fetch_option_data(ticker: str) -> Optional[dict]:
    """Obtener datos de opciones desde CBOE API con caché"""
    ensure_cache_dir()
    
    urls = [
        f"{API_BASE_URL}/_{ticker}.json",
        f"{API_BASE_URL}/{ticker}.json"
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            continue
    return None

def parse_option_data(raw_data: dict) -> Tuple[float, pd.DataFrame]:
    """Parsear datos crudos de opciones"""
    try:
        data = pd.DataFrame.from_dict(raw_data)
        spot_price = float(data.loc["current_price", "data"])
        option_data = pd.DataFrame(data.loc["options", "data"])
        return spot_price, option_data
    except Exception as e:
        st.error(f"Error parseando datos: {e}")
        return 0, pd.DataFrame()

def process_option_data_optimized(data: pd.DataFrame) -> pd.DataFrame:
    """OPTIMIZADO: Procesar y limpiar datos de opciones usando vectorización"""
    df = data.copy()
    
    df["type"] = df.option.str.extract(r'\d([CP])\d')
    df["strike_raw"] = df.option.str.extract(r'[CP](\d+)').astype(float)
    df["strike"] = df["strike_raw"] / 1000
    df["expiration_str"] = df.option.str.extract(r'[A-Z]+(\d{6})')
    df["expiration"] = pd.to_datetime(df["expiration_str"], format="%y%m%d", errors='coerce')
    
    numeric_cols = ['gamma', 'open_interest', 'volume', 'delta', 'vega', 'theta', 'iv']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    mask = (
        df['type'].notna() & 
        df['strike'].notna() & 
        df['expiration'].notna() & 
        df['gamma'].notna() & 
        df['open_interest'].notna() &
        (df['open_interest'] > 0) & 
        (df['gamma'] > 0)
    )
    
    return df[mask]

def calculate_gex_optimized(spot: float, data: pd.DataFrame, dealer_position: str = "standard") -> pd.DataFrame:
    """OPTIMIZADO: Cálculo vectorizado de GEX"""
    df = data.copy()
    
    df["GEX"] = df["gamma"] * df["open_interest"] * CONTRACT_SIZE * (spot ** 2) * 0.01
    
    if dealer_position == "standard":
        df["GEX"] = np.where(df["type"] == "P", -df["GEX"], df["GEX"])
    elif dealer_position == "inverse":
        df["GEX"] = np.where(df["type"] == "P", df["GEX"], -df["GEX"])
    
    total_gex = df["GEX"].sum()
    df["GEX_pct"] = (df["GEX"] / total_gex * 100) if total_gex != 0 else 0
    df["days_to_expiry"] = (df["expiration"] - pd.Timestamp.now()).dt.days
    
    return df

def calculate_max_pain_optimized(data: pd.DataFrame, spot_price: float) -> tuple:
    """OPTIMIZADO: Cálculo de Max Pain usando NumPy"""
    strikes = np.sort(data['strike'].unique())
    
    calls = data[data['type'] == 'C'][['strike', 'open_interest']].values
    puts = data[data['type'] == 'P'][['strike', 'open_interest']].values
    
    pain_by_strike = {}
    
    for exp_price in strikes:
        call_itm_mask = calls[:, 0] < exp_price
        call_pain = np.sum((exp_price - calls[call_itm_mask, 0]) * calls[call_itm_mask, 1] * CONTRACT_SIZE)
        
        put_itm_mask = puts[:, 0] > exp_price
        put_pain = np.sum((puts[put_itm_mask, 0] - exp_price) * puts[put_itm_mask, 1] * CONTRACT_SIZE)
        
        pain_by_strike[exp_price] = call_pain + put_pain
    
    if pain_by_strike:
        max_pain_strike = min(pain_by_strike, key=pain_by_strike.get)
        min_pain_value = pain_by_strike[max_pain_strike]
    else:
        max_pain_strike = spot_price
        min_pain_value = 0
    
    return max_pain_strike, pain_by_strike, min_pain_value

def calculate_all_metrics_batch(data: pd.DataFrame, spot_price: float) -> dict:
    """Calcular todas las métricas en batch"""
    gex_by_strike = data.groupby("strike")["GEX"].sum()
    gex_by_type = data.groupby("type")["GEX"].sum()
    oi_by_type = data.groupby("type")["open_interest"].sum()
    
    metrics = {
        'total_gex': data["GEX"].sum() / 1e9,
        'call_gex': gex_by_type.get('C', 0) / 1e9,
        'put_gex': gex_by_type.get('P', 0) / 1e9,
        'call_oi': oi_by_type.get('C', 0),
        'put_oi': oi_by_type.get('P', 0),
        'max_gex_strike': gex_by_strike.abs().idxmax() if len(gex_by_strike) > 0 else spot_price,
        'top_5_strikes': gex_by_strike.abs().nlargest(5).to_dict(),
        'nearest_expiry': data['expiration'].min(),
        'unique_call_strikes': data[data['type'] == 'C']['strike'].nunique(),
        'unique_put_strikes': data[data['type'] == 'P']['strike'].nunique(),
    }
    
    metrics['put_call_ratio'] = abs(metrics['put_gex'] / metrics['call_gex']) if metrics['call_gex'] != 0 else 0
    metrics['days_to_expiry'] = max(0, (metrics['nearest_expiry'] - pd.Timestamp.now()).days) if pd.notna(metrics['nearest_expiry']) else 0
    
    return metrics

# ===== NUEVAS FUNCIONES PARA GAMMA PROFILE =====

def black_scholes_gamma(S, K, T, r, sigma):
    """Calcula gamma usando Black-Scholes"""
    if T <= 0 or sigma <= 0:
        return 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

@st.cache_data(ttl=60)
def calculate_gex_at_spot(data, spot_level, r=0.05):
    """Calcula GEX total en un nivel de spot específico"""
    total_gex = 0
    
    for _, row in data.iterrows():
        T = row['days_to_expiry'] / 365
        if T <= 0:
            T = 1/365
        
        iv = row.get('iv', 0.20)
        if pd.isna(iv) or iv <= 0:
            iv = 0.20
            
        gamma = black_scholes_gamma(spot_level, row['strike'], T, r, iv)
        option_gex = gamma * row['open_interest'] * CONTRACT_SIZE * spot_level * spot_level * 0.01
        
        if row['type'] == 'P':
            option_gex = -option_gex
            
        total_gex += option_gex
    
    return total_gex / 1e9

def create_gamma_profile_chart(data, spot_price, strike_range_pct=20):
    """Crea el gráfico principal de Gamma Profile"""
    min_spot = spot_price * (1 - strike_range_pct/100)
    max_spot = spot_price * (1 + strike_range_pct/100)
    spot_levels = np.linspace(min_spot, max_spot, 80)  # Reducido para performance
    
    gex_profile = []
    with st.spinner('Calculando Gamma Profile...'):
        progress = st.progress(0)
        for i, level in enumerate(spot_levels):
            gex = calculate_gex_at_spot(data, level)
            gex_profile.append(gex)
            progress.progress((i + 1) / len(spot_levels))
        progress.empty()
    
    gex_array = np.array(gex_profile)
    zero_crossings = np.where(np.diff(np.sign(gex_array)))[0]
    
    gamma_flip = None
    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        x1, x2 = spot_levels[idx], spot_levels[idx + 1]
        y1, y2 = gex_array[idx], gex_array[idx + 1]
        gamma_flip = x1 - y1 * (x2 - x1) / (y2 - y1)
    
    fig = go.Figure()
    
    # Área sombreada negativa (volatilidad alta)
    neg_mask = gex_array < 0
    if np.any(neg_mask):
        fig.add_trace(go.Scatter(
            x=spot_levels[neg_mask],
            y=gex_array[neg_mask],
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Área sombreada positiva (volatilidad baja)
    pos_mask = gex_array > 0
    if np.any(pos_mask):
        fig.add_trace(go.Scatter(
            x=spot_levels[pos_mask],
            y=gex_array[pos_mask],
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Línea principal de Gamma Profile
    fig.add_trace(go.Scatter(
        x=spot_levels,
        y=gex_profile,
        mode='lines',
        name='Gamma Profile',
        line=dict(color='#00D9FF', width=3),
        hovertemplate='Spot: $%{x:.2f}<br>GEX: %{y:.2f}B<br><extra></extra>'
    ))
    
    # Líneas verticales importantes
    fig.add_vline(
        x=spot_price,
        line_dash="dash",
        line_color="#FFD700",
        line_width=2,
        annotation_text=f"Spot: ${spot_price:.2f}",
        annotation_position="top left"
    )
    
    if gamma_flip:
        fig.add_vline(
            x=gamma_flip,
            line_dash="dot",
            line_color="#00FF00",
            line_width=2,
            annotation_text=f"Gamma Flip: ${gamma_flip:.2f}",
            annotation_position="bottom right"
        )
    
    fig.add_hline(y=0, line_color="gray", line_width=1, opacity=0.5)
    
    # Anotaciones de zonas
    if gamma_flip:
        fig.add_annotation(
            x=min_spot + (gamma_flip - min_spot) / 2,
            y=max(gex_profile) * 0.9,
            text="← ZONA VOLÁTIL →",
            showarrow=False,
            font=dict(color="#FF6B6B", size=12),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#FF6B6B",
            borderwidth=1
        )
        
        fig.add_annotation(
            x=gamma_flip + (max_spot - gamma_flip) / 2,
            y=max(gex_profile) * 0.9,
            text="← ZONA ESTABLE →",
            showarrow=False,
            font=dict(color="#00FF00", size=12),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#00FF00",
            borderwidth=1
        )
    
    title_text = f'🌊 Gamma Exposure Profile - '
    if gamma_flip:
        distance_to_flip = ((gamma_flip - spot_price) / spot_price * 100)
        title_text += f'Flip @ ${gamma_flip:.2f} ({distance_to_flip:+.1f}%)'
    else:
        title_text += 'No Flip Detectado'
    
    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title="Precio Spot ($)",
        yaxis_title="Gamma Exposure ($Bn / 1% move)",
        template="plotly_dark",
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )
    
    return fig, gamma_flip

def create_gamma_profile_by_expiry(data, spot_price):
    """Gamma Profile excluyendo diferentes expiraciones"""
    next_expiry = data['expiration'].min()
    
    def is_third_friday(date):
        return date.weekday() == 4 and 15 <= date.day <= 21
    
    data['is_monthly'] = data['expiration'].apply(is_third_friday)
    monthly_expiries = data[data['is_monthly']]['expiration'].unique()
    next_monthly = monthly_expiries.min() if len(monthly_expiries) > 0 else next_expiry
    
    min_spot = spot_price * 0.85
    max_spot = spot_price * 1.15
    spot_levels = np.linspace(min_spot, max_spot, 60)  # Menos puntos para performance
    
    profile_all = []
    profile_ex_next = []
    profile_ex_monthly = []
    
    with st.spinner('Calculando perfiles por expiración...'):
        for level in spot_levels:
            gex_all = calculate_gex_at_spot(data, level)
            profile_all.append(gex_all)
            
            data_ex_next = data[data['expiration'] != next_expiry]
            if not data_ex_next.empty:
                gex_ex_next = calculate_gex_at_spot(data_ex_next, level)
                profile_ex_next.append(gex_ex_next)
            else:
                profile_ex_next.append(0)
            
            if next_monthly != next_expiry:
                data_ex_monthly = data[data['expiration'] != next_monthly]
                if not data_ex_monthly.empty:
                    gex_ex_monthly = calculate_gex_at_spot(data_ex_monthly, level)
                    profile_ex_monthly.append(gex_ex_monthly)
                else:
                    profile_ex_monthly.append(0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=spot_levels,
        y=profile_all,
        mode='lines',
        name='Todas las Expiraciones',
        line=dict(color='#00D9FF', width=3),
        hovertemplate='GEX: %{y:.2f}B<br><extra></extra>'
    ))
    
    if profile_ex_next:
        fig.add_trace(go.Scatter(
            x=spot_levels,
            y=profile_ex_next,
            mode='lines',
            name=f'Sin {next_expiry.strftime("%b %d")}',
            line=dict(color='#FFA500', width=2.5),
            hovertemplate='GEX: %{y:.2f}B<br><extra></extra>'
        ))
    
    if next_monthly != next_expiry and profile_ex_monthly:
        fig.add_trace(go.Scatter(
            x=spot_levels,
            y=profile_ex_monthly,
            mode='lines',
            name=f'Sin Monthly {next_monthly.strftime("%b %d")}',
            line=dict(color='#90EE90', width=2),
            hovertemplate='GEX: %{y:.2f}B<br><extra></extra>'
        ))
    
    fig.add_vline(
        x=spot_price,
        line_dash="dash",
        line_color="#FFD700",
        line_width=2,
        annotation_text=f"Spot: ${spot_price:.2f}"
    )
    
    fig.add_hline(y=0, line_color="gray", line_width=1, opacity=0.5)
    
    fig.update_layout(
        title='📅 Gamma Profile por Expiración',
        xaxis_title="Precio Spot ($)",
        yaxis_title="Gamma Exposure ($Bn / 1% move)",
        template="plotly_dark",
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )
    
    return fig

def calculate_key_levels(data, spot_price):
    """Calcula niveles clave de GEX"""
    min_spot = spot_price * 0.85
    max_spot = spot_price * 1.15
    spot_levels = np.linspace(min_spot, max_spot, 100)
    
    gex_profile = []
    for level in spot_levels:
        gex = calculate_gex_at_spot(data, level)
        gex_profile.append(gex)
    
    gex_array = np.array(gex_profile)
    
    zero_crossings = np.where(np.diff(np.sign(gex_array)))[0]
    gamma_flip = None
    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        x1, x2 = spot_levels[idx], spot_levels[idx + 1]
        y1, y2 = gex_array[idx], gex_array[idx + 1]
        gamma_flip = x1 - y1 * (x2 - x1) / (y2 - y1)
    
    max_gamma_idx = np.argmax(gex_array)
    max_gamma_level = spot_levels[max_gamma_idx]
    
    min_gamma_idx = np.argmin(gex_array)
    min_gamma_level = spot_levels[min_gamma_idx]
    
    return {
        'gamma_flip': gamma_flip,
        'max_gamma_level': max_gamma_level,
        'min_gamma_level': min_gamma_level,
        'current_gex': gex_array[np.argmin(np.abs(spot_levels - spot_price))]
    }

# ===== FUNCIONES DE GRÁFICOS BÁSICOS =====

def create_max_pain_chart_optimized(pain_by_strike: dict, max_pain: float, spot: float):
    """Gráfico de Max Pain limpio"""
    if not pain_by_strike:
        return go.Figure()
    
    strikes = sorted(pain_by_strike.keys())
    
    if len(strikes) > 100:
        step = len(strikes) // 100
        strikes_sampled = strikes[::step]
        if max_pain not in strikes_sampled:
            strikes_sampled.append(max_pain)
        if spot not in strikes_sampled:
            strikes_sampled.append(spot)
        strikes_sampled.sort()
    else:
        strikes_sampled = strikes
    
    pain_values = [pain_by_strike.get(s, 0) / 1e9 for s in strikes_sampled]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=strikes_sampled,
        y=pain_values,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#FF6B6B', width=2),
        fillcolor='rgba(255, 107, 107, 0.2)',
        name='Dolor Total',
        hovertemplate='Strike: $%{x:.2f}<br>Dolor: $%{y:.2f}B<br><extra></extra>'
    ))
    
    if max_pain in pain_by_strike:
        fig.add_trace(go.Scatter(
            x=[max_pain],
            y=[pain_by_strike[max_pain] / 1e9],
            mode='markers+text',
            marker=dict(size=15, color='#00FF00', symbol='diamond', line=dict(width=2, color='white')),
            text=['MAX PAIN'],
            textposition='top center',
            textfont=dict(size=14, color='#00FF00', family='Arial Black'),
            name='Max Pain',
            showlegend=False
        ))
    
    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_color="#FFD700",
        line_width=2,
        annotation_text=f"Spot: ${spot:.2f}",
        annotation_position="top left"
    )
    
    fig.add_vline(
        x=max_pain,
        line_dash="dot",
        line_color="#00FF00",
        line_width=1,
        opacity=0.5,
        annotation_text=f"Target: ${max_pain:.2f}",
        annotation_position="bottom"
    )
    
    distance_pct = ((max_pain - spot) / spot * 100)
    direction = "📈" if max_pain > spot else "📉"
    
    fig.update_layout(
        title={
            'text': f'🎯 MAX PAIN: ${max_pain:.2f} | Spot: ${spot:.2f} | {direction} {abs(distance_pct):.2f}%',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title="Precio Strike ($)",
        yaxis_title="Dolor Total ($B)",
        template="plotly_dark",
        height=450,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_gex_by_strike_plot(spot: float, data: pd.DataFrame, strike_range: float):
    """Gráfico de GEX por strike"""
    gex_by_strike = data.groupby("strike")["GEX"].sum() / 1e9
    
    strike_min = spot * (1 - strike_range/100)
    strike_max = spot * (1 + strike_range/100)
    mask = (gex_by_strike.index >= strike_min) & (gex_by_strike.index <= strike_max)
    gex_filtered = gex_by_strike[mask]
    
    fig = go.Figure()
    
    colors = ['#00D9FF' if x > 0 else '#FE53BB' for x in gex_filtered.values]
    
    fig.add_trace(go.Bar(
        x=gex_filtered.index,
        y=gex_filtered.values,
        marker_color=colors,
        opacity=0.8,
        hovertemplate='Strike: $%{x:.2f}<br>GEX: %{y:.3f}B<br><extra></extra>'
    ))
    
    fig.add_vline(x=spot, line_dash="dash", line_color="#FFD700", line_width=2,
                  annotation_text=f"Spot: ${spot:.2f}")
    
    if not gex_filtered.empty:
        max_gex_strike = gex_filtered.abs().idxmax()
        fig.add_vline(x=max_gex_strike, line_dash="dot", line_color="#FF6B6B", 
                      line_width=1.5, opacity=0.7,
                      annotation_text=f"Max GEX: ${max_gex_strike:.2f}")
    
    fig.update_layout(
        title='📊 Exposición Gamma por Strike',
        xaxis_title="Precio Strike ($)",
        yaxis_title="Exposición Gamma ($Bn / 1% movimiento)",
        template="plotly_dark",
        height=450,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_strike_distribution_plot(spot: float, data: pd.DataFrame):
    """Gráfico de distribución Calls vs Puts"""
    calls_data = data[data['type'] == 'C'].groupby('strike')['GEX'].sum() / 1e9
    puts_data = data[data['type'] == 'P'].groupby('strike')['GEX'].sum() / 1e9
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=calls_data.index,
        y=calls_data.values,
        name='Calls',
        marker_color='#00D9FF',
        opacity=0.7,
        hovertemplate='Strike: $%{x:.2f}<br>Call GEX: %{y:.3f}B<br><extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=puts_data.index,
        y=puts_data.values,
        name='Puts',
        marker_color='#FE53BB',
        opacity=0.7,
        hovertemplate='Strike: $%{x:.2f}<br>Put GEX: %{y:.3f}B<br><extra></extra>'
    ))
    
    fig.add_vline(x=spot, line_dash="dash", line_color="#FFD700", line_width=2,
                  annotation_text=f"Spot: ${spot:.2f}")
    
    fig.update_layout(
        title='🎯 Distribución GEX - Calls vs Puts',
        xaxis_title="Precio Strike ($)",
        yaxis_title="Exposición Gamma ($Bn / 1% movimiento)",
        template="plotly_dark",
        height=450,
        barmode='overlay',
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    return fig

def calculate_pinning_probability(max_pain: float, spot: float, gex: float, days_to_expiry: int) -> dict:
    """Calcular probabilidad de alcanzar max pain"""
    distance_pct = abs(max_pain - spot) / spot * 100
    
    base_prob = 50
    
    if days_to_expiry == 0:
        expiry_factor = 30
    elif days_to_expiry <= 1:
        expiry_factor = 20
    elif days_to_expiry <= 7:
        expiry_factor = 10
    else:
        expiry_factor = 5
    
    if distance_pct < 0.5:
        distance_factor = 20
    elif distance_pct < 1:
        distance_factor = 15
    elif distance_pct < 2:
        distance_factor = 10
    elif distance_pct < 3:
        distance_factor = 5
    else:
        distance_factor = 0
    
    gex_factor = 10 if gex > 0 else -5
    
    probability = min(95, max(5, base_prob + expiry_factor + distance_factor + gex_factor))
    
    return {
        'probability': probability,
        'direction': 'UP' if max_pain > spot else 'DOWN',
        'distance': distance_pct
    }

def display_probability_analysis_clean(max_pain, spot_price, metrics, prob_analysis):
    """Diseño limpio para el análisis de probabilidad"""
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    
    with col1:
        prob = prob_analysis['probability']
        
        if prob > 70:
            color = "#00FF00"
            bg_color = "rgba(0, 255, 0, 0.1)"
            status = "SEÑAL FUERTE"
            emoji = "🟢"
        elif prob > 50:
            color = "#FFD700"
            bg_color = "rgba(255, 215, 0, 0.1)"
            status = "SEÑAL MEDIA"
            emoji = "🟡"
        else:
            color = "#FF6B6B"
            bg_color = "rgba(255, 107, 107, 0.1)"
            status = "SEÑAL DÉBIL"
            emoji = "🔴"
        
        st.markdown(f"""
        <div style='background: {bg_color}; border: 2px solid {color}; 
                    border-radius: 15px; padding: 15px; text-align: center;'>
            <div style='font-size: 36px; color: {color}; font-weight: bold;'>
                {prob}%
            </div>
            <div style='color: {color}; font-size: 14px; margin-top: 5px;'>
                {emoji} {status}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        days = metrics['days_to_expiry']
        if days == 0:
            st.markdown("""
            <div style='text-align: center; padding: 10px;'>
                <div style='font-size: 28px; color: #00FF00;'>0DTE</div>
                <div style='color: #888; font-size: 11px;'>¡EXPIRA HOY!</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='text-align: center; padding: 10px;'>
                <div style='font-size: 28px; color: white;'>{days}d</div>
                <div style='color: #888; font-size: 11px;'>Días Restantes</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        distance = prob_analysis['distance']
        dist_color = "#00FF00" if distance < 1 else "#FFD700" if distance < 3 else "#FF6B6B"
        st.markdown(f"""
        <div style='text-align: center; padding: 10px;'>
            <div style='font-size: 28px; color: {dist_color};'>{distance:.1f}%</div>
            <div style='color: #888; font-size: 11px;'>Distancia</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if max_pain != spot_price:
            direction = "📈 ALCISTA" if max_pain > spot_price else "📉 BAJISTA"
            target_move = abs(max_pain - spot_price)
            speed = "Movimiento LENTO" if metrics['total_gex'] > 0 else "Movimiento RÁPIDO"
            gex_sign = "GEX +" if metrics['total_gex'] > 0 else "GEX -"
            
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); border-radius: 10px; padding: 10px;'>
                <div style='color: white; font-size: 16px; font-weight: bold;'>{direction}</div>
                <div style='color: #888; font-size: 12px; margin-top: 5px;'>
                    Target: ${max_pain:.2f} (${target_move:.2f})<br>
                    {speed} ({gex_sign})
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    progress_html = f"""
    <div style='width: 100%; height: 30px; background: rgba(255,255,255,0.1); border-radius: 15px; overflow: hidden;'>
        <div style='width: {prob}%; height: 100%; background: linear-gradient(90deg, #FF6B6B, #FFD700, #00FF00); 
                    display: flex; align-items: center; justify-content: end; padding-right: 10px;
                    transition: all 0.5s ease;'>
            <span style='color: black; font-weight: bold;'>{prob}%</span>
        </div>
    </div>
    <div style='display: flex; justify-content: space-between; margin-top: 5px; color: #888; font-size: 12px;'>
        <span>0% - Muy Improbable</span>
        <span>50% - Neutral</span>
        <span>100% - Muy Probable</span>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)

# INTERFAZ PRINCIPAL
def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <h1 style='font-size: 48px;'>🎯 GEX ANALYZER PRO</h1>
            <p style='font-size: 18px; color: #00D9FF;'>Análisis Avanzado con Gamma Profiles</p>
            <p style='font-size: 14px; color: #FE53BB;'>Desarrollado por @Gsnchez | bquantfinance.com</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        
        ticker = st.text_input(
            "📈 Símbolo del Activo",
            value="SPY",
            help="Ingrese el símbolo del ticker"
        ).upper()
        
        st.markdown("### 🎛️ Parámetros de Análisis")
        
        strike_range = st.slider(
            "Rango de Strikes (%)",
            min_value=5,
            max_value=30,
            value=10,
            step=5
        )
        
        max_expiration_days = st.slider(
            "Días hasta Vencimiento",
            min_value=7,
            max_value=90,
            value=30,
            step=7
        )
        
        min_open_interest = st.number_input(
            "Interés Abierto Mínimo",
            min_value=0,
            max_value=10000,
            value=500,
            step=100
        )
        
        st.markdown("### 🏦 Posicionamiento")
        dealer_position = st.selectbox(
            "Dealers",
            ["standard", "inverse", "neutral"],
            format_func=lambda x: {
                "standard": "Estándar (Short Puts, Long Calls)",
                "inverse": "Inverso (Long Puts, Short Calls)",
                "neutral": "Neutral (Sin Asunción)"
            }[x]
        )
        
        analyze_button = st.button(
            "🚀 Ejecutar Análisis",
            use_container_width=True,
            type="primary"
        )
    
    # Área principal
    if analyze_button:
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            status.text('🔄 Conectando con CBOE...')
            progress_bar.progress(20)
            raw_data = fetch_option_data(ticker)
            
            if raw_data:
                status.text('📊 Procesando opciones...')
                progress_bar.progress(40)
                spot_price, option_data = parse_option_data(raw_data)
                
                if not option_data.empty:
                    option_data = process_option_data_optimized(option_data)
                    option_data = option_data[option_data['open_interest'] >= min_open_interest]
                    
                    if not option_data.empty:
                        status.text('📈 Calculando GEX...')
                        progress_bar.progress(60)
                        option_data = calculate_gex_optimized(spot_price, option_data, dealer_position)
                        
                        status.text('🎯 Calculando Max Pain...')
                        progress_bar.progress(80)
                        
                        st.session_state.data_loaded = True
                        st.session_state.ticker_data = {
                            'ticker': ticker,
                            'spot': spot_price,
                            'data': option_data
                        }
                        
                        status.text('✅ ¡Análisis completado!')
                        progress_bar.progress(100)
                        
                        progress_bar.empty()
                        status.empty()
                        
                        display_results(ticker, spot_price, option_data, strike_range, max_expiration_days)
                    else:
                        progress_bar.empty()
                        status.empty()
                        st.error("❌ No hay datos válidos")
                else:
                    progress_bar.empty()
                    status.empty()
                    st.error("❌ No se encontraron opciones")
            else:
                progress_bar.empty()
                status.empty()
                st.error(f"❌ No se pudieron obtener datos para {ticker}")
        
        except Exception as e:
            progress_bar.empty()
            status.empty()
            st.error(f"❌ Error: {str(e)}")
    
    elif st.session_state.data_loaded:
        ticker = st.session_state.ticker_data['ticker']
        spot_price = st.session_state.ticker_data['spot']
        option_data = st.session_state.ticker_data['data']
        display_results(ticker, spot_price, option_data, strike_range, max_expiration_days)
    
    else:
        show_educational_content()

def display_results(ticker, spot_price, option_data, strike_range, max_expiration_days):
    """Mostrar resultados con análisis avanzado"""
    
    metrics = calculate_all_metrics_batch(option_data, spot_price)
    max_pain, pain_by_strike, total_pain = calculate_max_pain_optimized(option_data, spot_price)
    
    # Métricas principales
    st.markdown("### 📊 Métricas Principales")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💰 Spot", f"${spot_price:.2f}")
    
    with col2:
        st.metric("🎯 Max Pain", f"${max_pain:.2f}",
                 delta=f"{((max_pain-spot_price)/spot_price*100):+.1f}%")
    
    with col3:
        st.metric("📊 GEX Total", f"${metrics['total_gex']:.2f}B",
                 delta="Positivo" if metrics['total_gex'] > 0 else "Negativo",
                 delta_color="normal" if metrics['total_gex'] > 0 else "inverse")
    
    with col4:
        st.metric("📈 Calls", f"${metrics['call_gex']:.2f}B")
    
    with col5:
        st.metric("📉 Puts", f"${metrics['put_gex']:.2f}B")
    
    # Interpretación
    st.markdown("### 🎯 Interpretación del Mercado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if metrics['total_gex'] > 0:
            st.success("""
            **📈 GEX POSITIVO - Dealers LARGOS**
            - Volatilidad reducida
            - Movimientos suaves
            """)
        else:
            st.warning("""
            **📉 GEX NEGATIVO - Dealers CORTOS**
            - Volatilidad amplificada
            - Movimientos bruscos
            """)
    
    with col2:
        if abs(max_pain - spot_price) > 0.01:
            direction = "ALCISTA" if max_pain > spot_price else "BAJISTA"
            speed = "LENTO" if metrics['total_gex'] > 0 else "RÁPIDO"
            st.info(f"""
            **🎯 SEÑAL: {direction}**
            - Target: ${max_pain:.2f}
            - Movimiento: **{speed}**
            """)
    
    # Tabs mejorados
    tabs = st.tabs([
        "💎 MAX PAIN",
        "🌊 GAMMA PROFILE", 
        "📊 Por Strike",
        "🎯 Calls vs Puts",
        "🔬 Análisis Avanzado"
    ])
    
    with tabs[0]:
        fig = create_max_pain_chart_optimized(pain_by_strike, max_pain, spot_price)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 🎲 Probabilidad de Pin")
        
        prob_analysis = calculate_pinning_probability(
            max_pain, spot_price, metrics['total_gex'], metrics['days_to_expiry']
        )
        
        display_probability_analysis_clean(max_pain, spot_price, metrics, prob_analysis)
    
    with tabs[1]:
        st.markdown("#### 🌊 Gamma Exposure Profile")
        
        profile_fig, gamma_flip = create_gamma_profile_chart(option_data, spot_price, strike_range)
        st.plotly_chart(profile_fig, use_container_width=True)
        
        # Métricas del Gamma Profile
        key_levels = calculate_key_levels(option_data, spot_price)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if key_levels['gamma_flip']:
                flip_dist = ((key_levels['gamma_flip']-spot_price)/spot_price*100)
                st.metric(
                    "🔄 Gamma Flip",
                    f"${key_levels['gamma_flip']:.2f}",
                    delta=f"{flip_dist:+.1f}%"
                )
            else:
                st.metric("🔄 Gamma Flip", "No detectado")
        
        with col2:
            st.metric(
                "📈 Max Gamma",
                f"${key_levels['max_gamma_level']:.2f}",
                delta="Zona estable"
            )
        
        with col3:
            st.metric(
                "📉 Min Gamma",
                f"${key_levels['min_gamma_level']:.2f}",
                delta="Zona volátil"
            )
        
        # Explicación del Gamma Flip
        if key_levels['gamma_flip']:
            if spot_price > key_levels['gamma_flip']:
                st.info("""
                **✅ Estamos en Zona ESTABLE (por encima del Gamma Flip)**
                - Los dealers estabilizan el mercado
                - Venden rallies, compran caídas
                - Menor volatilidad esperada
                """)
            else:
                st.warning("""
                **⚠️ Estamos en Zona VOLÁTIL (por debajo del Gamma Flip)**
                - Los dealers amplifican movimientos
                - Compran rallies, venden caídas
                - Mayor volatilidad esperada
                """)
    
    with tabs[2]:
        fig = create_gex_by_strike_plot(spot_price, option_data, strike_range)
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        fig = create_strike_distribution_plot(spot_price, option_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:
        st.markdown("#### 📅 Gamma Profile por Expiración")
        expiry_fig = create_gamma_profile_by_expiry(option_data, spot_price)
        st.plotly_chart(expiry_fig, use_container_width=True)
        
        st.info("""
        **📊 Cómo interpretar estos perfiles:**
        
        • **Línea Azul (Todas)**: GEX actual con todas las expiraciones
        • **Línea Naranja (Sin próxima)**: Cómo quedará el GEX después del próximo vencimiento
        • **Línea Verde (Sin mensual)**: Impacto de la expiración mensual (OPEX)
        
        **💡 Señales clave:**
        - Si las líneas divergen mucho → Gran impacto del vencimiento
        - Si el Gamma Flip cambia entre líneas → Cambio de régimen esperado
        """)

def show_educational_content():
    """Contenido educativo actualizado"""
    st.markdown("""
    <div class='info-box'>
    <h2>📚 GEX Analyzer Pro - Guía Completa</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌊 Gamma Profile
        
        El **Gamma Profile** muestra cómo cambiará el GEX si el precio se mueve.
        
        **Gamma Flip Point:**
        - Precio donde GEX = 0
        - Divide zonas volátiles de estables
        - Actúa como imán o barrera
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Max Pain + GEX
        
        **La combinación perfecta:**
        - Max Pain = DÓNDE (precio objetivo)
        - GEX = CÓMO (velocidad)
        - Gamma Flip = CUÁNDO (cambio de régimen)
        """)
    
    st.markdown("""
    ### 🔬 Conceptos Avanzados
    
    **1. Gamma Flip** - El nivel más importante:
    - Por encima: Mercado estable, volatilidad baja
    - Por debajo: Mercado nervioso, volatilidad alta
    - En el flip: Máxima incertidumbre
    
    **2. Perfiles por Expiración**:
    - Muestra el impacto de cada vencimiento
    - Identifica cambios estructurales futuros
    
    **3. Zonas de Control**:
    - Verde: Dealers controlan, movimientos suaves
    - Roja: Dealers amplifican, movimientos violentos
    """)
    
    st.markdown("""
    <div style='margin-top: 50px; padding: 20px; background: linear-gradient(135deg, rgba(0,217,255,0.1), rgba(254,83,187,0.1)); border-radius: 15px;'>
        <h4 style='text-align: center;'>🚀 Herramientas Institucionales para Todos</h4>
        <p style='text-align: center;'>
            <a href='https://bquantfinance.com' style='color: #00D9FF;'>bquantfinance.com</a> | 
            <a href='https://twitter.com/Gsnchez' style='color: #FE53BB;'>@Gsnchez</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
