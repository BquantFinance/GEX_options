"""
Analizador de Exposición Gamma (GEX) con Max Pain
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

# Configuración de la página
st.set_page_config(
    page_title="GEX Analyzer | bquantfinance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para tema oscuro
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
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e, #0E1117);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(0,217,255,0.1), rgba(254,83,187,0.1));
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Success/Warning/Error boxes con gradientes */
    .stSuccess, .stWarning, .stError {
        background: linear-gradient(135deg, rgba(0,0,0,0.4), rgba(0,0,0,0.2));
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0,217,255,0.3);
        border-radius: 50%;
        border-top-color: #00D9FF;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
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

def ensure_cache_dir():
    """Crear directorio de caché si no existe"""
    os.makedirs(CACHE_DIR, exist_ok=True)

@st.cache_data(ttl=3600)
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

def process_option_data(data: pd.DataFrame) -> pd.DataFrame:
    """Procesar y limpiar datos de opciones"""
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
    
    df = df.dropna(subset=['type', 'strike', 'expiration', 'gamma', 'open_interest'])
    df = df[(df['open_interest'] > 0) & (df['gamma'] > 0)]
    
    return df

def calculate_gex(spot: float, data: pd.DataFrame, dealer_position: str = "standard") -> pd.DataFrame:
    """Calcular Exposición Gamma (GEX)"""
    df = data.copy()
    
    df["GEX"] = df["gamma"] * df["open_interest"] * CONTRACT_SIZE * (spot ** 2) * 0.01
    
    if dealer_position == "standard":
        # Estándar: dealers cortos en puts, largos en calls
        df["GEX"] = df.apply(lambda x: -x["GEX"] if x["type"] == "P" else x["GEX"], axis=1)
    elif dealer_position == "inverse":
        # Inverso: dealers largos en puts, cortos en calls
        df["GEX"] = df.apply(lambda x: x["GEX"] if x["type"] == "P" else -x["GEX"], axis=1)
    # Si es "neutral", no aplicamos signo
    
    df["GEX_pct"] = df["GEX"] / df["GEX"].sum() * 100 if df["GEX"].sum() != 0 else 0
    df["days_to_expiry"] = (df["expiration"] - datetime.now()).dt.days
    
    return df

def calculate_max_pain(data: pd.DataFrame, spot_price: float) -> tuple:
    """
    Calcular Max Pain - el precio donde la mayoría de opciones expiran sin valor
    """
    # Obtener strikes únicos
    strikes = sorted(data['strike'].unique())
    
    # Calcular dolor para cada precio potencial de expiración
    pain_by_strike = {}
    
    for exp_price in strikes:
        total_pain = 0
        
        # Calcular dolor de calls (calls ITM cuestan dinero a dealers)
        calls = data[data['type'] == 'C']
        for _, call in calls.iterrows():
            if call['strike'] < exp_price:
                # Call ITM - dealers pagan
                intrinsic_value = exp_price - call['strike']
                pain = intrinsic_value * call['open_interest'] * CONTRACT_SIZE
                total_pain += pain
        
        # Calcular dolor de puts (puts ITM cuestan dinero a dealers)
        puts = data[data['type'] == 'P']
        for _, put in puts.iterrows():
            if put['strike'] > exp_price:
                # Put ITM - dealers pagan
                intrinsic_value = put['strike'] - exp_price
                pain = intrinsic_value * put['open_interest'] * CONTRACT_SIZE
                total_pain += pain
        
        pain_by_strike[exp_price] = total_pain
    
    # Encontrar strike con mínimo dolor (max pain para compradores de opciones)
    if pain_by_strike:
        max_pain_strike = min(pain_by_strike, key=pain_by_strike.get)
        min_pain_value = pain_by_strike[max_pain_strike]
    else:
        max_pain_strike = spot_price
        min_pain_value = 0
    
    return max_pain_strike, pain_by_strike, min_pain_value

def create_max_pain_chart(pain_by_strike: dict, max_pain: float, spot: float):
    """
    Crear visualización de Max Pain
    """
    if not pain_by_strike:
        return go.Figure()
    
    strikes = sorted(pain_by_strike.keys())
    pain_values = [pain_by_strike[s] / 1e9 for s in strikes]  # Convertir a billones
    
    fig = go.Figure()
    
    # Curva de dolor
    fig.add_trace(go.Scatter(
        x=strikes,
        y=pain_values,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#FF6B6B', width=3),
        fillcolor='rgba(255, 107, 107, 0.2)',
        name='Dolor Total',
        hovertemplate='Strike: $%{x:.2f}<br>Dolor: $%{y:.2f}B<br><extra></extra>'
    ))
    
    # Punto Max Pain
    if max_pain in pain_by_strike:
        fig.add_trace(go.Scatter(
            x=[max_pain],
            y=[pain_by_strike[max_pain] / 1e9],
            mode='markers+text',
            marker=dict(size=15, color='#00FF00', symbol='diamond', line=dict(color='white', width=2)),
            text=['MAX PAIN'],
            textposition='top center',
            textfont=dict(size=14, color='#00FF00', family='Arial Black'),
            name='Max Pain',
            hovertemplate='Max Pain: $%{x:.2f}<br><extra></extra>'
        ))
    
    # Precio spot actual
    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_color="#FFD700",
        line_width=2,
        annotation_text=f"Spot: ${spot:.2f}",
        annotation_position="top"
    )
    
    # Añadir flecha mostrando dirección
    if abs(spot - max_pain) > 0.5 and pain_values:
        # Flecha desde spot hacia max pain
        arrow_color = "#00FF00" if max_pain > spot else "#FF0000"
        fig.add_annotation(
            x=max_pain,
            y=max(pain_values) * 0.5,
            ax=spot,
            ay=max(pain_values) * 0.5,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor=arrow_color,
            opacity=0.7
        )
        
        # Texto de dirección
        direction_text = "→" if max_pain > spot else "←"
        fig.add_annotation(
            x=(max_pain + spot) / 2,
            y=max(pain_values) * 0.5,
            text=f"<b>{direction_text} ${abs(max_pain - spot):.2f}</b>",
            showarrow=False,
            font=dict(color=arrow_color, size=16),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor=arrow_color,
            borderwidth=1
        )
    
    fig.update_layout(
        title={
            'text': f'🎯 MAX PAIN: ${max_pain:.2f} | Distancia: {((max_pain-spot)/spot*100):.2f}%',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': 'white', 'family': 'Arial Black'}
        },
        xaxis_title="Precio Strike ($)",
        yaxis_title="Dolor Total ($B)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            range=[min(strikes) - 5, max(strikes) + 5]
        ),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=500,
        hovermode='x unified'
    )
    
    return fig

def calculate_pinning_probability(max_pain: float, spot: float, gex: float, days_to_expiry: int) -> dict:
    """
    Calcular probabilidad de alcanzar max pain basado en condiciones actuales
    """
    distance_pct = abs(max_pain - spot) / spot * 100
    
    # Factores base de probabilidad
    base_prob = 50  # Comenzar en 50%
    
    # Ajustar por días hasta expiración (más fuerte cerca de expiración)
    if days_to_expiry == 0:  # 0DTE
        expiry_factor = 30
    elif days_to_expiry <= 1:
        expiry_factor = 20
    elif days_to_expiry <= 7:
        expiry_factor = 10
    else:
        expiry_factor = 5
    
    # Ajustar por distancia (más cerca = mayor probabilidad)
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
    
    # Ajustar por GEX (GEX positivo = mayor probabilidad de pin)
    if gex > 0:
        gex_factor = 10
    else:
        gex_factor = -5  # GEX negativo hace el pin menos probable
    
    # Calcular probabilidad final
    probability = min(95, max(5, base_prob + expiry_factor + distance_factor + gex_factor))
    
    # Sugerencia de trading basada en probabilidad
    if probability > 70:
        if max_pain > spot:
            suggestion = "📈 COMPRAR: Alta probabilidad de subida hacia Max Pain"
            strategy = "Comprar Calls ATM o Bull Call Spreads"
        else:
            suggestion = "📉 VENDER: Alta probabilidad de caída hacia Max Pain"
            strategy = "Comprar Puts ATM o Bear Put Spreads"
    elif probability > 50:
        suggestion = "⚖️ NEUTRAL: Probabilidad moderada de pin"
        strategy = "Iron Condors o Butterflies centrados en Max Pain"
    else:
        suggestion = "⚠️ CUIDADO: Baja probabilidad de pin"
        strategy = "Evitar estrategias basadas en Max Pain"
    
    return {
        'probability': probability,
        'suggestion': suggestion,
        'strategy': strategy,
        'direction': 'UP' if max_pain > spot else 'DOWN',
        'distance': distance_pct
    }

def create_gex_by_strike_plot(spot: float, data: pd.DataFrame, strike_range: float):
    """Crear gráfico de GEX por strike"""
    gex_by_strike = data.groupby("strike")["GEX"].sum() / 1e9
    
    strike_min = spot * (1 - strike_range/100)
    strike_max = spot * (1 + strike_range/100)
    mask = (gex_by_strike.index >= strike_min) & (gex_by_strike.index <= strike_max)
    gex_filtered = gex_by_strike[mask]
    
    fig = go.Figure()
    
    # Barras con colores según signo
    colors = ['#00D9FF' if x > 0 else '#FE53BB' for x in gex_filtered.values]
    
    fig.add_trace(go.Bar(
        x=gex_filtered.index,
        y=gex_filtered.values,
        marker_color=colors,
        opacity=0.8,
        name='GEX',
        hovertemplate='Strike: $%{x:.2f}<br>GEX: %{y:.3f}B<br><extra></extra>'
    ))
    
    # Línea de precio spot
    fig.add_vline(
        x=spot, 
        line_dash="dash", 
        line_color="#FFD700", 
        line_width=2,
        annotation_text=f"Spot: ${spot:.2f}",
        annotation_position="top"
    )
    
    # Strike con máximo GEX
    if not gex_filtered.empty:
        max_gex_strike = gex_filtered.abs().idxmax()
        fig.add_vline(
            x=max_gex_strike, 
            line_dash="dot", 
            line_color="#FF6B6B", 
            line_width=1.5,
            opacity=0.7,
            annotation_text=f"Max GEX: ${max_gex_strike:.2f}",
            annotation_position="bottom"
        )
    
    fig.update_layout(
        title={
            'text': '📊 Exposición Gamma por Strike',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': 'white'}
        },
        xaxis_title="Precio Strike ($)",
        yaxis_title="Exposición Gamma ($Bn / 1% movimiento)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', zeroline=True, zerolinecolor='rgba(255,255,255,0.2)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', zeroline=True, zerolinecolor='rgba(255,255,255,0.2)'),
        height=500
    )
    
    return fig

def create_gex_by_expiration_plot(data: pd.DataFrame, max_days: int):
    """Crear gráfico de GEX por vencimiento"""
    max_date = datetime.now() + timedelta(days=max_days)
    data_filtered = data[data["expiration"] <= max_date]
    
    gex_by_exp = data_filtered.groupby("expiration")["GEX"].sum() / 1e9
    
    fig = go.Figure()
    
    colors = ['#00D9FF' if x > 0 else '#FE53BB' for x in gex_by_exp.values]
    
    fig.add_trace(go.Bar(
        x=gex_by_exp.index,
        y=gex_by_exp.values,
        marker_color=colors,
        opacity=0.8,
        name='GEX',
        hovertemplate='Fecha: %{x|%b %d}<br>GEX: %{y:.3f}B<br><extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '📅 Exposición Gamma por Vencimiento',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': 'white'}
        },
        xaxis_title="Fecha de Vencimiento",
        yaxis_title="Exposición Gamma ($Bn / 1% movimiento)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickformat='%b %d'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', zeroline=True, zerolinecolor='rgba(255,255,255,0.2)'),
        height=500
    )
    
    return fig

def create_strike_distribution_plot(spot: float, data: pd.DataFrame):
    """Crear gráfico de distribución de GEX por tipo de opción"""
    # Separar calls y puts
    calls_data = data[data['type'] == 'C'].groupby('strike')['GEX'].sum() / 1e9
    puts_data = data[data['type'] == 'P'].groupby('strike')['GEX'].sum() / 1e9
    
    fig = go.Figure()
    
    # Añadir calls
    fig.add_trace(go.Bar(
        x=calls_data.index,
        y=calls_data.values,
        name='Calls',
        marker_color='#00D9FF',
        opacity=0.7,
        hovertemplate='Strike: $%{x:.2f}<br>Call GEX: %{y:.3f}B<br><extra></extra>'
    ))
    
    # Añadir puts
    fig.add_trace(go.Bar(
        x=puts_data.index,
        y=puts_data.values,
        name='Puts',
        marker_color='#FE53BB',
        opacity=0.7,
        hovertemplate='Strike: $%{x:.2f}<br>Put GEX: %{y:.3f}B<br><extra></extra>'
    ))
    
    # Línea de precio spot
    fig.add_vline(
        x=spot, 
        line_dash="dash", 
        line_color="#FFD700", 
        line_width=2,
        annotation_text=f"Spot: ${spot:.2f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title={
            'text': '🎯 Distribución GEX - Calls vs Puts',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': 'white'}
        },
        xaxis_title="Precio Strike ($)",
        yaxis_title="Exposición Gamma ($Bn / 1% movimiento)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', zeroline=True, zerolinecolor='rgba(255,255,255,0.2)'),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)'),
        barmode='overlay',
        height=500
    )
    
    return fig

def create_cumulative_gex_plot(data: pd.DataFrame):
    """Crear gráfico de GEX acumulativo por días hasta vencimiento"""
    # Agrupar por días hasta vencimiento
    data_copy = data.copy()
    data_copy['days_to_expiry'] = (data_copy['expiration'] - datetime.now()).dt.days
    
    # Filtrar datos válidos
    data_copy = data_copy[data_copy['days_to_expiry'] >= 0]
    
    # Calcular GEX acumulativo
    gex_by_days = data_copy.groupby('days_to_expiry')['GEX'].sum().sort_index() / 1e9
    gex_cumulative = gex_by_days.cumsum()
    
    fig = go.Figure()
    
    # Línea de GEX acumulativo
    fig.add_trace(go.Scatter(
        x=gex_cumulative.index,
        y=gex_cumulative.values,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#00D9FF', width=3),
        fillcolor='rgba(0, 217, 255, 0.2)',
        name='GEX Acumulativo',
        hovertemplate='Días: %{x}<br>GEX Acum: %{y:.3f}B<br><extra></extra>'
    ))
    
    # Marcadores para vencimientos importantes
    major_expirations = [30, 60, 90, 180, 365]
    for exp in major_expirations:
        if exp in gex_cumulative.index:
            fig.add_vline(
                x=exp,
                line_dash="dot",
                line_color="rgba(255,255,255,0.3)",
                line_width=1,
                annotation_text=f"{exp}d",
                annotation_position="top"
            )
    
    fig.update_layout(
        title={
            'text': '📈 GEX Acumulativo por Tiempo hasta Vencimiento',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': 'white'}
        },
        xaxis_title="Días hasta Vencimiento",
        yaxis_title="GEX Acumulativo ($Bn)",
        hovermode='x',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=500
    )
    
    return fig

# INTERFAZ PRINCIPAL
def main():
    # Header con logo y título
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <h1 style='font-size: 48px;'>🎯 GEX ANALYZER + MAX PAIN</h1>
            <p style='font-size: 18px; color: #00D9FF;'>Análisis Profesional de Exposición Gamma y Max Pain</p>
            <p style='font-size: 14px; color: #FE53BB;'>Desarrollado por @Gsnchez | bquantfinance.com</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar con configuración
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        
        # Input del ticker
        ticker = st.text_input(
            "📈 Símbolo del Activo",
            value="SPY",
            help="Ingrese el símbolo del ticker (ej: SPY, AAPL, TSLA)"
        ).upper()
        
        st.markdown("### 🎛️ Parámetros de Análisis")
        
        # Parámetros configurables
        strike_range = st.slider(
            "Rango de Strikes (%)",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            help="Porcentaje alrededor del precio spot a analizar"
        )
        
        max_expiration_days = st.slider(
            "Días Máximos hasta Vencimiento",
            min_value=30,
            max_value=720,
            value=365,
            step=30,
            help="Filtrar opciones por días hasta vencimiento"
        )
        
        min_open_interest = st.number_input(
            "Interés Abierto Mínimo",
            min_value=0,
            max_value=10000,
            value=100,
            step=100,
            help="Filtrar opciones con bajo interés abierto"
        )
        
        st.markdown("### 🏦 Posicionamiento de Dealers")
        dealer_position = st.selectbox(
            "Asunción de Posicionamiento",
            ["standard", "inverse", "neutral"],
            format_func=lambda x: {
                "standard": "Estándar (Short Puts, Long Calls)",
                "inverse": "Inverso (Long Puts, Short Calls)",
                "neutral": "Neutral (Sin Asunción)"
            }[x],
            help="Cómo asumir el posicionamiento de los dealers"
        )
        
        # Botón de análisis
        analyze_button = st.button(
            "🚀 Ejecutar Análisis",
            use_container_width=True,
            type="primary"
        )
    
    # Área principal
    if analyze_button:
        with st.spinner('🔄 Obteniendo datos de opciones...'):
            # Obtener datos
            raw_data = fetch_option_data(ticker)
            
            if raw_data:
                spot_price, option_data = parse_option_data(raw_data)
                
                if not option_data.empty:
                    # Procesar datos
                    option_data = process_option_data(option_data)
                    
                    # Filtrar por interés abierto mínimo
                    option_data = option_data[option_data['open_interest'] >= min_open_interest]
                    
                    if not option_data.empty:
                        # Calcular GEX
                        option_data = calculate_gex(spot_price, option_data, dealer_position)
                        
                        # Guardar en session state
                        st.session_state.data_loaded = True
                        st.session_state.ticker_data = {
                            'ticker': ticker,
                            'spot': spot_price,
                            'data': option_data
                        }
                        
                        # Mostrar resultados
                        display_results(ticker, spot_price, option_data, strike_range, max_expiration_days)
                    else:
                        st.error("❌ No hay datos válidos después del filtrado")
                else:
                    st.error("❌ No se encontraron datos de opciones")
            else:
                st.error(f"❌ No se pudieron obtener datos para {ticker}")
    
    # Mostrar resultados anteriores si existen
    elif st.session_state.data_loaded:
        ticker = st.session_state.ticker_data['ticker']
        spot_price = st.session_state.ticker_data['spot']
        option_data = st.session_state.ticker_data['data']
        display_results(ticker, spot_price, option_data, strike_range, max_expiration_days)
    
    # Guía educativa
    else:
        show_educational_content()

def display_results(ticker, spot_price, option_data, strike_range, max_expiration_days):
    """Mostrar resultados del análisis"""
    
    # Calcular métricas
    total_gex = option_data["GEX"].sum() / 1e9
    call_gex = option_data[option_data["type"] == "C"]["GEX"].sum() / 1e9
    put_gex = option_data[option_data["type"] == "P"]["GEX"].sum() / 1e9
    put_call_ratio = abs(put_gex / call_gex) if call_gex != 0 else 0
    
    # Calcular Max Pain
    max_pain, pain_by_strike, total_pain = calculate_max_pain(option_data, spot_price)
    
    # Métricas principales con Max Pain
    st.markdown("### 📊 Métricas Principales")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Precio Spot",
            f"${spot_price:.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "🎯 Max Pain",
            f"${max_pain:.2f}",
            delta=f"{((max_pain-spot_price)/spot_price*100):+.1f}%"
        )
    
    with col3:
        st.metric(
            "📊 GEX Total",
            f"${total_gex:.2f}B",
            delta=f"{'Positivo' if total_gex > 0 else 'Negativo'}",
            delta_color="normal" if total_gex > 0 else "inverse"
        )
    
    with col4:
        st.metric(
            "📈 GEX Calls",
            f"${call_gex:.2f}B",
            delta=f"{(call_gex/total_gex*100):.1f}%" if total_gex != 0 else "0%"
        )
    
    with col5:
        st.metric(
            "📉 GEX Puts",
            f"${put_gex:.2f}B",
            delta=f"P/C: {put_call_ratio:.2f}"
        )
    
    # Interpretación del mercado con Max Pain
    st.markdown("### 🎯 Interpretación del Mercado")
    
    # Análisis combinado GEX + Max Pain
    col1, col2 = st.columns(2)
    
    with col1:
        if total_gex > 0:
            st.success("""
            **📈 GEX POSITIVO - Dealers LARGOS en Gamma**
            - Los creadores de mercado VENDERÁN en rallies y COMPRARÁN en caídas
            - Esto actúa como un **amortiguador de volatilidad**
            - El mercado tiende a moverse de forma más **ordenada y predecible**
            - Niveles de soporte y resistencia más **respetados**
            """)
        else:
            st.warning("""
            **📉 GEX NEGATIVO - Dealers CORTOS en Gamma**
            - Los creadores de mercado COMPRARÁN en rallies y VENDERÁN en caídas
            - Esto actúa como un **amplificador de volatilidad**
            - El mercado puede experimentar movimientos más **bruscos y extremos**
            - Mayor probabilidad de **rupturas de niveles clave**
            """)
    
    with col2:
        # Max Pain signal
        if abs(max_pain - spot_price) > 0.01:
            if max_pain > spot_price:
                st.info(f"""
                **🎯 MAX PAIN SEÑAL: ALCISTA**
                - Max Pain en ${max_pain:.2f} (${max_pain - spot_price:+.2f})
                - Precio tiende a subir hacia Max Pain
                - {'Movimiento LENTO' if total_gex > 0 else 'Movimiento RÁPIDO'} esperado
                - **Estrategia**: {'Vender Puts OTM' if total_gex > 0 else 'Comprar Calls ATM'}
                """)
            else:
                st.info(f"""
                **🎯 MAX PAIN SEÑAL: BAJISTA**
                - Max Pain en ${max_pain:.2f} (${max_pain - spot_price:+.2f})
                - Precio tiende a bajar hacia Max Pain
                - {'Movimiento LENTO' if total_gex > 0 else 'Movimiento RÁPIDO'} esperado
                - **Estrategia**: {'Vender Calls OTM' if total_gex > 0 else 'Comprar Puts ATM'}
                """)
        else:
            st.success("""
            **✅ PRECIO EN MAX PAIN**
            - El precio está en equilibrio
            - Baja volatilidad esperada
            - **Estrategia**: Vender volatilidad (Strangles/Straddles)
            """)
    
    # Tabs para diferentes visualizaciones
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "💎 MAX PAIN", 
        "📊 Por Strike", 
        "📅 Por Vencimiento", 
        "🎯 Calls vs Puts", 
        "📈 GEX Acumulativo", 
        "📋 Datos",
        "📚 Estrategias"
    ])
    
    with tab1:
        # MAX PAIN Analysis
        fig = create_max_pain_chart(pain_by_strike, max_pain, spot_price)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de probabilidad
        st.markdown("#### 🎲 Análisis de Probabilidad de Pin")
        
        # Calcular días hasta expiración más cercana
        nearest_expiry = option_data['expiration'].min()
        days_to_exp = max(0, (nearest_expiry - pd.Timestamp.now()).days)
        
        prob_analysis = calculate_pinning_probability(max_pain, spot_price, total_gex, days_to_exp)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # Gauge de probabilidad
            st.metric(
                "📊 Probabilidad de Pin",
                f"{prob_analysis['probability']}%",
                delta=prob_analysis['direction']
            )
            
            # Indicador visual
            if prob_analysis['probability'] > 70:
                st.success("✅ SEÑAL FUERTE")
            elif prob_analysis['probability'] > 50:
                st.warning("⚖️ SEÑAL MODERADA")
            else:
                st.error("❌ SEÑAL DÉBIL")
        
        with col2:
            st.metric("⏰ Días hasta Exp", f"{days_to_exp}d", 
                     delta="0DTE!" if days_to_exp == 0 else None)
            st.metric("📏 Distancia", f"{prob_analysis['distance']:.2f}%",
                     delta="Cerca" if prob_analysis['distance'] < 1 else "Lejos")
        
        with col3:
            st.info(f"""
            **{prob_analysis['suggestion']}**
            
            📊 **Estrategia Recomendada:**
            {prob_analysis['strategy']}
            
            💡 **Factores Clave:**
            - {'✅ 0DTE: Máxima atracción' if days_to_exp == 0 else f'📅 {days_to_exp} días hasta expiración'}
            - {'✅ GEX Positivo facilita el pin' if total_gex > 0 else '⚠️ GEX Negativo dificulta el pin'}
            - {'✅ Muy cerca del Max Pain' if prob_analysis['distance'] < 1 else '⚠️ Lejos del Max Pain'}
            """)
    
    with tab2:
        fig = create_gex_by_strike_plot(spot_price, option_data, strike_range)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top strikes
        st.markdown("#### 🎯 Top 5 Strikes con Mayor GEX")
        top_strikes = option_data.groupby('strike')['GEX'].sum().abs().nlargest(5)
        for strike, gex in top_strikes.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"Strike ${strike:.2f}")
            with col2:
                st.write(f"${gex/1e9:.3f}B")
            with col3:
                distance = ((strike - spot_price) / spot_price * 100)
                st.write(f"{distance:+.1f}% desde spot")
    
    with tab3:
        fig = create_gex_by_expiration_plot(option_data, max_expiration_days)
        st.plotly_chart(fig, use_container_width=True)
        
        # Próximos vencimientos importantes
        st.markdown("#### 📅 Próximos Vencimientos Importantes")
        next_expirations = option_data.groupby('expiration')['GEX'].sum().abs().nlargest(5)
        for exp_date, gex in next_expirations.items():
            days_to_exp = (exp_date - datetime.now()).days
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"{exp_date.strftime('%d %b %Y')}")
            with col2:
                st.write(f"${gex/1e9:.3f}B")
            with col3:
                st.write(f"{days_to_exp} días")
    
    with tab4:
        fig = create_strike_distribution_plot(spot_price, option_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Resumen Calls vs Puts
        st.markdown("#### 🎯 Análisis Calls vs Puts")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **📈 CALLS**
            - GEX Total: ${call_gex:.3f}B
            - Strikes Activos: {len(option_data[option_data['type'] == 'C']['strike'].unique())}
            - OI Total: {option_data[option_data['type'] == 'C']['open_interest'].sum():,.0f}
            """)
        with col2:
            st.info(f"""
            **📉 PUTS**  
            - GEX Total: ${put_gex:.3f}B
            - Strikes Activos: {len(option_data[option_data['type'] == 'P']['strike'].unique())}
            - OI Total: {option_data[option_data['type'] == 'P']['open_interest'].sum():,.0f}
            """)
    
    with tab5:
        fig = create_cumulative_gex_plot(option_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis temporal
        st.markdown("#### ⏰ Análisis Temporal de GEX")
        
        # Calcular GEX por periodos
        data_temp = option_data.copy()
        data_temp['days_to_expiry'] = (data_temp['expiration'] - datetime.now()).dt.days
        
        periods = {
            "0-7 días": (0, 7),
            "7-30 días": (7, 30),
            "30-60 días": (30, 60),
            "60-90 días": (60, 90),
            "90+ días": (90, 999)
        }
        
        col1, col2, col3, col4, col5 = st.columns(5)
        cols = [col1, col2, col3, col4, col5]
        
        for i, (period, (min_days, max_days)) in enumerate(periods.items()):
            mask = (data_temp['days_to_expiry'] >= min_days) & (data_temp['days_to_expiry'] < max_days)
            period_gex = data_temp[mask]['GEX'].sum() / 1e9
            with cols[i]:
                st.metric(period, f"${period_gex:.2f}B")
    
    with tab6:
        st.markdown("#### 📋 Datos de Opciones Procesados")
        
        # Filtros para la tabla
        col1, col2, col3 = st.columns(3)
        with col1:
            type_filter = st.selectbox("Tipo", ["Todas", "Calls", "Puts"])
        with col2:
            sort_by = st.selectbox("Ordenar por", ["GEX", "open_interest", "volume", "strike"])
        with col3:
            n_rows = st.number_input("Filas a mostrar", min_value=10, max_value=100, value=20)
        
        # Aplicar filtros
        display_data = option_data.copy()
        if type_filter == "Calls":
            display_data = display_data[display_data['type'] == 'C']
        elif type_filter == "Puts":
            display_data = display_data[display_data['type'] == 'P']
        
        # Ordenar y mostrar
        display_data = display_data.sort_values(sort_by, ascending=False).head(n_rows)
        
        # Formatear columnas para display
        display_cols = ['option', 'type', 'strike', 'expiration', 'GEX', 'open_interest', 'volume', 'gamma', 'iv']
        display_data_formatted = display_data[display_cols].copy()
        display_data_formatted['GEX'] = display_data_formatted['GEX'].apply(lambda x: f"${x/1e6:.2f}M")
        display_data_formatted['expiration'] = display_data_formatted['expiration'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(
            display_data_formatted,
            use_container_width=True,
            height=400
        )
        
        # Botón de descarga
        csv = display_data.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Datos CSV",
            data=csv,
            file_name=f"{ticker}_gex_maxpain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tab7:
        # Estrategias de Trading
        st.markdown("### 📚 Estrategias de Trading con Max Pain + GEX")
        
        strategy_tab1, strategy_tab2, strategy_tab3 = st.tabs(["0DTE", "Semanal", "OPEX"])
        
        with strategy_tab1:
            st.markdown("""
            #### 🎯 Estrategia 0DTE (Same Day Expiry)
            
            **Configuración Ideal:**
            - Max Pain ≠ Spot (>0.5% diferencia)
            - GEX Positivo (movimiento lento)
            - Viernes o día de expiración
            
            **Ejecución:**
            1. **9:30-10:00 AM**: Identificar Max Pain y GEX
            2. **10:00 AM**: Entrar si hay divergencia
            3. **Entrada**:
               - Si Spot > Max Pain: Vender Call ATM + Comprar Put hacia Max Pain
               - Si Spot < Max Pain: Vender Put ATM + Comprar Call hacia Max Pain
            4. **Gestión**: 
               - Stop Loss: 2x premio recibido
               - Take Profit: 80% del máximo beneficio
            5. **Salida**: 3:00-3:30 PM o al alcanzar objetivo
            
            **Ejemplo Real:**
            - SPY @ $450, Max Pain @ $447, GEX Positivo
            - Acción: Vender $450 Call, Comprar $447 Put
            - Resultado típico: SPY cierra @ $447.50
            - Ganancia: 75-80% del crédito recibido
            """)
        
        with strategy_tab2:
            st.markdown("""
            #### 📅 Estrategia Semanal
            
            **Configuración Ideal:**
            - Miércoles: Identificar Max Pain del viernes
            - GEX > 0 para mayor probabilidad
            - Divergencia > 1% pero < 3%
            
            **Iron Condor en Max Pain:**
            1. **Entrada**: Miércoles o Jueves
            2. **Estructura**:
               - Short Strikes: Max Pain ± 0.5%
               - Long Strikes: Max Pain ± 2%
            3. **Ajustes**:
               - Si toca un lado: Rolar el lado no tocado
               - Si GEX cambia signo: Cerrar posición
            4. **Salida**: Viernes 3:30 PM
            
            **Gestión de Riesgo:**
            - Max Loss: 3x crédito recibido
            - Target: 50-70% del crédito
            - Win Rate esperado: 65-70%
            """)
        
        with strategy_tab3:
            st.markdown("""
            #### 📊 Estrategia OPEX (Monthly Expiration)
            
            **Por qué OPEX es especial:**
            - Mayor volumen de opciones expirando
            - Pin más fuerte debido a gamma masivo
            - Institucionales rebalanceando
            
            **Butterfly en Max Pain:**
            1. **Timing**: Lunes de la semana OPEX
            2. **Estructura**:
               - Centro: Max Pain exacto
               - Alas: ± 1-1.5% del Max Pain
            3. **Entrada escalada**:
               - 33% Lunes
               - 33% Miércoles  
               - 34% Viernes mañana
            4. **Exit**: 
               - Parcial: 50% en 2x ganancia
               - Final: 30 min antes del cierre
            
            **Tips Pro:**
            - Si IV > 20: Preferir Calendar Spreads
            - Si GEX negativo: Reducir tamaño 50%
            - Cerrar si Max Pain se mueve >2%
            """)

def show_educational_content():
    """Mostrar contenido educativo sobre GEX y Max Pain"""
    st.markdown("""
    <div class='info-box'>
    <h2>📚 ¿Qué es GEX y Max Pain?</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🎓 Gamma Exposure (GEX)
        
        **GEX** mide la exposición agregada de los market makers a los cambios en el precio debido a sus posiciones en opciones.
        
        **Interpretación:**
        - **GEX Positivo**: Mercado estable, volatilidad reducida
        - **GEX Negativo**: Mercado volátil, movimientos amplificados
        
        **Fórmula:**
        ```
        GEX = Γ × OI × S² × CS × 0.01
        ```
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Max Pain Theory
        
        **Max Pain** es el precio donde la mayoría de opciones expiran sin valor, minimizando el pago de los market makers.
        
        **Por qué funciona:**
        - Market makers controlan ~85% del volumen
        - Hedging dinámico mueve el precio
        - Efecto "imán" en días de expiración
        
        **Uso práctico:**
        - Pin más fuerte en 0DTE
        - Combinar con GEX para timing
        """)
    
    st.markdown("""
    <div class='info-box'>
    <h3>💡 La Combinación Perfecta: GEX + Max Pain</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Matriz de estrategias
    st.markdown("### 🎯 Matriz de Estrategias")
    
    strategies_data = {
        "Condición": ["Max Pain > Spot\nGEX > 0", "Max Pain > Spot\nGEX < 0", 
                      "Max Pain < Spot\nGEX > 0", "Max Pain < Spot\nGEX < 0"],
        "Expectativa": ["Subida lenta", "Subida rápida", 
                       "Bajada lenta", "Bajada rápida"],
        "Estrategia": ["Vender Puts OTM", "Comprar Calls ATM",
                      "Vender Calls OTM", "Comprar Puts ATM"],
        "Win Rate": ["75%", "60%", "75%", "60%"]
    }
    
    df_strategies = pd.DataFrame(strategies_data)
    st.table(df_strategies)
    
    # Footer
    st.markdown("""
    <div style='margin-top: 50px; padding: 20px; background: linear-gradient(135deg, rgba(0,217,255,0.1), rgba(254,83,187,0.1)); border-radius: 15px;'>
        <h4 style='text-align: center;'>🚀 Desarrollado con pasión para la comunidad de trading cuantitativo</h4>
        <p style='text-align: center;'>
            <a href='https://bquantfinance.com' style='color: #00D9FF;'>bquantfinance.com</a> | 
            <a href='https://twitter.com/Gsnchez' style='color: #FE53BB;'>@Gsnchez</a>
        </p>
        <p style='text-align: center; color: #888; font-size: 12px;'>
            Max Pain + GEX = El edge que necesitas para ganar consistentemente
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
