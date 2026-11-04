#!/usr/bin/env python3
import os

part1 = '''"""Real-Time Market Chaos & Phase-Space Dashboard.

Run with:
    streamlit run app.py

Requires:
    streamlit, plotly, pandas, numpy
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chaos_metrics import (
    classify_regime,
    correlation_dimension,
    hurst_exponent,
    lyapunov_exponent,
    sample_entropy,
)
from data_feed import FeedConfig, MarketFeed
from utils import (
    DEFAULT_SYMBOL,
    DEFAULT_WINDOW,
    MarketBuffer,
    ensure_state,
    format_metric,
    format_percent,
    format_price,
    get_buffer,
    regime_badge_html,
)

st.set_page_config(
    page_title="Market Chaos & Phase-Space Dashboard",
    page_icon="\U0001f4c9",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {max-width: 100%;}
    .caption-small {font-size:0.8rem;color:#64748b;margin-top:-8px;margin-bottom:16px;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("\u2699\ufe0f Controls")
    symbol = st.text_input("Symbol", value=st.session_state.get("symbol", DEFAULT_SYMBOL))
    st.session_state["symbol"] = symbol
    window = st.number_input("Rolling window (ticks)", min_value=100, max_value=20000, value=int(st.session_state.get("window", DEFAULT_WINDOW)), step=100)
    st.session_state["window"] = int(window)
    running = st.toggle("Live feed", value=bool(st.session_state.get("running", True)))
    st.session_state["running"] = running
    st.divider()
    st.subheader("Market simulation")
    drift = st.number_input("Drift (per tick)", min_value=-0.01, max_value=0.01, value=float(st.session_state.get("drift", 0.0002)), step=0.0001, format="%.4f")
    st.session_state["drift"] = drift
    vol_regime = st.selectbox("Volatility regime", options=["low", "medium", "high"], index=["low", "medium", "high"].index(st.session_state.get("vol_regime", "medium")))
    st.session_state["vol_regime"] = vol_regime
    noise = st.slider("Noise level", min_value=0.0, max_value=0.1, value=float(st.session_state.get("noise_level", 0.02)), step=0.005)
    st.session_state["noise_level"] = noise
    st.divider()
    if st.button("Reset buffer", use_container_width=True, type="primary"):
        st.session_state["market_buffer"] = MarketBuffer(window=int(window))
        st.session_state["last_price"] = 100.0
        st.session_state["regime_history"] = deque(maxlen=200)
        if "feed" in st.session_state:
            del st.session_state["feed"]
        st.rerun()

ensure_state(
    symbol=st.session_state.get("symbol", DEFAULT_SYMBOL),
    window=int(st.session_state.get("window", DEFAULT_WINDOW)),
    running=bool(st.session_state.get("running", True)),
)

if "feed" not in st.session_state:
    st.session_state["feed"] = MarketFeed(
        FeedConfig(
            symbol=st.session_state["symbol"],
            drift=st.session_state["drift"],
            vol_regime=st.session_state["vol_regime"],
            noise_level=st.session_state["noise_level"],
            seed=42,
        )
    )
@st.fragment(run_every=1.0)
def _live():
    now = pd.Timestamp.now("UTC")
    if st.session_state.get("running", True):
        last_price = float(st.session_state.get("last_price", 100.0))
        feed.price = last_price
        tick = feed.next_tick(now)
        st.session_state["last_price"] = tick.price
        buffer.append(tick)
    df = buffer.to_dataframe()
    if df.empty:
        st.info("Starting feed... wait for the first tick.")
        return
    prices = df["price"].to_numpy(dtype=np.float64)
    returns = df["return"].to_numpy(dtype=np.float64)
    vols = df["volatility"].to_numpy(dtype=np.float64)
    n = len(df)
    current_price = float(prices[-1])
    current_return = float(returns[-1]) if n >= 1 else 0.0
    lyap = lyapunov_exponent(prices, delay=1, dim=3)
    hurst = hurst_exponent(returns, min_lag=10, max_lag=max(10, n // 8))
    corr_dim = correlation_dimension(prices, delay=1, dim=3)
    sampen = sample_entropy(returns, dim=2, tolerance_frac=0.2)
    regime_key = classify_regime(lyap, hurst, corr_dim, sampen)
    regime_hist: deque = st.session_state.get("regime_history", deque(maxlen=200))
    regime_hist.append((now, regime_key))
    st.session_state["regime_history"] = regime_hist
    col1, col2, col3, col4, col5 = st.columns([1.6, 1, 1, 1, 1])
    with col1:
        st.metric(label=f"{st.session_state.get('symbol', DEFAULT_SYMBOL)} Price", value=format_price(current_price), delta=format_percent(current_return))
    with col2:
        st.metric(label="Lyapunov", value=format_metric(lyap, 3))
    with col3:
        st.metric(label="Hurst", value=format_metric(hurst, 3))
    with col4:
        st.metric(label="Corr. Dim.", value=format_metric(corr_dim, 2))
    with col5:
        st.metric(label="SampEn", value=format_metric(sampen, 2))
    st.markdown(f"**Regime:** {regime_badge_html(regime_key)} | Ticks: **{n}** / {int(window)} | {now.strftime('%H:%M:%S')} UTC", unsafe_allow_html=True)
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=df["timestamp"], y=df["price"], mode="lines", name="Price", line=dict(color="#3b82f6", width=2)))
    price_fig.update_layout(title="Live Price", xaxis_title="Time", yaxis_title="Price", height=280, margin=dict(l=40, r=20, t=40, b=30), hovermode="x unified")
    st.plotly_chart(price_fig, use_container_width=True)
'''
with open('app.py', 'a') as f:
    f.write(part1) if False else print('Part 2 handler')

# Actually append to app.py
with open('app.py', 'a') as f:
    f.write(part1)  # This won't work right - need to think

print('Part 2: fragment function')
feed = st.session_state["feed"]
buffer = get_buffer()
'''
with open('app.py', 'w') as f:
    f.write(part1)
print('Part 1 written: header through state init')