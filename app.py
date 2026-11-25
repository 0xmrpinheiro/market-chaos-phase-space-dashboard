"""Real-Time Market Chaos & Phase-Space Dashboard.

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
from data_feed import FeedConfig, MarketFeed, chaos_noise_reference
from utils import (
    DEFAULT_SYMBOL,
    DEFAULT_WINDOW,
    INITIAL_SAMPLE_TICKS,
    MAX_WINDOW,
    METRIC_WINDOW,
    REGIME_NAMES,
    MarketBuffer,
    format_metric,
    format_percent,
    format_price,
)

st.set_page_config(
    page_title="Market Chaos & Phase-Space Dashboard",
    page_icon="\U0001f4c9",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Market Chaos & Phase-Space Dashboard")
st.caption(
    "Explore a simulated market feed and compare deterministic chaos with noise. "
    "No live market data or API keys."
)

with st.sidebar:
    st.header("Controls")
    symbol = st.text_input("Symbol label", value=DEFAULT_SYMBOL, key="symbol")
    window = st.number_input(
        "Chart window (ticks)", min_value=100, max_value=MAX_WINDOW,
        value=DEFAULT_WINDOW, step=100, key="window",
    )
    running = st.toggle("Play simulation", value=True, key="running")
    st.divider()
    st.subheader("Market simulation")
    drift = st.number_input(
        "Drift (per tick)", min_value=-0.01, max_value=0.01,
        value=0.0002, step=0.0001, format="%.4f", key="drift",
    )
    vol_regime = st.selectbox(
        "Volatility setting", options=["low", "medium", "high"], index=1, key="volatility",
    )
    noise = st.slider(
        "Noise level", min_value=0.0, max_value=0.1, value=0.02, step=0.005, key="noise",
    )
    restart = st.button("Restart replay", width="stretch", type="primary", key="restart")

config = FeedConfig(symbol=symbol, drift=drift, vol_regime=vol_regime, noise_level=noise, seed=42)
if restart or "feed" not in st.session_state:
    feed = MarketFeed(config)
    buffer = MarketBuffer(window=int(window))
    now = pd.Timestamp.now("UTC").floor("s")
    for offset in range(INITIAL_SAMPLE_TICKS, 0, -1):
        buffer.append(feed.next_tick(now - pd.Timedelta(seconds=offset)))
    st.session_state["feed"] = feed
    st.session_state["market_buffer"] = buffer
    st.session_state["regime_history"] = deque(maxlen=200)

feed = st.session_state["feed"]
feed.config = config
buffer = st.session_state["market_buffer"]
buffer.resize(int(window))


@st.fragment(run_every=1.0 if running else None)
def _live():
    now = pd.Timestamp.now("UTC").floor("s")
    if running and now > buffer.ticks[-1].timestamp:
        buffer.append(feed.next_tick(now))
    df = buffer.to_dataframe()
    prices = df["price"].to_numpy(dtype=np.float64)
    returns = df["return"].to_numpy(dtype=np.float64)
    vols = df["volatility"].to_numpy(dtype=np.float64)
    n = len(df)
    current_price = float(prices[-1])
    current_return = float(returns[-1])
    metric_prices = prices[-METRIC_WINDOW:]
    metric_returns = returns[-METRIC_WINDOW:]
    lyap = lyapunov_exponent(metric_prices, delay=1, dim=3)
    hurst = hurst_exponent(metric_returns, min_lag=10)
    corr_dim = correlation_dimension(metric_prices, delay=1, dim=3)
    sampen = sample_entropy(metric_returns, dim=2, tolerance_frac=0.2)
    regime_key = classify_regime(lyap, hurst)
    regime_hist = st.session_state["regime_history"]
    last_tick = buffer.ticks[-1].timestamp
    if not regime_hist or last_tick > regime_hist[-1][0]:
        regime_hist.append((last_tick, regime_key))
    else:
        regime_hist[-1] = (last_tick, regime_key)
    col1, col2, col3, col4, col5 = st.columns([1.6, 1, 1, 1, 1])
    with col1:
        st.metric(label=f"{symbol or DEFAULT_SYMBOL} simulated price", value=format_price(current_price), delta=format_percent(current_return))
    with col2:
        st.metric(label="Lyapunov", value=format_metric(lyap, 3))
    with col3:
        st.metric(label="Hurst", value=format_metric(hurst, 3))
    with col4:
        st.metric(label="Corr. Dim.", value=format_metric(corr_dim, 2))
    with col5:
        st.metric(label="SampEn", value=format_metric(sampen, 2))
    st.caption(
        f"Heuristic label: {REGIME_NAMES[regime_key]} · {n}/{buffer.window} chart ticks · "
        f"{len(metric_prices)} metric ticks · Last tick {last_tick:%H:%M:%S} UTC"
    )
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=df["timestamp"], y=df["price"], mode="lines", name="Price", line=dict(color="#3b82f6", width=2)))
    price_fig.update_layout(title="Simulated price", xaxis_title="Time (UTC)", yaxis_title="Price", height=260, margin=dict(l=40, r=20, t=40, b=30), hovermode="x unified")
    if n > 2:
        phase_fig = go.Figure()
        phase_fig.add_trace(go.Scatter(x=prices[:-1], y=prices[1:], mode="markers", marker=dict(size=4, color=np.arange(n - 1), colorscale="Viridis", showscale=True, colorbar=dict(title="Time"), line=dict(width=0.3, color="#475569")), name="Delay embedding"))
        pmin, pmax = float(prices.min()), float(prices.max())
        phase_fig.add_trace(go.Scatter(x=[pmin, pmax], y=[pmin, pmax], mode="lines", line=dict(color="#ef4444", width=1, dash="dot"), name="price[t] = price[t-1]", showlegend=True))
        phase_fig.update_layout(title="Phase space: consecutive prices", xaxis_title="Price at t−1", yaxis_title="Price at t", height=260, margin=dict(l=40, r=20, t=40, b=30), showlegend=False)
        price_panel, phase_panel = st.columns(2)
        price_panel.plotly_chart(price_fig, width="stretch", key="price")
        phase_panel.plotly_chart(phase_fig, width="stretch", key="phase")
    rv_fig = go.Figure()
    rv_fig.add_trace(go.Scatter(x=vols, y=returns, mode="markers", marker=dict(size=5, color=returns, colorscale="RdBu", cmin=-float(np.abs(returns).max()) if len(returns) else -1, cmax=float(np.abs(returns).max()) if len(returns) else 1, showscale=True, colorbar=dict(title="Return"), line=dict(width=0.3, color="#475569")), name="Returns vs. vol"))
    rv_fig.add_hline(y=0, line_dash="dot", line_color="#64748b", line_width=1)
    rv_fig.update_layout(title="Returns vs. simulated volatility", xaxis_title="Volatility (cluster \u03c3)", yaxis_title="Log return", height=260, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(rv_fig, width="stretch", key="returns")
    if len(regime_hist) > 1:
        hist_df = pd.DataFrame(regime_hist, columns=["timestamp", "regime"])
        hist_df["label"] = hist_df["regime"].map(REGIME_NAMES)
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["label"], mode="lines+markers", line=dict(color="#2563eb", width=2), marker=dict(size=5)))
        hist_fig.update_yaxes(categoryorder="array", categoryarray=list(REGIME_NAMES.values()))
        hist_fig.update_layout(title="Heuristic label history", xaxis_title="Time (UTC)", height=200, margin=dict(l=40, r=20, t=40, b=20))
        st.plotly_chart(hist_fig, width="stretch", key="history")


market_tab, reference_tab = st.tabs(["Market simulation", "Chaos vs. noise"])
with market_tab:
    _live()
    with st.expander("How to read the metrics"):
        st.markdown("""
| Metric | What this dashboard estimates |
|--------|-------------------------------|
| **Lyapunov exponent** | Divergence of nearby paths in the price embedding. Noise can also give a positive estimate. |
| **Hurst exponent** | Persistence (above 0.5) or anti-persistence (below 0.5) of simulated returns. |
| **Correlation dimension** | How pairwise distances scale in the price embedding; sensitive to the embedding and sample. |
| **Sample entropy** | Irregularity in the simulated returns; higher values mean fewer patterns repeat within the selected tolerance. |

The labels use illustrative thresholds, not a validated classifier. Neither the metrics nor the phase-space plot diagnoses chaos in a financial market.
""")

with reference_tab:
    st.subheader("What does deterministic chaos look like?")
    st.markdown(
        "These are synthetic signals, not market prices. Plot each point against the next. "
        "The logistic map follows a curve even though the sequence is chaotic; "
        "independent noise has no such update rule and spreads across the plot."
    )
    chaotic, noise_reference = chaos_noise_reference()
    for panel, values, title, color, key in zip(
        st.columns(2), (chaotic, noise_reference),
        ("Logistic map (r = 4)", "Independent uniform noise"), ("#2563eb", "#b45309"),
        ("logistic_map", "uniform_noise"),
    ):
        fig = go.Figure(go.Scatter(
            x=values[:-1], y=values[1:], mode="markers", marker=dict(size=5, color=color),
        ))
        fig.update_layout(
            title=title, xaxis_title="Value at t", yaxis_title="Value at t+1", height=320,
            margin=dict(l=35, r=20, t=45, b=35),
        )
        fig.update_xaxes(range=[-0.02, 1.02])
        fig.update_yaxes(range=[-0.02, 1.02])
        panel.plotly_chart(fig, width="stretch", key=key)
