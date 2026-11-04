"""Utility helpers for session state, formatting, and shared constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np
import pandas as pd
from collections import deque


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WINDOW = 2000
DEFAULT_REFRESH_MS = 1000
DEFAULT_SYMBOL = "SPX"

REGIME_NAMES = {
    "trending": "Trending",
    "mean_reverting": "Mean-Reverting",
    "chaotic": "Chaotic",
    "random": "Random",
}

REGIME_COLORS = {
    "trending": "#22c55e",        # green
    "mean_reverting": "#3b82f6",  # blue
    "chaotic": "#ef4444",         # red
    "random": "#f59e0b",          # amber
}


# ---------------------------------------------------------------------------
# Rolling buffer backed by a deque for O(1) append/pop
# ---------------------------------------------------------------------------


@dataclass
class Tick:
    timestamp: pd.Timestamp
    price: float
    return_: float
    velocity: float
    volatility: float


@dataclass
class MarketBuffer:
    window: int
    ticks: Deque[Tick] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.ticks = deque(maxlen=self.window)

    def append(self, tick: Tick) -> None:
        self.ticks.append(tick)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.ticks:
            return pd.DataFrame(
                columns=["timestamp", "price", "return", "velocity", "volatility"]
            )
        return pd.DataFrame(
            [
                {
                    "timestamp": t.timestamp,
                    "price": t.price,
                    "return": t.return_,
                    "velocity": t.velocity,
                    "volatility": t.volatility,
                }
                for t in self.ticks
            ]
        )

    def __len__(self) -> int:
        return len(self.ticks)

    def clear(self) -> None:
        self.ticks.clear()


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def ensure_state(
    symbol: str = DEFAULT_SYMBOL,
    window: int = DEFAULT_WINDOW,
    refresh_ms: int = DEFAULT_REFRESH_MS,
    running: bool = True,
) -> None:
    """Initialise Streamlit session-state keys if absent."""
    import streamlit as st  # imported here so utils stays importable in tests

    defaults: Dict[str, object] = {
        "market_buffer": MarketBuffer(window=window),
        "symbol": symbol,
        "window": window,
        "refresh_ms": refresh_ms,
        "running": running,
        "drift": 0.0002,
        "vol_regime": "medium",  # low | medium | high
        "noise_level": 0.02,
        "last_price": 100.0,
        "regime_history:": deque(maxlen=200),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_buffer() -> MarketBuffer:
    import streamlit as st

    return st.session_state["market_buffer"]


def format_price(value: float) -> str:
    return f"{value:,.2f}"


def format_percent(value: float) -> str:
    return f"{value * 100:+.2f}%"


def format_metric(value: Optional[float], ndigits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    return f"{value:.{ndigits}f}"


def regime_badge_html(regime_key: str) -> str:
    name = REGIME_NAMES.get(regime_key, regime_key)
    color = REGIME_COLORS.get(regime_key, "#64748b")
    return (
        f"<span style='display:inline-block;padding:4px 10px;border-radius:12px;"
        f"background:{color};color:#fff;font-weight:700;font-size:0.9rem;' "
        f"title='Current market regime'>{name}</span>"
    )