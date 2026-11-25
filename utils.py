"""Rolling market buffer, formatting helpers, and shared constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np
import pandas as pd
from collections import deque


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WINDOW = 500
MAX_WINDOW = 2000
METRIC_WINDOW = 256
INITIAL_SAMPLE_TICKS = 120
DEFAULT_SYMBOL = "SPX"

REGIME_NAMES = {
    "warming_up": "Not enough data",
    "no_signal": "No strong signal",
    "mean_reverting": "Anti-persistent returns",
    "trending": "Persistent returns",
    "high_divergence": "High estimated divergence",
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
        self.ticks = deque(self.ticks, maxlen=self.window)

    def resize(self, window: int) -> None:
        if window != self.window:
            self.ticks = deque(self.ticks, maxlen=window)
            self.window = window

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
# Formatting helpers
# ---------------------------------------------------------------------------


def format_price(value: float) -> str:
    return f"{value:,.2f}"


def format_percent(value: float) -> str:
    return f"{value * 100:+.2f}%"


def format_metric(value: Optional[float], ndigits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    return f"{value:.{ndigits}f}"