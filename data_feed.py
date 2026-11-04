"""Simulated real-time market data generator.

Produces a stream of ticks using geometric Brownian motion with regime
switching and volatility clustering.  Entirely offline -- no API keys or
network access required.  Swap this module for a real WebSocket/REST feed
without changing the UI layer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Deque

import numpy as np
import pandas as pd

from utils import Tick


@dataclass
class FeedConfig:
    symbol: str = "SPX"
    drift: float = 0.0002          # per-tick expected return
    vol_regime: str = "medium"     # low | medium | high
    noise_level: float = 0.02      # additional observation noise
    seed: int | None = None


class MarketFeed:
    """Generates a single random-walk price path with regime-dependent vol."""

    VOL_MULTIPLIERS = {"low": 0.5, "medium": 1.0, "high": 2.0}

    def __init__(self, config: FeedConfig | None = None) -> None:
        self.config = config or FeedConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.price = 100.0
        self.last_return = 0.0
        # Volatility clustering: current vol decays toward baseline but is
        # kicked upward by recent absolute returns (GARCH-like).
        self.cluster_vol = 0.01
        self.tick_index = 0

    # ------------------------------------------------------------------ #
    def reset(self, price: float = 100.0) -> None:
        self.price = price
        self.last_return = 0.0
        self.cluster_vol = 0.01
        self.tick_index = 0
        self.rng = np.random.default_rng(self.config.seed)

    # ------------------------------------------------------------------ #
    def next_tick(self, timestamp: pd.Timestamp) -> Tick:
        """Advance the simulation by one tick and return the new Tick."""
        self.tick_index += 1

        # Regime-dependent base volatility.
        base_vol = 0.01 * self.VOL_MULTIPLIERS.get(self.config.vol_regime, 1.0)

        # GARCH-like clustering: vol reacts to squared recent return.
        shock = self.rng.standard_normal()
        innovation = self.cluster_vol * shock
        ret = self.config.drift + innovation + self.config.noise_level * self.rng.standard_normal()

        # Update clustering state (mean-reverting toward base_vol, boosted by |ret|).
        self.cluster_vol = 0.95 * self.cluster_vol + 0.05 * base_vol + 0.3 * abs(ret)
        self.cluster_vol = float(np.clip(self.cluster_vol, 0.001, 0.05))

        # Update price (geometric Brownian step).
        self.price = float(self.price * np.exp(ret))
        self.last_return = ret

        # Rolling volatility estimate from recent returns is computed upstream;
        # here we store the instantaneous cluster vol.
        return Tick(
            timestamp=timestamp,
            price=self.price,
            return_=ret,
            velocity=ret,  # first difference of price proxy
            volatility=self.cluster_vol,
        )