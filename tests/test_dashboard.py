from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from chaos_metrics import classify_regime
from data_feed import FeedConfig, MarketFeed, chaos_noise_reference
from utils import INITIAL_SAMPLE_TICKS, MarketBuffer


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class SimulationTests(TestCase):
    def test_reference_signals_are_repeatable_and_distinct(self):
        chaotic, noise = chaos_noise_reference(size=120)
        other_chaotic, other_noise = chaos_noise_reference(size=120)

        np.testing.assert_array_equal(chaotic, other_chaotic)
        np.testing.assert_array_equal(noise, other_noise)
        np.testing.assert_allclose(chaotic[1:], 4 * chaotic[:-1] * (1 - chaotic[:-1]))
        self.assertGreater(np.abs(noise[1:] - 4 * noise[:-1] * (1 - noise[:-1])).mean(), 0.2)

    def test_shrinking_buffer_keeps_latest_ticks(self):
        feed = MarketFeed(FeedConfig(seed=42))
        ticks = [feed.next_tick(time) for time in pd.date_range("2026-01-01", periods=4, freq="s")]
        buffer = MarketBuffer(window=4)
        for tick in ticks:
            buffer.append(tick)

        buffer.resize(2)
        self.assertEqual(list(buffer.ticks), ticks[-2:])
        buffer.resize(4)
        self.assertEqual(list(buffer.ticks), ticks[-2:])

    def test_high_divergence_is_not_labeled_chaos(self):
        self.assertEqual(classify_regime(None, None), "warming_up")
        self.assertEqual(classify_regime(0.1, 0.6), "high_divergence")
        self.assertEqual(classify_regime(0.0, 0.6), "trending")
        self.assertEqual(classify_regime(0.0, 0.4), "mean_reverting")
        self.assertEqual(classify_regime(0.0, 0.5), "no_signal")


class DashboardSmokeTests(TestCase):
    def test_first_run_renders_both_tabs_with_seeded_data(self):
        app = AppTest.from_file(str(APP_PATH)).run()

        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        self.assertEqual([tab.label for tab in app.tabs], ["Market simulation", "Chaos vs. noise"])
        self.assertGreaterEqual(len(app.session_state["market_buffer"]), INITIAL_SAMPLE_TICKS)
        self.assertGreaterEqual(len(app.get("plotly_chart")), 5)

    def test_controls_pause_resize_and_restart_the_replay(self):
        app = AppTest.from_file(str(APP_PATH)).run()
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        app.toggle(key="running").set_value(False).run()
        app.number_input(key="window").set_value(100).run()
        buffer = app.session_state["market_buffer"]
        self.assertEqual(len(buffer), 100)
        last_timestamp = buffer.ticks[-1].timestamp

        app.selectbox(key="volatility").set_value("high").run()
        self.assertEqual(app.session_state["market_buffer"].ticks[-1].timestamp, last_timestamp)
        self.assertEqual(app.session_state["feed"].config.vol_regime, "high")

        app.button(key="restart").click().run()
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        self.assertEqual(len(app.session_state["market_buffer"]), 100)
        self.assertEqual(len(app.session_state["regime_history"]), 1)
        feed = MarketFeed(FeedConfig(vol_regime="high", seed=42))
        for _ in range(INITIAL_SAMPLE_TICKS):
            tick = feed.next_tick(last_timestamp)
        self.assertAlmostEqual(app.session_state["market_buffer"].ticks[-1].price, tick.price)