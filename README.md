# 📉 Real-Time Market Chaos & Phase-Space Dashboard

A live, offline-first dashboard that streams simulated market ticks and computes
chaos-theory metrics in a rolling window: **Lyapunov exponent**, **Hurst exponent**,
**correlation dimension**, and **sample entropy**. It visualizes price action,
phase-space trajectories (price vs. velocity), volatility clustering, and regime
history — no API keys or network access required.

## Features

| Panel | What it shows |
|-------|---------------|
| **Live price** | Streaming price series with per-tick return delta |
| **Phase space** | Price vs. velocity scatter — reveals attractors / strange-attractor structure |
| **Returns vs. volatility** | Volatility clustering (large returns follow large returns) |
| **Chaos metrics** | Lyapunov, Hurst, correlation dimension, sample entropy as gauges |
| **Regime** | Live classification: Trending / Mean-Reverting / Chaotic / Random |
| **Regime history** | Mini time-line of regime transitions |

## Chaos metrics

| Metric | Interpretation |
|--------|----------------|
| **Lyapunov exponent** | Rate of divergence of nearby trajectories. **> 0 = chaotic** (sensitive dependence). ≈ 0 = random walk; < 0 = convergent/stable. |
| **Hurst exponent** | **> 0.55 trending**, **< 0.45 mean-reverting**, ≈ 0.5 random walk. |
| **Correlation dimension** | Dimensionality of the attractor; low non-integer values suggest deterministic chaos. |
| **Sample entropy** | Signal irregularity — higher = more complex/unpredictable. |

## Quick start

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Controls (sidebar)

- **Symbol** — label shown in the header (cosmetic).
- **Rolling window** — number of ticks kept in the buffer (default 2000).
- **Refresh rate** — how often a new tick is emitted (ms).
- **Live feed** — pause/resume the simulation.
- **Drift** — per-tick expected return of the simulated price path.
- **Volatility regime** — low / medium / high base volatility.
- **Noise level** — extra observation noise added to each tick.
- **Reset buffer** — clear the history and restart at 100.

## Swapping in real market data

The simulator lives in `data_feed.py` (`MarketFeed.next_tick` returns a `Tick`).
Replace that class with a WebSocket/REST client that yields the same `Tick`
(timestamp, price, return, velocity, volatility) — the UI and chaos metrics
require no changes.

## Project structure

```
app.py              # Streamlit UI + orchestration
data_feed.py        # Simulated real-time market data generator
chaos_metrics.py    # Chaos-theory calculations
utils.py            # Session-state helpers, formatting, constants
requirements.txt    # Python dependencies
README.md           # This file
```

## Requirements

- Python 3.10+
- streamlit, plotly, pandas, numpy, streamlit-autorefresh

## License

MIT