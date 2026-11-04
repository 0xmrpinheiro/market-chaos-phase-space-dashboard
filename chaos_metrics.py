"""Chaos-theory metrics computed on a rolling market window.

Implements:
    * Largest Lyapunov exponent (nearest-neighbor divergence in delay-embedded
      phase space)
    * Hurst exponent (rescaled-range analysis)
    * Correlation dimension (Grassberger-Procaccia)
    * Sample entropy (signal irregularity / complexity)

All functions gracefully return ``None`` when the input is too short or
degenerate, so the dashboard never crashes on early ticks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ========================================================================= #
# Helpers
# ========================================================================= #


def _delay_embed(series: np.ndarray, delay: int = 1, dim: int = 3) -> np.ndarray:
    """Time-delay embedding of a 1-D series.

    Returns an array of shape (n_embedded, dim) where each row is
    [x[t], x[t+delay], ..., x[t+(dim-1)*delay]].
    """
    n = len(series)
    if n < delay * (dim - 1) + 1:
        return np.empty((0, dim))
    n_emb = n - delay * (dim - 1)
    embedded = np.column_stack(
        [series[i : i + n_emb] for i in range(0, delay * dim, delay)]
    )
    return embedded


def _require(series: np.ndarray, min_len: int) -> Optional[np.ndarray]:
    """Return the series as float64 if long enough and non-constant, else None."""
    if series is None or len(series) < min_len:
        return None
    x = np.asarray(series, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if len(x) < min_len or np.ptp(x) == 0:
        return None
    return x


# ========================================================================= #
# Largest Lyapunov exponent
# ========================================================================= #


def lyapunov_exponent(
    series: np.ndarray,
    delay: int = 1,
    dim: int = 3,
    max_t: Optional[int] = None,
) -> Optional[float]:
    """Estimate the largest Lyapunov exponent using nearest-neighbor divergence.

    A positive value indicates sensitive dependence on initial conditions
    (chaos); a value near zero indicates a random walk; a negative value
    indicates convergence to a fixed point or limit cycle.

    Parameters
    ----------
    series : 1-D array-like
        Input time series (e.g. price or returns).
    delay : int
        Time-delay for embedding.
    dim : int
        Embedding dimension.
    max_t : int | None
        Maximum number of divergence steps to average over.  Defaults to
        ``len // 4``.
    """
    x = _require(series, delay * (dim - 1) + 10)
    if x is None:
        return None

    embedded = _delay_embed(x, delay=delay, dim=dim)
    n = len(embedded)
    if n < 10:
        return None

    # Use a KD-tree-ish nearest neighbor via broadcasting (fine for n <= few
    # thousand).  Exclude temporally-close points (Theiler window) to avoid
    # serial correlation.
    theiler = max(dim, delay)
    diffs = embedded[:, None, :] - embedded[None, :, :]  # (n, n, dim)
    dist = np.linalg.norm(diffs, axis=2)  # (n, n)
    np.fill_diagonal(dist, np.inf)

    # Mask near-neighbors in time.
    for i in range(n):
        lo = max(0, i - theiler)
        hi = min(n, i + theiler + 1)
        dist[i, lo:hi] = np.inf

    nearest = np.argmin(dist, axis=1)
    nearest_dist = dist[np.arange(n), nearest]

    # Only keep pairs that start with a meaningful separation.
    valid = np.isfinite(nearest_dist) & (nearest_dist > 1e-12)
    if valid.sum() < 5:
        return None

    if max_t is None:
        max_t = max(1, n // 4)

    # Divergence of each pair over time.
    div_rates = []
    for i in np.where(valid)[0]:
        j = int(nearest[i])
        d0 = nearest_dist[i]
        # Follow both trajectories forward, measuring separation at each step.
        steps = min(max_t, n - 1 - i, n - 1 - j)
        if steps < 2:
            continue
        traj_i = embedded[i : i + steps + 1]
        traj_j = embedded[j : j + steps + 1]
        seps = np.linalg.norm(traj_i - traj_j, axis=1)
        # Avoid log(0).
        seps = np.clip(seps, 1e-12, None)
        # Average divergence rate over the trajectory. seps[0] == d0, so start
        # at index 1 to align with step numbers 1..steps.
        rates = np.log(seps[1:] / d0) / np.arange(1, steps + 1)
        div_rates.append(np.mean(rates))

    if not div_rates:
        return None

    lam = float(np.mean(div_rates))
    return lam if np.isfinite(lam) else None


# ========================================================================= #
# Hurst exponent (rescaled-range analysis)
# ========================================================================= #


def hurst_exponent(series: np.ndarray, min_lag: int = 10, max_lag: int | None = None) -> Optional[float]:
    """Estimate the Hurst exponent via classical rescaled-range (R/S) analysis.

    H < 0.5 : mean-reverting
    H ≈ 0.5 : random walk / Brownian
    H > 0.5 : trending / persistent
    """
    x = _require(series, 2 * min_lag + 1)
    if x is None:
        return None

    n = len(x)
    if max_lag is None:
        max_lag = max(min_lag + 1, n // 4)
    max_lag = min(max_lag, n // 2)
    if max_lag <= min_lag:
        return None

    lags = np.arange(min_lag, max_lag + 1)
    rs_values = []

    for lag in lags:
        # Split into non-overlapping blocks of size `lag`.
        n_blocks = n // lag
        if n_blocks < 2:
            continue
        rs_block = []
        for b in range(n_blocks):
            block = x[b * lag : (b + 1) * lag]
            mean = block.mean()
            deviations = block - mean
            cumdev = np.cumsum(deviations)
            r = cumdev.max() - cumdev.min()
            s = block.std(ddof=1)
            if s > 1e-12:
                rs_block.append(r / s)
        if rs_block:
            rs_values.append(float(np.mean(rs_block)))

    if len(rs_values) < 3:
        return None

    rs_values = np.array(rs_values)
    lags_used = lags[: len(rs_values)]

    # Regress log(R/S) on log(lag); slope = H.
    l1 = np.log(lags_used)
    l2 = np.log(rs_values)
    A = np.vstack([l1, np.ones_like(l1)]).T
    try:
        slope, _ = np.linalg.lstsq(A, l2, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    h = float(slope)
    return h if np.isfinite(h) else None


# ========================================================================= #
# Correlation dimension (Grassberger-Procaccia)
# ========================================================================= #


def correlation_dimension(
    series: np.ndarray,
    delay: int = 1,
    dim: int = 3,
    n_r: int = 20,
    r_min_frac: float = 0.01,
    r_max_frac: float = 0.5,
) -> Optional[float]:
    """Estimate correlation dimension via the Grassberger-Procaccia algorithm.

    C(r) = (2 / (N(N-1))) * sum_{i<j} I(||x_i - x_j|| < r)
    The correlation dimension D2 is the slope of log C(r) vs log r in the
    scaling region.
    """
    x = _require(series, delay * (dim - 1) + 20)
    if x is None:
        return None

    embedded = _delay_embed(x, delay=delay, dim=dim)
    n = len(embedded)
    if n < 10:
        return None

    # Pairwise distances (upper triangle).
    diffs = embedded[:, None, :] - embedded[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]
    pair_dists = pair_dists[pair_dists > 0]

    if len(pair_dists) < 50:
        return None

    dmax = float(pair_dists.max())
    r_min = dmax * r_min_frac
    r_max = dmax * r_max_frac
    if r_max <= r_min:
        return None

    rs = np.geomspace(r_min, r_max, n_r)
    c_vals = []
    for r in rs:
        c = float(np.mean(pair_dists < r))  # fraction of pairs within r
        c_vals.append(c if c > 0 else np.nan)
    c_vals = np.array(c_vals)

    valid = np.isfinite(c_vals) & (c_vals > 0)
    if valid.sum() < 4:
        return None

    log_r = np.log(rs[valid])
    log_c = np.log(c_vals[valid])

    # Use the central, most linear portion for the slope (exclude extremes).
    lo = len(log_r) // 4
    hi = len(log_r) - len(log_r) // 4
    if hi - lo < 3:
        lo, hi = 0, len(log_r)

    A = np.vstack([log_r[lo:hi], np.ones(hi - lo)]).T
    try:
        slope, _ = np.linalg.lstsq(A, log_c[lo:hi], rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    d2 = float(slope)
    return d2 if np.isfinite(d2) else None


# ========================================================================= #
# Sample entropy
# ========================================================================= #


def sample_entropy(
    series: np.ndarray, dim: int = 2, tolerance_frac: float = 0.2
) -> Optional[float]:
    """Compute Sample Entropy (SampEn) of a 1-D series.

    SampEn quantifies the -log conditional probability that subsequences of
    length `dim` that match within tolerance also match at length `dim+1`.
    Higher values = more irregular / complex; lower = more regular.
    """
    x = _require(series, dim + 2)
    if x is None:
        return None

    n = len(x)
    tol = tolerance_frac * np.std(x)
    if tol <= 0:
        return None

    def _count_matches(m: int) -> int:
        # Build embedded vectors of length m.
        emb = np.array([x[i : i + m] for i in range(n - m + 1)])
        count = 0
        for i in range(len(emb)):
            for j in range(i + 1, len(emb)):
                if np.max(np.abs(emb[i] - emb[j])) <= tol:
                    count += 1
        return count

    try:
        a = _count_matches(dim)
        b = _count_matches(dim + 1)
    except MemoryError:
        return None

    if b == 0 or a == 0:
        return None

    return float(-np.log(b / a))


# ========================================================================= #
# Regime classifier
# ========================================================================= #


def classify_regime(
    lyapunov: Optional[float],
    hurst: Optional[float],
    correlation_dim: Optional[float],
    entropy: Optional[float],
) -> str:
    """Classify the market regime from chaos metrics.

    Priority:
        1. Chaotic     : Lyapunov > 0 (strong sensitive dependence)
        2. Trending    : H > 0.55
        3. MeanRev     : H < 0.45
        4. Random      : otherwise
    """
    if lyapunov is not None and np.isfinite(lyapunov) and lyapunov > 0.05:
        return "chaotic"
    if hurst is not None and np.isfinite(hurst):
        if hurst > 0.55:
            return "trending"
        if hurst < 0.45:
            return "mean_reverting"
    return "random"