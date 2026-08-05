"""Small set of causal indicators. Nothing here may read a future bar."""

from __future__ import annotations

import numpy as np


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    return np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )


def atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Wilder's ATR.

    The first ``period`` values are seeded with an expanding mean of true range
    so early bars carry a usable (if noisy) volatility estimate rather than NaN.
    Callers should still skip the warm-up window before trading.
    """
    tr = true_range(high, low, close)
    n = tr.size
    out = np.empty(n)
    if n == 0:
        return out

    running = 0.0
    for i in range(min(period, n)):
        running += tr[i]
        out[i] = running / (i + 1)
    alpha = 1.0 / period
    for i in range(period, n):
        out[i] = out[i - 1] + alpha * (tr[i] - out[i - 1])
    return out


def bodies(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    return np.abs(close - open_)


def body_ratio(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Body as a fraction of total range. High = conviction, low = indecision."""
    rng = high - low
    return np.divide(
        np.abs(close - open_), rng, out=np.zeros_like(rng), where=rng > 0
    )
