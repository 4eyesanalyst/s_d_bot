"""Market structure: swing points, trend, and the premium/discount curve.

A supply/demand zone is only half a setup. The other half is *where* it sits:
demand in the discount half of an uptrend is a different trade from demand in
the premium half of a downtrend, even when the zone itself looks identical.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from .data import Bars

UP = 1
DOWN = -1
RANGE = 0

_TREND_NAME = {UP: "up", DOWN: "down", RANGE: "range"}


def trend_name(value: int) -> str:
    return _TREND_NAME[int(value)]


@dataclass(frozen=True)
class Swing:
    index: int          # bar the swing printed on
    price: float
    is_high: bool
    confirmed: int      # bar at which it could first be known


def find_swings(bars: Bars, lookback: int = 2) -> list[Swing]:
    """Williams-style fractals.

    A swing high needs ``lookback`` bars on each side with a lower high. The
    right-hand side is why ``confirmed`` sits ``lookback`` bars later: in live
    trading you cannot know about the swing until those bars have printed.
    """
    n = len(bars)
    out: list[Swing] = []
    if n < 2 * lookback + 1:
        return out

    high, low = bars.high, bars.low
    for i in range(lookback, n - lookback):
        left = slice(i - lookback, i)
        right = slice(i + 1, i + lookback + 1)
        if high[i] >= high[left].max() and high[i] > high[right].max():
            out.append(Swing(i, float(high[i]), True, i + lookback))
        # A single bar can be both, on an outside-bar reversal.
        if low[i] <= low[left].min() and low[i] < low[right].min():
            out.append(Swing(i, float(low[i]), False, i + lookback))

    out.sort(key=lambda s: (s.confirmed, s.index))
    return out


class Structure:
    """Precomputed, causal structure read for every bar of a series."""

    def __init__(self, bars: Bars, lookback: int = 2, fallback_window: int = 100):
        self.bars = bars
        self.lookback = lookback
        self.swings = find_swings(bars, lookback)

        self._high_idx = [s.index for s in self.swings if s.is_high]
        self._high_conf = [s.confirmed for s in self.swings if s.is_high]
        self._high_price = [s.price for s in self.swings if s.is_high]
        self._low_idx = [s.index for s in self.swings if not s.is_high]
        self._low_conf = [s.confirmed for s in self.swings if not s.is_high]
        self._low_price = [s.price for s in self.swings if not s.is_high]

        n = len(bars)
        self.trend = np.zeros(n, dtype=np.int8)
        self.range_high = np.full(n, np.nan)
        self.range_low = np.full(n, np.nan)
        self.equilibrium = np.full(n, np.nan)

        highs: list[float] = []
        lows: list[float] = []
        ptr = 0
        for i in range(n):
            while ptr < len(self.swings) and self.swings[ptr].confirmed <= i:
                s = self.swings[ptr]
                (highs if s.is_high else lows).append(s.price)
                ptr += 1

            if len(highs) >= 2 and len(lows) >= 2:
                hh, hl = highs[-1] > highs[-2], lows[-1] > lows[-2]
                lh, ll = highs[-1] < highs[-2], lows[-1] < lows[-2]
                if hh and hl:
                    self.trend[i] = UP
                elif lh and ll:
                    self.trend[i] = DOWN

            hi = highs[-1] if highs else np.nan
            lo = lows[-1] if lows else np.nan
            if not (np.isfinite(hi) and np.isfinite(lo)) or hi <= lo:
                # No clean swing range yet (or an inverted one straight after a
                # structure break) -- fall back to a rolling window.
                start = max(0, i - fallback_window + 1)
                hi = float(bars.high[start : i + 1].max())
                lo = float(bars.low[start : i + 1].min())
            self.range_high[i] = hi
            self.range_low[i] = lo
            self.equilibrium[i] = (hi + lo) / 2.0

    # -- queries used by zone detection and scoring ---------------------------

    def swing_high_before(self, index: int) -> float | None:
        """Price of the newest swing high already *confirmed* by ``index``."""
        pos = bisect.bisect_right(self._high_conf, index) - 1
        return self._high_price[pos] if pos >= 0 else None

    def swing_low_before(self, index: int) -> float | None:
        pos = bisect.bisect_right(self._low_conf, index) - 1
        return self._low_price[pos] if pos >= 0 else None

    def recent_swing_high_price(self, index: int) -> float | None:
        """Newest swing high that *printed* at or before ``index`` (for trails)."""
        pos = bisect.bisect_right(self._high_idx, index) - 1
        return self._high_price[pos] if pos >= 0 else None

    def recent_swing_low_price(self, index: int) -> float | None:
        pos = bisect.bisect_right(self._low_idx, index) - 1
        return self._low_price[pos] if pos >= 0 else None

    def curve_position(self, index: int, price: float) -> float:
        """Where ``price`` sits in the dealing range: 0.0 = low, 1.0 = high.

        Below 0.5 is discount (where you want to be buying), above is premium.
        """
        lo, hi = self.range_low[index], self.range_high[index]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.5
        return float((price - lo) / (hi - lo))
