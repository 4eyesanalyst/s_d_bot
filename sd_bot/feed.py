"""Real-time market data.

Two backends, same interface:

* **MetaTrader 5** -- your broker's own prices, including live spread. Preferred
  when a terminal is running, because the signal is then priced against the book
  you would actually trade.
* **Dukascopy** -- a bank feed, ~30-60s behind, no terminal or account needed.

Both return *closed* bars only. A forming bar is discarded: acting on it means
acting on a candle whose high, low and close can all still change, which is the
single easiest way to make a live bot behave nothing like its backtest.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from . import sources
from . import timeframes as tfmod
from .data import Bars

logging.getLogger("DUKASCRIPT").setLevel(logging.ERROR)


class Feed:
    """Rolling per-symbol bar history, topped up from a live source."""

    def __init__(self, backend: str = "auto", history_dir: str = "data"):
        self.backend = self._resolve(backend)
        self.history_dir = history_dir
        self._bars: dict[tuple[str, str], Bars] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _resolve(backend: str) -> str:
        if backend != "auto":
            return backend
        try:
            from . import broker

            broker.ensure_initialized()
            return "mt5"
        except Exception:
            return "dukascopy"

    # -- prices ---------------------------------------------------------------

    def tick(self, symbol: str) -> tuple[float, float] | None:
        """Current ``(bid, ask)``, or None when unavailable."""
        if self.backend == "mt5":
            import MetaTrader5 as mt5

            t = mt5.symbol_info_tick(symbol)
            return (t.bid, t.ask) if t else None

        bars = self.bars(symbol, "M15", 2)
        if bars is None or len(bars) == 0:
            return None
        # Dukascopy bars are bid; synthesise the ask from the typical spread.
        bid = float(bars.close[-1])
        return bid, bid + sources.spread_points(symbol) * sources.spec_for(symbol).point

    def spread_points(self, symbol: str) -> float:
        if self.backend == "mt5":
            from . import broker

            return broker.current_spread_points(symbol)
        return sources.spread_points(symbol)

    # -- bars -----------------------------------------------------------------

    def bars(self, symbol: str, timeframe: str, count: int) -> Bars | None:
        """Latest ``count`` **closed** bars, refreshed from the live source."""
        with self._lock:
            key = (symbol, timeframe)
            try:
                self._refresh(key, count)
            except Exception:
                pass  # keep serving what we already have; caller sees stale data
            bars = self._bars.get(key)
            if bars is None or len(bars) == 0:
                return None
            return bars.slice(max(0, len(bars) - count), None)

    def _refresh(self, key: tuple[str, str], count: int) -> None:
        symbol, timeframe = key
        existing = self._bars.get(key)

        if existing is None:
            self._bars[key] = self._seed(symbol, timeframe, count)
            return

        # Only reach out when the next bar could plausibly have closed.
        step = tfmod.seconds(timeframe)
        now = datetime.now(timezone.utc).timestamp()
        if now < int(existing.time[-1]) + 2 * step:
            return

        fresh = self._download(
            symbol, timeframe,
            datetime.fromtimestamp(int(existing.time[-1]), tz=timezone.utc),
            datetime.now(timezone.utc),
        )
        if fresh is not None and len(fresh):
            self._bars[key] = _merge(existing, fresh, count * 3)

    def _seed(self, symbol: str, timeframe: str, count: int) -> Bars:
        """Start from the cached CSV where possible, then top up."""
        step = tfmod.seconds(timeframe)
        need = datetime.now(timezone.utc) - timedelta(seconds=step * count * 2)

        cached: Bars | None = None
        try:
            from pathlib import Path

            from .data import cache_path

            path = Path(cache_path(self.history_dir, symbol, timeframe))
            if path.exists():
                cached = Bars.from_csv(path, symbol, timeframe)
        except Exception:
            cached = None

        start = need
        if cached is not None and len(cached):
            start = datetime.fromtimestamp(int(cached.time[-1]), tz=timezone.utc)

        fresh = self._download(symbol, timeframe, start, datetime.now(timezone.utc))
        if cached is None:
            if fresh is None:
                raise RuntimeError(f"no data available for {symbol} {timeframe}")
            return fresh
        if fresh is None or not len(fresh):
            return cached
        return _merge(cached, fresh, count * 3)

    def _download(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> Bars | None:
        if self.backend == "mt5":
            from .data import recent

            return recent(symbol, timeframe, 500)
        return _dukascopy_live(symbol, timeframe, start, end)


def _merge(old: Bars, new: Bars, keep: int) -> Bars:
    """Splice newer bars onto older ones, newest value winning on overlap."""
    cut = int(np.searchsorted(old.time, int(new.time[0]), "left"))
    joined = Bars(
        symbol=old.symbol,
        timeframe=old.timeframe,
        time=np.concatenate([old.time[:cut], new.time]),
        open=np.concatenate([old.open[:cut], new.open]),
        high=np.concatenate([old.high[:cut], new.high]),
        low=np.concatenate([old.low[:cut], new.low]),
        close=np.concatenate([old.close[:cut], new.close]),
        volume=np.concatenate([old.volume[:cut], new.volume]),
        spread=np.concatenate([old.spread[:cut], new.spread]),
    )
    if len(joined) > keep:
        joined = joined.slice(len(joined) - keep, None)
    return joined


def _dukascopy_live(
    symbol: str, timeframe: str, start: datetime, end: datetime
) -> Bars | None:
    """Recent bars from Dukascopy, with the forming bar removed.

    Uses the plain historical endpoint rather than ``live_fetch``: the latter
    returns a generator of tens of thousands of tiny frames for even a few
    hours, and both expose the same most-recent bar.
    """
    step = tfmod.seconds(timeframe)
    # Never ask for less than a few bars, or the request comes back empty.
    start = min(start, end - timedelta(seconds=step * 4))

    # Dukascopy is a free public feed and stalls or fails intermittently. One
    # bad response should cost a few seconds, not the whole scan.
    bars = None
    for attempt in range(3):
        try:
            bars = sources.fetch(symbol, timeframe, start, end)
            if bars is not None and len(bars):
                break
        except Exception:
            if attempt == 2:
                raise
        time.sleep(2 * (attempt + 1))
    if bars is None or len(bars) == 0:
        return None

    # Drop the bar that is still forming -- its OHLC is not final yet, and
    # acting on it is how a live bot stops resembling its backtest.
    cutoff = int(datetime.now(timezone.utc).timestamp()) // step * step
    keep = bars.time < cutoff
    if not keep.any():
        return None
    return bars.slice(0, int(keep.sum()))
