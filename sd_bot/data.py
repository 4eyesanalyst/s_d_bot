"""Bar containers, MT5 download, CSV cache and no-lookahead timeframe alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import timeframes as tfmod


@dataclass
class Bars:
    """OHLC series as parallel numpy arrays.

    ``time`` holds the bar's *open* time as a UTC epoch in seconds.
    """

    symbol: str
    timeframe: str
    time: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    spread: np.ndarray

    def __len__(self) -> int:
        return int(self.time.size)

    def __post_init__(self) -> None:
        n = self.time.size
        for name in ("open", "high", "low", "close", "volume", "spread"):
            arr = getattr(self, name)
            if arr.size != n:
                raise ValueError(
                    f"{self.symbol} {self.timeframe}: {name} has {arr.size} rows, "
                    f"expected {n}"
                )
        if n:
            self.validate()

    def validate(self) -> None:
        """Refuse data that would silently poison every downstream result.

        Corrupt timestamps are the dangerous case: bars still look like valid
        OHLC, the backtest still runs, and every number it produces is fiction.
        This has already happened once (a millisecond epoch read as nanoseconds
        collapsed a whole year onto one instant), so it is checked, not assumed.
        """
        label = f"{self.symbol} {self.timeframe}"

        # Plausible epoch seconds: year 2000 to year 2100.
        if self.time[0] < 946_684_800 or self.time[-1] > 4_102_444_800:
            raise ValueError(
                f"{label}: timestamps out of range "
                f"({self.time[0]} .. {self.time[-1]}). Expected epoch *seconds* "
                f"-- check the source's datetime resolution."
            )
        if np.any(np.diff(self.time) <= 0):
            bad = int(np.argmin(np.diff(self.time)))
            raise ValueError(
                f"{label}: timestamps are not strictly increasing at row {bad} "
                f"({self.time[bad]} -> {self.time[bad + 1]})"
            )
        if np.any(self.high < self.low):
            raise ValueError(f"{label}: found bars with high < low")
        if np.any(self.high < self.open) or np.any(self.high < self.close):
            raise ValueError(f"{label}: found bars with high below open/close")
        if np.any(self.low > self.open) or np.any(self.low > self.close):
            raise ValueError(f"{label}: found bars with low above open/close")
        if np.any(~np.isfinite(self.close)) or np.any(self.close <= 0):
            raise ValueError(f"{label}: found non-finite or non-positive prices")

    def slice(self, start: int, end: int | None = None) -> "Bars":
        s = slice(start, end)
        return Bars(
            symbol=self.symbol,
            timeframe=self.timeframe,
            time=self.time[s],
            open=self.open[s],
            high=self.high[s],
            low=self.low[s],
            close=self.close[s],
            volume=self.volume[s],
            spread=self.spread[s],
        )

    def datetimes(self) -> list[datetime]:
        return [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in self.time]

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = np.column_stack(
            [self.time, self.open, self.high, self.low, self.close,
             self.volume, self.spread]
        )
        header = "time,open,high,low,close,volume,spread"
        np.savetxt(path, rows, delimiter=",", header=header, comments="",
                   fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f", "%.2f", "%.2f"])

    @classmethod
    def from_csv(cls, path: str | Path, symbol: str, timeframe: str) -> "Bars":
        # pandas' C parser, not numpy's text reader: these files run to hundreds
        # of thousands of rows and genfromtxt takes tens of seconds each.
        import pandas as pd

        raw = pd.read_csv(path)
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            time=raw["time"].to_numpy(dtype=np.int64),
            open=raw["open"].to_numpy(dtype=float),
            high=raw["high"].to_numpy(dtype=float),
            low=raw["low"].to_numpy(dtype=float),
            close=raw["close"].to_numpy(dtype=float),
            volume=raw["volume"].to_numpy(dtype=float),
            spread=raw["spread"].to_numpy(dtype=float),
        )


def cache_path(cache_dir: str | Path, symbol: str, timeframe: str) -> Path:
    return Path(cache_dir) / f"{symbol}_{timeframe}.csv"


def load(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    cache_dir: str | Path = "data",
    refresh: bool = False,
) -> Bars:
    """Return bars, downloading from MT5 only when the cache cannot serve them."""
    path = cache_path(cache_dir, symbol, timeframe)
    if path.exists() and not refresh:
        bars = Bars.from_csv(path, symbol, timeframe)
        if len(bars) and bars.time[0] <= start.timestamp() and \
                bars.time[-1] >= end.timestamp() - tfmod.seconds(timeframe) * 2:
            return _trim(bars, start, end)
    bars = download(symbol, timeframe, start, end)
    bars.to_csv(path)
    return _trim(bars, start, end)


def _trim(bars: Bars, start: datetime, end: datetime) -> Bars:
    lo = int(np.searchsorted(bars.time, int(start.timestamp()), side="left"))
    hi = int(np.searchsorted(bars.time, int(end.timestamp()), side="right"))
    return bars.slice(lo, hi)


def download(symbol: str, timeframe: str, start: datetime, end: datetime) -> Bars:
    """Pull bars straight from a running MetaTrader 5 terminal."""
    import MetaTrader5 as mt5

    from .broker import ensure_initialized, ensure_symbol

    ensure_initialized()
    ensure_symbol(symbol)

    rates = mt5.copy_rates_range(
        symbol, tfmod.mt5_constant(timeframe), start, end
    )
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"MT5 returned no {timeframe} bars for {symbol} between "
            f"{start:%Y-%m-%d} and {end:%Y-%m-%d}. Check the symbol is visible in "
            f"Market Watch and that history is downloaded (last error: "
            f"{mt5.last_error()})."
        )
    return Bars(
        symbol=symbol,
        timeframe=timeframe,
        time=rates["time"].astype(np.int64),
        open=rates["open"].astype(float),
        high=rates["high"].astype(float),
        low=rates["low"].astype(float),
        close=rates["close"].astype(float),
        volume=rates["tick_volume"].astype(float),
        spread=rates["spread"].astype(float),
    )


def recent(
    symbol: str, timeframe: str, count: int, drop_forming: bool = True
) -> Bars:
    """Latest ``count`` bars from the terminal.

    ``drop_forming`` removes the bar that is still building, which is the whole
    point: every decision this bot makes is taken on closed bars only.
    """
    import MetaTrader5 as mt5

    from .broker import ensure_initialized, ensure_symbol

    ensure_initialized()
    ensure_symbol(symbol)

    n = count + (1 if drop_forming else 0)
    rates = mt5.copy_rates_from_pos(symbol, tfmod.mt5_constant(timeframe), 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"no {timeframe} bars available for {symbol}: {mt5.last_error()}"
        )
    if drop_forming and len(rates) > 1:
        rates = rates[:-1]

    return Bars(
        symbol=symbol,
        timeframe=timeframe,
        time=rates["time"].astype(np.int64),
        open=rates["open"].astype(float),
        high=rates["high"].astype(float),
        low=rates["low"].astype(float),
        close=rates["close"].astype(float),
        volume=rates["tick_volume"].astype(float),
        spread=rates["spread"].astype(float),
    )


def align(htf: Bars, ltf: Bars) -> np.ndarray:
    """Map each LTF bar to the last **fully closed** HTF bar.

    Returns an int array the length of ``ltf`` holding indices into ``htf``, or
    ``-1`` where no HTF bar has closed yet. This is the guard that keeps the
    backtester honest: a decision taken at the close of an M15 bar may only see
    H4 bars that had already finished by then.
    """
    htf_close = htf.time + tfmod.seconds(htf.timeframe)
    ltf_close = ltf.time + tfmod.seconds(ltf.timeframe)
    idx = np.searchsorted(htf_close, ltf_close, side="right") - 1
    return idx.astype(np.int64)


def resample(bars: Bars, target: str) -> Bars:
    """Aggregate bars up to a higher timeframe on calendar-aligned buckets."""
    step = tfmod.seconds(target)
    if step <= tfmod.seconds(bars.timeframe):
        raise ValueError(
            f"cannot resample {bars.timeframe} up to {target}: target is not higher"
        )

    buckets = (bars.time // step) * step
    # Bucket boundaries: where the bucket id changes.
    starts = np.flatnonzero(np.diff(buckets, prepend=buckets[0] - 1))
    ends = np.append(starts[1:], len(bars))

    o = bars.open[starts]
    c = bars.close[ends - 1]
    h = np.array([bars.high[s:e].max() for s, e in zip(starts, ends)])
    lo = np.array([bars.low[s:e].min() for s, e in zip(starts, ends)])
    v = np.array([bars.volume[s:e].sum() for s, e in zip(starts, ends)])
    sp = np.array([bars.spread[s:e].mean() for s, e in zip(starts, ends)])

    return Bars(
        symbol=bars.symbol,
        timeframe=target,
        time=buckets[starts],
        open=o,
        high=h,
        low=lo,
        close=c,
        volume=v,
        spread=sp,
    )


def synthetic(
    symbol: str = "TESTFX",
    timeframe: str = "M15",
    n: int = 4000,
    seed: int = 7,
    start: datetime | None = None,
    price: float = 1.1000,
    trendiness: float = 0.7,
) -> Bars:
    """Generate bars with a realistic impulse-and-base rhythm.

    Volatility is scaled so an M15 bar spans roughly 8-12 pips on a 1.10 pair,
    which matters: unrealistically quiet data produces four-pip stops, and
    trading costs then swamp every result.

    ``trendiness`` is the share of time spent in a drifting regime. Set it to
    ``0.0`` for a pure driftless random walk -- useful for checking that the
    engine has no directional bias of its own.

    This is a correctness fixture, not a market model. Never judge the strategy
    on it.
    """
    rng = np.random.default_rng(seed)
    step = tfmod.seconds(timeframe)
    t0 = int((start or datetime(2023, 1, 2, tzinfo=timezone.utc)).timestamp())

    tick = 0.00001 if price < 50 else 0.01
    # ~7 pips of per-bar sigma at M15, scaled by the bar length.
    vol = 70 * tick * (price / 1.1) * math.sqrt(tfmod.minutes(timeframe) / 15.0)

    closes = np.empty(n)
    opens = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)

    trend_p = max(0.0, min(1.0, trendiness)) / 2.0
    weights = [trend_p, trend_p, 1.0 - 2.0 * trend_p]

    p = price
    regime_left, drift = 0, 0.0
    for i in range(n):
        if regime_left <= 0:
            # Alternate impulsive drift with quiet consolidation, which is what
            # actually creates bases and departures for the engine to find.
            regime = rng.choice([1.0, -1.0, 0.0], p=weights)
            regime_left = int(rng.integers(15, 60))
            drift = regime * vol * rng.uniform(0.25, 0.7)
        regime_left -= 1

        o = p
        body = drift + rng.normal(0, vol)
        c = o + body
        wick = abs(rng.normal(0, vol * 0.5))
        h = max(o, c) + wick * rng.uniform(0.2, 1.0)
        lo = min(o, c) - wick * rng.uniform(0.2, 1.0)
        opens[i], closes[i], highs[i], lows[i] = o, c, h, lo
        p = c

    return Bars(
        symbol=symbol,
        timeframe=timeframe,
        time=np.arange(t0, t0 + n * step, step, dtype=np.int64)[:n],
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        volume=rng.integers(200, 2000, n).astype(float),
        spread=np.full(n, 8.0),
    )
