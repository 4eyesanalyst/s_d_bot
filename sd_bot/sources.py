"""Historical data from Dukascopy, for backtesting without an MT5 terminal.

Dukascopy publishes free bank-feed history going back further than most retail
MT5 servers keep it. Bars land in the same CSV cache format :class:`Bars` reads,
so the backtester cannot tell the difference between this and broker data.

Spreads are not part of the free bar feed, so each symbol carries a typical
retail figure below. They are deliberately on the wide side -- understating cost
is the fastest way to manufacture an edge that does not exist.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import timeframes as tfmod
from .data import Bars, cache_path
from .trades import SymbolSpec, pip_size  # noqa: F401  (re-exported)

# Quiet the per-chunk progress spam from the library.
logging.getLogger("DUKASCRIPT").setLevel(logging.WARNING)

_INTERVALS = {
    "M1": "INTERVAL_MIN_1",
    "M5": "INTERVAL_MIN_5",
    "M15": "INTERVAL_MIN_15",
    "M30": "INTERVAL_MIN_30",
    "H1": "INTERVAL_HOUR_1",
    "H4": "INTERVAL_HOUR_4",
    "D1": "INTERVAL_DAY_1",
}

# symbol -> (dukascopy instrument, typical retail spread in pips)
CATALOGUE: dict[str, tuple[str, float]] = {
    # majors
    "EURUSD": ("EUR/USD", 0.6),
    "GBPUSD": ("GBP/USD", 0.9),
    "USDJPY": ("USD/JPY", 0.7),
    "USDCHF": ("USD/CHF", 1.0),
    "AUDUSD": ("AUD/USD", 0.8),
    "USDCAD": ("USD/CAD", 1.0),
    "NZDUSD": ("NZD/USD", 1.2),
    # crosses
    "EURJPY": ("EUR/JPY", 1.2),
    "GBPJPY": ("GBP/JPY", 1.8),
    "EURGBP": ("EUR/GBP", 1.0),
    "AUDJPY": ("AUD/JPY", 1.4),
    "CADJPY": ("CAD/JPY", 1.6),
    "CHFJPY": ("CHF/JPY", 2.0),
    "NZDJPY": ("NZD/JPY", 1.8),
    "EURAUD": ("EUR/AUD", 1.6),
    "EURCAD": ("EUR/CAD", 1.8),
    "EURCHF": ("EUR/CHF", 1.2),
    "GBPAUD": ("GBP/AUD", 2.2),
    "GBPCAD": ("GBP/CAD", 2.4),
    "AUDNZD": ("AUD/NZD", 1.8),
    "AUDCAD": ("AUD/CAD", 1.6),
    "NZDCAD": ("NZD/CAD", 2.0),
    # metals
    "XAUUSD": ("XAU/USD", 2.5),
    "XAGUSD": ("XAG/USD", 2.5),
}

DEFAULT_BASKET = list(CATALOGUE)


def spec_for(symbol: str) -> SymbolSpec:
    """Contract spec matching a typical retail broker, for offline backtests."""
    s = symbol.upper()
    if s.startswith("XAU"):
        return SymbolSpec(symbol, digits=2, point=0.01, tick_size=0.01,
                          tick_value=1.0, contract_size=100.0)
    if s.startswith("XAG"):
        return SymbolSpec(symbol, digits=3, point=0.001, tick_size=0.001,
                          tick_value=5.0, contract_size=5000.0)
    if s.endswith("JPY"):
        # 1 lot = 100,000 base; 0.001 JPY per unit ~= $0.67 at 150 USDJPY.
        return SymbolSpec(symbol, digits=3, point=0.001, tick_size=0.001,
                          tick_value=0.67, contract_size=100_000.0)
    if s.endswith("USD"):
        return SymbolSpec(symbol, digits=5, point=0.00001, tick_size=0.00001,
                          tick_value=1.0, contract_size=100_000.0)
    # USD is the base (USDCHF, USDCAD) or a pure cross: value varies with rate.
    # 0.85 is a reasonable standing approximation for the majors involved.
    return SymbolSpec(symbol, digits=5, point=0.00001, tick_size=0.00001,
                      tick_value=0.85, contract_size=100_000.0)


def spread_points(symbol: str) -> float:
    """Typical spread expressed in points (tick units), as MT5 reports it."""
    _, pips = CATALOGUE.get(symbol.upper(), ("", 1.5))
    return pips * pip_size(symbol) / spec_for(symbol).tick_size


def fetch(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> Bars:
    """Download bars from Dukascopy and wrap them in a :class:`Bars`."""
    import dukascopy_python

    if symbol.upper() not in CATALOGUE:
        raise ValueError(
            f"{symbol!r} is not in the catalogue. Add it to sources.CATALOGUE "
            f"with its Dukascopy name and typical spread."
        )
    instrument, _ = CATALOGUE[symbol.upper()]
    interval = getattr(dukascopy_python, _INTERVALS[timeframe.upper()])

    df = dukascopy_python.fetch(
        instrument, interval, dukascopy_python.OFFER_SIDE_BID, start, end
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"Dukascopy returned no {timeframe} data for {symbol}")

    import pandas as pd

    df = df[~df.index.duplicated(keep="first")].sort_index()
    # Dukascopy hands back a datetime64[ms] index, not the nanoseconds pandas
    # historically used. Convert through an explicit second-resolution dtype so
    # this stays correct whatever unit the library returns.
    index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    times = index.to_numpy(dtype="datetime64[s]").astype(np.int64)

    return Bars(
        symbol=symbol,
        timeframe=timeframe,
        time=times,
        open=df["open"].to_numpy(dtype=float),
        high=df["high"].to_numpy(dtype=float),
        low=df["low"].to_numpy(dtype=float),
        close=df["close"].to_numpy(dtype=float),
        volume=df["volume"].to_numpy(dtype=float),
        spread=np.full(len(df), spread_points(symbol)),
    )


def load(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    cache_dir: str | Path = "data",
    refresh: bool = False,
) -> Bars:
    """Cached fetch. Downloads only when the cache cannot serve the range."""
    path = cache_path(cache_dir, symbol, timeframe)
    if path.exists() and not refresh:
        bars = Bars.from_csv(path, symbol, timeframe)
        span = tfmod.seconds(timeframe)
        if len(bars) and bars.time[0] <= start.timestamp() + span and \
                bars.time[-1] >= end.timestamp() - span * 3:
            lo = int(np.searchsorted(bars.time, int(start.timestamp()), "left"))
            hi = int(np.searchsorted(bars.time, int(end.timestamp()), "right"))
            return bars.slice(lo, hi)

    bars = fetch(symbol, timeframe, start, end)
    bars.to_csv(path)
    return bars


def load_series(
    cfg, symbol: str, cache_dir: str | Path = "data", refresh: bool = False
) -> dict[str, Bars]:
    """The three timeframes the strategy needs, all from Dukascopy."""
    bt = cfg.backtest
    start = datetime.fromisoformat(bt.start).replace(tzinfo=timezone.utc)
    end = (
        datetime.fromisoformat(bt.end).replace(tzinfo=timezone.utc)
        if bt.end else datetime.now(timezone.utc)
    )
    ex = cfg.execution
    out = {}
    for tf in {ex.entry_timeframe, ex.zone_timeframe, ex.bias_timeframe}:
        out[tf] = load(symbol, tf, start, end, cache_dir=cache_dir, refresh=refresh)
    return out
