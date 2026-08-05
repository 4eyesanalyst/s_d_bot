"""Timeframe helpers.

Kept free of any MetaTrader5 import so the backtester and the zone engine can be
used (and unit-tested) on a machine with no terminal installed.
"""

from __future__ import annotations

# Timeframe label -> duration in minutes.
MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
}


def minutes(tf: str) -> int:
    """Duration of one bar of ``tf`` in minutes."""
    try:
        return MINUTES[tf.upper()]
    except KeyError:
        raise ValueError(
            f"unknown timeframe {tf!r}; expected one of {sorted(MINUTES)}"
        ) from None


def seconds(tf: str) -> int:
    """Duration of one bar of ``tf`` in seconds."""
    return minutes(tf) * 60


def is_higher(a: str, b: str) -> bool:
    """True when timeframe ``a`` is strictly higher (slower) than ``b``."""
    return minutes(a) > minutes(b)


def mt5_constant(tf: str) -> int:
    """Map a timeframe label onto the MetaTrader5 ``TIMEFRAME_*`` constant.

    Imported lazily so this module stays usable without the MT5 package.
    """
    import MetaTrader5 as mt5

    return getattr(mt5, f"TIMEFRAME_{tf.upper()}")
