"""Session filtering.

Supply and demand zones need participation to work. A demand zone tapped during
the Asian session on a EUR pair usually just leaks through it; the same zone
tapped at the London open gets defended. Restricting hours removes a large slice
of the losing tail for free.

Times are derived with integer arithmetic rather than ``datetime`` objects. The
backtester calls these functions once per bar per symbol -- millions of times in
a portfolio run -- and constructing a timezone-aware datetime each call dominated
the profile.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .config import ExecutionConfig

_DAY = 86_400


def utc_weekday(ts: int) -> int:
    """Monday=0 ... Sunday=6, matching ``datetime.weekday()``.

    1970-01-01 was a Thursday, which is weekday 3.
    """
    return (ts // _DAY + 3) % 7


def utc_hour(ts: int) -> int:
    return (ts % _DAY) // 3600


def day_index(ts: int) -> int:
    """Days since the epoch -- a cheap identity for 'same UTC day'."""
    return ts // _DAY


def week_index(ts: int) -> int:
    """Monday-based week number since the epoch."""
    return (ts // _DAY + 3) // 7


def in_session(ts: int, cfg: ExecutionConfig) -> bool:
    if utc_weekday(ts) not in cfg.trade_days:
        return False
    return utc_hour(ts) in cfg.session_hours_utc


def is_week_close(ts: int, cfg: ExecutionConfig) -> bool:
    """True once we are inside the Friday flatten window."""
    return utc_weekday(ts) == 4 and utc_hour(ts) >= cfg.friday_close_hour_utc


def to_datetime(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)
