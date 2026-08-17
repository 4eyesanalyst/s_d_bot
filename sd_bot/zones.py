"""Supply and demand zone detection.

The premise: institutions cannot fill size at one price. They accumulate inside
a small consolidation (the *base*), and when the remainder of the order finally
lifts the book, price leaves in a straight line (the *departure*). The unfilled
remainder of that order sits at the base. Price returning there is the only
place we are interested in trading.

Four shapes, all the same idea:

    DBR  drop-base-rally      -> demand   (reversal)
    RBR  rally-base-rally     -> demand   (continuation)
    RBD  rally-base-drop      -> supply   (reversal)
    DBD  drop-base-drop       -> supply   (continuation)

Boundaries follow the conservative convention: the *proximal* line (the side we
enter from) uses candle bodies, the *distal* line (the side the stop lives
beyond) uses wicks. That gives away a little fill rate to buy a lot of safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ZoneConfig
from .data import Bars
from .indicators import atr as atr_fn, body_ratio, bodies
from .structure import Structure

DEMAND = 1
SUPPLY = -1


@dataclass
class Zone:
    """One supply or demand zone, plus the state it accumulates over time."""

    symbol: str
    timeframe: str
    kind: int                 # DEMAND or SUPPLY
    pattern: str              # DBR / RBR / RBD / DBD
    base_start: int
    base_end: int
    created: int              # bar the departure completed; tradeable after this
    created_time: int
    proximal: float           # entry side
    distal: float             # stop side
    height: float
    atr_at_base: float
    departure_ratio: float
    caused_bos: bool
    has_imbalance: bool
    curve: float              # 0 = range low, 1 = range high, at creation
    htf_trend: int = 0

    tests: int = 0
    invalidated: bool = False
    traded: bool = False
    score: float = 0.0
    score_parts: dict[str, float] = field(default_factory=dict)

    # -- geometry -------------------------------------------------------------

    @property
    def is_demand(self) -> bool:
        return self.kind == DEMAND

    @property
    def top(self) -> float:
        return self.proximal if self.is_demand else self.distal

    @property
    def bottom(self) -> float:
        return self.distal if self.is_demand else self.proximal

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def entry_price(self, depth: float) -> float:
        """Price ``depth`` of the way from the proximal line into the zone."""
        span = self.proximal - self.distal  # signed: +ve for demand
        return self.proximal - span * depth

    def overlaps(self, other: "Zone") -> float:
        """Overlap as a fraction of the narrower zone (0 = disjoint, 1 = nested)."""
        lo = max(self.bottom, other.bottom)
        hi = min(self.top, other.top)
        if hi <= lo:
            return 0.0
        return (hi - lo) / max(min(self.height, other.height), 1e-12)

    # -- lifecycle ------------------------------------------------------------

    def touched(self, high: float, low: float) -> bool:
        """True when the bar's range reached into the zone from the proximal side."""
        return low <= self.proximal if self.is_demand else high >= self.proximal

    def broken(self, close: float) -> bool:
        """True when price *closed* through the distal line: the zone failed."""
        return close < self.distal if self.is_demand else close > self.distal

    def update(self, high: float, low: float, close: float) -> None:
        """Advance zone state with one new bar."""
        if self.invalidated:
            return
        if self.broken(close):
            self.invalidated = True
            return
        if self.touched(high, low):
            self.tests += 1

    def is_live(self, cfg: ZoneConfig, index: int) -> bool:
        """Still eligible to be traded at bar ``index``."""
        return (
            not self.invalidated
            and not self.traded
            and self.tests < cfg.max_tests
            and index - self.created <= cfg.max_zone_age_bars
        )

    def describe(self) -> str:
        side = "DEMAND" if self.is_demand else "SUPPLY"
        return (
            f"{self.symbol} {self.timeframe} {side} [{self.pattern}] "
            f"{self.bottom:.5f}-{self.top:.5f} "
            f"score={self.score:.0f} dep={self.departure_ratio:.1f}x "
            f"tests={self.tests}{' BOS' if self.caused_bos else ''}"
            f"{' FVG' if self.has_imbalance else ''}"
        )


def find_zones(
    bars: Bars,
    cfg: ZoneConfig,
    structure: Structure | None = None,
    atr_period: int = 14,
) -> list[Zone]:
    """Scan a series for every zone that passes the hard structural filters.

    The base is a *maximal run* of consolidation candles: the consolidation ends
    where a candle stops looking like consolidation, which is how a human draws
    it. What qualifies the zone is then how price left -- measured over a window,
    because a real departure often builds over two or three candles rather than
    printing one heroic bar.

    Scoring (which needs higher-timeframe context) happens separately in
    :mod:`sd_bot.scoring`.
    """
    n = len(bars)
    if n < atr_period + cfg.max_base_candles + cfg.leg_lookahead + 5:
        return []

    o, h, l, c = bars.open, bars.high, bars.low, bars.close
    a = atr_fn(h, l, c, atr_period)
    body = bodies(o, c)
    ratio = body_ratio(o, h, l, c)

    is_base = (body <= cfg.base_body_atr * a) & (ratio <= cfg.base_body_ratio)
    is_impulse = (body >= cfg.leg_body_atr * a) & (ratio >= cfg.leg_body_ratio)

    body_top = np.maximum(o, c)
    body_bottom = np.minimum(o, c)

    zones: list[Zone] = []
    first = atr_period + cfg.leg_in_lookback + 1
    # Scan right up to the newest bar. Reserving `leg_lookahead` bars at the end
    # would blind a live scanner to any zone whose departure has only just
    # completed -- and those are exactly the zones price is about to return to.
    # `_departure` already clamps its own window to the end of the data, so a
    # zone near the edge simply gets less room to prove itself, which is correct
    # rather than something to guard against.
    last = n - 1

    i = first
    while i < last:
        if not is_base[i]:
            i += 1
            continue

        # Extend to the end of this consolidation.
        s = i
        e = i
        while e + 1 < last and is_base[e + 1]:
            e += 1
        i = e + 1  # next scan resumes after the run, whatever we decide here

        length = e - s + 1
        if length < cfg.min_base_candles or length > cfg.max_base_candles:
            continue  # too brief to be accumulation, or too long: order absorbed
        if cfg.require_impulse_leg_in and not is_impulse[s - 1]:
            continue
        if cfg.require_impulse_leg_out and not is_impulse[e + 1]:
            continue

        zone = _build(
            bars, cfg, structure, s, e, a, body_top, body_bottom, h, l, c
        )
        if zone is not None:
            zones.append(zone)

    return _dedupe(zones)


def _departure(
    cfg: ZoneConfig,
    kind: int,
    proximal: float,
    distal: float,
    height: float,
    atr_base: float,
    e: int,
    limit: int,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
) -> tuple[int, float] | None:
    """Did price leave this base decisively? Returns ``(bar, ratio)`` or None.

    Two conditions, and both matter. The move must be large relative to the
    zone (an imbalance rather than a drift) *and* large in absolute terms -- a
    razor-thin base clears a ratio test far too easily, and its stop would sit
    inside the spread. Price closing back through the distal line first means
    the base failed, not departed.
    """
    extreme = proximal
    for k in range(e + 1, limit + 1):
        close = float(c[k])
        if kind == DEMAND:
            if close < distal:
                return None
            extreme = max(extreme, float(h[k]))
        else:
            if close > distal:
                return None
            extreme = min(extreme, float(l[k]))

        travelled = abs(extreme - proximal)
        if (
            travelled / height >= cfg.min_departure_ratio
            and travelled >= cfg.min_leg_out_atr * atr_base
        ):
            return k, travelled / height
    return None


def _build(
    bars: Bars,
    cfg: ZoneConfig,
    structure: Structure | None,
    s: int,
    e: int,
    a: np.ndarray,
    body_top: np.ndarray,
    body_bottom: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
) -> Zone | None:
    n = len(bars)
    atr_base = float(a[e])
    if atr_base <= 0:
        return None

    limit = min(e + cfg.leg_lookahead, n - 1)

    # Test both readings of the same base on their own merits rather than
    # letting one candle's colour decide which one we look at.
    candidates: list[tuple[int, float, float, float, int, float]] = []
    for kind in (DEMAND, SUPPLY):
        if kind == DEMAND:
            proximal = float(body_top[s : e + 1].max())
            distal = float(l[s : e + 1].min())
        else:
            proximal = float(body_bottom[s : e + 1].min())
            distal = float(h[s : e + 1].max())

        height = abs(proximal - distal)
        # A zone with no thickness cannot hold a stop clear of the spread, so
        # widen to the full base range before giving up on it.
        if height < cfg.min_zone_atr * atr_base:
            proximal = (
                float(h[s : e + 1].max()) if kind == DEMAND
                else float(l[s : e + 1].min())
            )
            height = abs(proximal - distal)
            if height < cfg.min_zone_atr * atr_base:
                continue
        # Too thick and it is a range, not an order block; the stop destroys R.
        if height > cfg.max_zone_atr * atr_base:
            continue

        found = _departure(
            cfg, kind, proximal, distal, height, atr_base, e, limit, h, l, c
        )
        if found is None:
            continue
        created, ratio = found
        candidates.append((kind, proximal, distal, height, created, ratio))

    if not candidates:
        return None  # price did not leave in a hurry: not an imbalance
    # If both readings somehow qualify, keep the more violent departure.
    kind, proximal, distal, height, created, departure_ratio = max(
        candidates, key=lambda t: t[5]
    )

    # Leg in: the net move over the candles approaching the base. This only
    # labels the pattern (reversal vs continuation); the departure is what
    # qualifies the zone.
    prior = c[max(0, s - 1 - cfg.leg_in_lookback)]
    in_is_bull = float(c[s - 1]) >= float(prior)
    if kind == DEMAND:
        pattern = "RBR" if in_is_bull else "DBR"
    else:
        pattern = "RBD" if in_is_bull else "DBD"

    caused_bos = False
    if structure is not None:
        level = (
            structure.swing_high_before(s)
            if kind == DEMAND
            else structure.swing_low_before(s)
        )
        if level is not None:
            window = slice(e + 1, created + 1)
            if kind == DEMAND:
                caused_bos = bool((c[window] > level).any())
            else:
                caused_bos = bool((c[window] < level).any())

    has_imbalance = _has_fvg(h, l, e, created, kind == DEMAND)

    curve = structure.curve_position(created, proximal) if structure else 0.5
    htf_trend = int(structure.trend[created]) if structure else 0

    return Zone(
        symbol=bars.symbol,
        timeframe=bars.timeframe,
        kind=kind,
        pattern=pattern,
        base_start=s,
        base_end=e,
        created=created,
        created_time=int(bars.time[created]),
        proximal=proximal,
        distal=distal,
        height=height,
        atr_at_base=atr_base,
        departure_ratio=float(departure_ratio),
        caused_bos=caused_bos,
        has_imbalance=has_imbalance,
        curve=float(curve),
        htf_trend=htf_trend,
    )


def _has_fvg(
    h: np.ndarray, l: np.ndarray, base_end: int, created: int, bullish: bool
) -> bool:
    """Three-candle fair value gap inside the departure: unfilled imbalance."""
    for k in range(base_end + 1, min(created, h.size - 2)):
        if bullish and l[k + 1] > h[k - 1]:
            return True
        if not bullish and h[k + 1] < l[k - 1]:
            return True
    return False


def _dedupe(zones: list[Zone]) -> list[Zone]:
    """Collapse stacked zones from consecutive bases into the strongest one."""
    kept: list[Zone] = []
    for z in sorted(zones, key=lambda z: z.created):
        duplicate = False
        for k in kept:
            if k.kind != z.kind or k.invalidated:
                continue
            if k.overlaps(z) < 0.5:
                continue
            # Same order block seen twice: keep the one that formed first.
            #
            # Deliberately NOT "keep the stronger departure". Swapping in a
            # later zone would rewrite history: at the moment the earlier zone
            # was live and tradeable, the replacement had not formed yet. A live
            # scanner cannot see it, so a backtest that does is reading the
            # future and reporting results the bot could never achieve.
            duplicate = True
            break
        if not duplicate:
            kept.append(z)
    kept.sort(key=lambda z: z.created)
    return kept


def update_all(zones: list[Zone], high: float, low: float, close: float) -> None:
    for z in zones:
        z.update(high, low, close)


def settle_on(zones: list[Zone], bars: Bars, zone_seconds: int,
              entry_seconds: int = 0) -> list[Zone]:
    """Replay zone state using a *lower* timeframe than the zones came from.

    The backtester ages zones with entry-timeframe bars, so a zone invalidated
    by an M15 close is dead even if the enclosing H1 candle closed back inside
    it. Settling on H1 instead would keep that zone alive and the live bot would
    signal a setup the backtest never took. Replaying on the same bars the
    backtester uses removes that whole class of divergence.

    ``entry_seconds`` aligns the *first* aged bar with the backtester, which
    activates a zone at the bar where ``bar_time + entry_seconds >=
    created_time + zone_seconds`` -- one entry bar earlier than the naive
    ``bar_time >= created_time + zone_seconds``. That single skipped bar is
    enough to miss a touch, leaving a zone the backtester had already retired
    looking fresh, and the live bot then signals a setup the strategy had spent.
    """
    for z in zones:
        # A zone becomes tradeable when the bar that completed its departure
        # closes; start ageing from the first entry bar whose *close* is at or
        # after that moment.
        start = int(np.searchsorted(
            bars.time, z.created_time + zone_seconds - entry_seconds, "left"))
        for k in range(start, len(bars)):
            z.update(float(bars.high[k]), float(bars.low[k]), float(bars.close[k]))
            if z.invalidated:
                break
    return zones


def settle(zones: list[Zone], bars: Bars) -> list[Zone]:
    """Replay history from each zone's creation so it carries its true state.

    The backtester ages zones incrementally as it walks the series. The live
    trader rebuilds zones from scratch on every bar, so it needs this to know
    which of them have already been tested or blown through.
    """
    n = len(bars)
    for z in zones:
        for k in range(z.created + 1, n):
            z.update(float(bars.high[k]), float(bars.low[k]), float(bars.close[k]))
            if z.invalidated:
                break
    return zones
