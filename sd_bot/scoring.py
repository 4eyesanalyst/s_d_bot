"""Zone quality scoring.

Every zone that passes the structural filters in :mod:`sd_bot.zones` is *valid*.
Very few are *worth trading*. This module grades them 0-100 so the bot can hold
out for the top of the distribution instead of taking everything it finds.

The weights encode an opinion about what actually pays:

  freshness      the biggest single edge. Untouched zones still hold unfilled
                 orders; retested ones have already given them away.
  departure      proof that the move away was an imbalance, not a drift.
  BOS            the departure broke structure, so it was institutional intent
                 rather than noise inside a range.
  HTF trend      trading demand in a downtrend is how good zones still lose.
  curve          buy demand in the discount half, sell supply in the premium
                 half. Location beats zone quality more often than traders like.
  base tightness fewer base candles = less of the order already absorbed.
  confluence     a zone nested inside a higher-timeframe zone is the A+ setup.
  imbalance      an unfilled FVG in the departure leg confirms the vacuum.
"""

from __future__ import annotations

from .config import Config
from .structure import DOWN, RANGE, UP
from .zones import DEMAND, Zone


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def freshness_value(tests: int) -> float:
    if tests <= 0:
        return 1.0
    if tests == 1:
        return 0.45
    return 0.0


def departure_value(ratio: float) -> float:
    """2x zone height scores nothing, 6x scores full marks."""
    return _clamp((ratio - 2.0) / 4.0)


def base_tightness_value(candles: int) -> float:
    """One base candle is ideal; every extra candle absorbs more of the order."""
    return _clamp(1.0 - (candles - 1) * 0.14)


def trend_value(zone_kind: int, htf_trend: int) -> float:
    if htf_trend == RANGE:
        return 0.5
    aligned = (zone_kind == DEMAND and htf_trend == UP) or (
        zone_kind != DEMAND and htf_trend == DOWN
    )
    return 1.0 if aligned else 0.0


def curve_value(zone_kind: int, curve: float, tolerance: float) -> float:
    """Reward demand in discount and supply in premium, linearly."""
    span = 0.5 + tolerance
    if zone_kind == DEMAND:
        return _clamp((span - curve) / span)
    return _clamp((curve - 0.5 + tolerance) / span)


def confluence_value(zone: Zone, htf_zones: list[Zone] | None) -> float:
    if not htf_zones:
        return 0.0
    best = 0.0
    for hz in htf_zones:
        if hz.kind != zone.kind or hz.invalidated:
            continue
        best = max(best, zone.overlaps(hz))
    if best >= 0.5:
        return 1.0
    return 0.5 if best > 0.0 else 0.0


def score_zone(
    zone: Zone,
    cfg: Config,
    htf_zones: list[Zone] | None = None,
    htf_trend: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Grade a zone at decision time. Returns ``(total, contributions)``."""
    z = cfg.zone
    trend = zone.htf_trend if htf_trend is None else htf_trend
    candles = zone.base_end - zone.base_start + 1

    parts = {
        "freshness": z.w_freshness * freshness_value(zone.tests),
        "departure": z.w_departure * departure_value(zone.departure_ratio),
        "base_tightness": z.w_base_tightness * base_tightness_value(candles),
        "bos": z.w_bos * (1.0 if zone.caused_bos else 0.0),
        "htf_trend": z.w_htf_trend * trend_value(zone.kind, trend),
        "curve": z.w_curve
        * curve_value(zone.kind, zone.curve, cfg.structure.equilibrium_tolerance),
        "htf_confluence": z.w_htf_confluence * confluence_value(zone, htf_zones),
        "imbalance": z.w_imbalance * (1.0 if zone.has_imbalance else 0.0),
    }
    return sum(parts.values()), parts


def apply_score(
    zone: Zone,
    cfg: Config,
    htf_zones: list[Zone] | None = None,
    htf_trend: int | None = None,
) -> float:
    """Score a zone and write the result back onto it."""
    total, parts = score_zone(zone, cfg, htf_zones, htf_trend)
    zone.score = total
    zone.score_parts = parts
    return total


def grade(score: float) -> str:
    if score >= 85:
        return "A+"
    if score >= 75:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def explain(zone: Zone) -> str:
    """One-line breakdown, for the journal and for eyeballing live decisions."""
    if not zone.score_parts:
        return zone.describe()
    parts = " ".join(
        f"{k}={v:.0f}" for k, v in sorted(
            zone.score_parts.items(), key=lambda kv: -kv[1]
        )
    )
    return f"{zone.describe()} grade={grade(zone.score)} | {parts}"
