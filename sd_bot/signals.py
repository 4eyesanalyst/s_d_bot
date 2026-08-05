"""Turning a graded zone into an executable plan.

Two rules here do most of the work:

1. The stop goes beyond the *distal* line plus a volatility buffer -- never a
   fixed pip count. The market decides where the trade is wrong, not the trader.

2. Profit margin. A zone is only tradeable if there is enough clear air between
   it and the next opposing zone to pay the minimum reward ratio. Buying a
   beautiful demand zone twenty pips under a fat untested supply is the most
   common way this strategy loses money, and it is entirely avoidable.
"""

from __future__ import annotations

from .config import Config
from .scoring import grade
from .trades import LONG, SHORT, TradePlan, pip_size
from .zones import DEMAND, Zone


def nearest_opposing(
    zone: Zone, zones: list[Zone], entry: float, direction: int
) -> Zone | None:
    """The first opposing zone standing between the entry and open space."""
    best: Zone | None = None
    for other in zones:
        if other.invalidated or other is zone:
            continue
        if other.kind == zone.kind:
            continue
        if direction == LONG:
            # Overhead supply we would have to trade through.
            if other.proximal <= entry:
                continue
            if best is None or other.proximal < best.proximal:
                best = other
        else:
            if other.proximal >= entry:
                continue
            if best is None or other.proximal > best.proximal:
                best = other
    return best


def build_plan(
    zone: Zone,
    cfg: Config,
    index: int,
    atr_value: float,
    all_zones: list[Zone],
    spread_price: float = 0.0,
) -> tuple[TradePlan | None, str]:
    """Build a trade plan for ``zone``, or explain why there isn't one."""
    s = cfg.signal
    direction = LONG if zone.kind == DEMAND else SHORT

    if zone.score < cfg.zone.min_score:
        return None, f"score {zone.score:.0f} below {cfg.zone.min_score:.0f}"

    entry = zone.entry_price(s.entry_depth)

    buffer = max(s.stop_buffer_zone * zone.height, s.stop_buffer_atr * atr_value)
    # A stop inside the spread is not a stop.
    buffer = max(buffer, spread_price * 2.0)
    stop = zone.distal - buffer if direction == LONG else zone.distal + buffer

    risk_distance = abs(entry - stop)
    if risk_distance <= 0:
        return None, "degenerate stop distance"

    pip = pip_size(zone.symbol)
    stop_pips = risk_distance / pip if pip > 0 else 0.0
    if s.min_stop_pips > 0 and stop_pips < s.min_stop_pips:
        return None, f"stop only {stop_pips:.0f} pips (min {s.min_stop_pips:.0f})"
    if s.max_stop_pips > 0 and stop_pips > s.max_stop_pips:
        return None, f"stop {stop_pips:.0f} pips too wide (max {s.max_stop_pips:.0f})"

    # --- profit margin: is there room to make the minimum reward? ------------
    blocker = nearest_opposing(zone, all_zones, entry, direction)
    if blocker is not None:
        headroom_r = (blocker.proximal - entry) * direction / risk_distance
        if headroom_r < s.min_risk_reward:
            return None, (
                f"only {headroom_r:.1f}R to opposing zone at "
                f"{blocker.proximal:.5f} (need {s.min_risk_reward:.1f}R)"
            )
    else:
        headroom_r = s.tp2_r

    tp1 = entry + direction * s.tp1_r * risk_distance

    if s.tp2_mode == "opposing_zone" and blocker is not None:
        # Park just in front of the opposing zone; do not ask price to trade
        # into it and come back out.
        tp2 = blocker.proximal - direction * buffer
    else:
        tp2 = entry + direction * s.tp2_r * risk_distance

    rr2 = (tp2 - entry) * direction / risk_distance
    if rr2 < s.min_risk_reward:
        return None, f"target only {rr2:.1f}R (need {s.min_risk_reward:.1f}R)"
    if rr2 < s.tp1_r:
        return None, "second target sits inside the first"

    return (
        TradePlan(
            symbol=zone.symbol,
            direction=direction,
            entry=entry,
            stop=stop,
            tp1=tp1,
            tp2=tp2,
            risk_distance=risk_distance,
            rr1=s.tp1_r,
            rr2=rr2,
            score=zone.score,
            zone_id=id(zone),
            zone_note=f"{zone.pattern} {grade(zone.score)} "
            f"{zone.timeframe} dep={zone.departure_ratio:.1f}x",
            created_index=index,
            expires_index=index + s.pending_expiry_bars,
        ),
        "",
    )


def confirmed_entry(
    zone: Zone,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> bool:
    """Confirmation-mode trigger.

    Price must have traded into the zone and then *closed* back out through the
    proximal line in our direction -- a rejection, not just a touch. Costs some
    reward ratio, buys a materially higher hit rate on counter-trend zones.
    """
    if zone.is_demand:
        return low <= zone.proximal and close > zone.proximal and close > open_
    return high >= zone.proximal and close < zone.proximal and close < open_
