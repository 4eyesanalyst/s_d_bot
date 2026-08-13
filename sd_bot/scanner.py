"""Real-time signal service.

Watches a list of instruments, applies exactly the rules the backtester applies,
and pushes an alert the moment a setup becomes actionable. It then keeps
following that signal and tells you when it fills, banks its first target, or
stops out -- so the exit is delivered as reliably as the entry.

Three alert kinds:

    SET ORDER    a zone qualifies and price is heading toward it. This is the
                 actionable one: place a pending limit order at the given price.
                 It arrives hours early on purpose, because a scheduled runner
                 can be delayed and a resting order does not care when it was
                 told about -- it fills the moment price arrives.
    TRIGGERED    price reached the level; a resting order has filled.
    UPDATE       a live signal hit TP1/TP2, stopped out, or was invalidated.

State is persisted, so a restart does not re-alert setups you have already seen.
"""

from __future__ import annotations

import json
import time

import numpy as np
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import sources
from . import timeframes as tfmod
from .config import Config
from .feed import Feed
from .indicators import atr as atr_fn
from .notify import Broadcaster
from .scoring import apply_score, grade
from .sessions import in_session, is_week_close
from .signals import build_plan
from .structure import Structure, trend_name
from .trades import LONG, pip_size
from .zones import DEMAND, find_zones, settle, settle_on, update_all

# Generous windows on purpose. Swing structure, trend and the dealing range are
# all history-dependent, so a short window makes the live bot score zones
# differently from the backtest and silently changes the strategy.
# Bars examined per scan after a gap. GitHub can delay a 20-minute cron by two
# hours or more, so a scan routinely has 6-9 bars to catch up on; this cap only
# bites after a real outage, where replaying days of resolved setups would just
# produce stale alerts.
MAX_CATCHUP_BARS = 200

BARS_ENTRY = 1500
BARS_ZONE = 4000
BARS_BIAS = 1500


class DeliveryError(RuntimeError):
    """An alert was generated but never reached the user."""


@dataclass
class ActiveSignal:
    """A signal we have alerted on and are still following."""

    key: str
    symbol: str
    direction: int
    entry: float
    stop: float
    tp1: float
    tp2: float
    score: float
    zone_note: str
    created: str
    status: str = "pending"      # pending -> filled -> tp1 -> closed
    alerted_approach: bool = False
    # Epoch seconds marking when this signal entered its current phase. Exits
    # must only be judged against price action *after* that moment: a resting
    # order cannot be filled by a bar that predates it, and a target cannot be
    # hit by a spike that happened before entry.
    created_ts: int = 0
    filled_ts: int = 0

    @property
    def is_long(self) -> bool:
        return self.direction == LONG


class SignalScanner:
    def __init__(self, cfg: Config, broadcaster: Broadcaster,
                 feed: Feed, state_dir: str = "signals"):
        self.cfg = cfg
        self.out = broadcaster
        self.feed = feed
        self.state_path = Path(state_dir) / "state.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.active: dict[str, ActiveSignal] = {}
        self.cooldown: dict[str, float] = {}
        # Every entry ever signalled this session, for parity testing against
        # the backtester.
        self.all_signalled: list[ActiveSignal] = []
        # Zone keys already traded. The backtester sets zone.traded once and
        # that zone is gone for good -- neither re-entered nor counted as an
        # obstacle. The scanner rebuilds its zone map from scratch whenever a
        # zone-timeframe bar closes, which would resurrect spent zones, so the
        # fact has to be remembered here and reapplied.
        self.signalled_keys: set[str] = set()
        # Zones we have already told the user to place an order at. A zone is a
        # fixed price level: once the pending order is resting there, repeating
        # the message is noise, not information. One alert per zone, ever.
        self.ordered_keys: set[str] = set()
        self._last_bar: dict[str, int] = {}
        # Per-symbol zone map, rebuilt only when a zone-timeframe bar closes.
        self._analysis: dict[str, dict] = {}
        self._last_heartbeat = 0.0
        # UTC date of the last daily heartbeat, persisted so a scheduled runner
        # (a fresh process each time) knows whether today's has already gone.
        self.last_heartbeat_day = ""
        self._load()

    @staticmethod
    def _zone_key(symbol: str, zone) -> str:
        return f"{symbol}:{zone.timeframe}:{zone.created_time}:{zone.kind}"

    # -- persistence ----------------------------------------------------------

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.active = {k: ActiveSignal(**v) for k, v in raw.get("active", {}).items()}
            self.cooldown = raw.get("cooldown", {})
            self.signalled_keys = set(raw.get("signalled", []))
            self.ordered_keys = set(raw.get("ordered", []))
            self.last_heartbeat_day = raw.get("heartbeat_day", "")
        except Exception:
            self.active, self.cooldown = {}, {}
            self.signalled_keys, self.ordered_keys = set(), set()

    def _save(self) -> None:
        payload = {
            "active": {k: asdict(v) for k, v in self.active.items()},
            "cooldown": self.cooldown,
            # Bounded: only the most recent keys matter, older zones age out.
            "signalled": sorted(self.signalled_keys)[-400:],
            "ordered": sorted(self.ordered_keys)[-400:],
            "heartbeat_day": self.last_heartbeat_day,
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- main loop ------------------------------------------------------------

    def run(self) -> None:
        self.startup_report()
        try:
            while True:
                try:
                    self.poll()
                except Exception as exc:
                    print(f"[{_now()}] scan error: {exc!r}", flush=True)
                time.sleep(self.cfg.alerts.poll_seconds)
        except KeyboardInterrupt:
            print("\nstopped.")
            self._save()

    def poll(self) -> None:
        now = int(datetime.now(timezone.utc).timestamp())

        # Follow live signals every poll: exits are time-critical.
        self.track(now)

        for symbol in self.cfg.execution.symbols:
            try:
                self.scan(symbol, now)
            except Exception as exc:
                print(f"[{_now()}] {symbol}: {exc!r}", flush=True)

        self.heartbeat()
        self._save()

    # -- following open signals -----------------------------------------------

    def track(self, now: int) -> None:
        for key, sig in list(self.active.items()):
            tick = self.feed.tick(sig.symbol)
            if tick is None:
                continue
            bid, ask = tick
            close = bid if sig.is_long else ask
            digits = sources.spec_for(sig.symbol).digits

            # Levels are hit by the bar's *range*, not its close.
            #
            # A broker fills a take-profit the instant price touches it. Judging
            # exits on the closing price alone means a spike straight through
            # TP1 that closes back below it is never reported -- the trade is
            # closed at the broker while the bot still thinks it is running.
            # Worse, a scheduled runner may not wake for an hour, so several
            # bars can pass between checks. Both are handled by scanning every
            # bar since the signal was last evaluated and using its extremes.
            # ...but only bars that belong to this phase of the trade.
            #
            # A fixed lookback window was wrong and dangerously so: it reported
            # TP1 on a gold trade using a spike that occurred two hours before
            # the entry filled. A level can only be counted as hit by price that
            # traded while we were actually in that state.
            boundary = sig.filled_ts if sig.status in ("filled", "tp1") \
                else sig.created_ts
            recent = self.feed.bars(sig.symbol, self.cfg.execution.entry_timeframe, 24)
            hi = lo = close
            if recent is not None and len(recent):
                if boundary:
                    keep = recent.time >= boundary
                    if keep.any():
                        hi = float(recent.high[keep].max())
                        lo = float(recent.low[keep].min())
                else:
                    # No boundary recorded (state written before this fix):
                    # fall back to the latest closed bar only, never a window.
                    hi = float(recent.high[-1])
                    lo = float(recent.low[-1])

            # For a long, favourable levels are reached by the high and adverse
            # ones by the low; mirrored for a short.
            best = hi if sig.is_long else lo
            worst = lo if sig.is_long else hi
            price = close

            if sig.status == "pending":
                reached = worst <= sig.entry if sig.is_long else worst >= sig.entry
                # A zone that fails before we ever fill is dead, not a trade.
                blown = worst < sig.stop if sig.is_long else worst > sig.stop
                if blown:
                    self.emit("UPDATE", sig.symbol,
                              f"{sig.symbol} setup INVALIDATED before entry",
                              f"price {price:.{digits}f} broke the zone at "
                              f"{sig.stop:.{digits}f} without filling.\n"
                              f"No trade. Removing from the watchlist.",
                              {"side": "", "kind": "invalidated"})
                    del self.active[key]
                elif reached:
                    sig.status = "filled"
                    # From here on, only price after this instant counts.
                    sig.filled_ts = now
                    self.emit("UPDATE", sig.symbol,
                              f"{sig.symbol} ENTRY FILLED @ {sig.entry:.{digits}f}",
                              self._levels(sig, digits),
                              {"side": "BUY" if sig.is_long else "SELL",
                               "kind": "filled"})

            elif sig.status == "filled":
                if (best >= sig.tp1) if sig.is_long else (best <= sig.tp1):
                    sig.status = "tp1"
                    self.emit("UPDATE", sig.symbol,
                              f"{sig.symbol} TP1 HIT @ {sig.tp1:.{digits}f}",
                              f"+{self._pips(sig, sig.tp1):.0f} pips. Bank half.\n"
                              f"Consider moving the stop to entry "
                              f"({sig.entry:.{digits}f}).\n"
                              f"TP2 remains {sig.tp2:.{digits}f} "
                              f"(+{self._pips(sig, sig.tp2):.0f} pips).",
                              {"side": "", "kind": "tp1"})
                elif (worst <= sig.stop) if sig.is_long else (worst >= sig.stop):
                    self.emit("UPDATE", sig.symbol,
                              f"{sig.symbol} STOPPED OUT @ {sig.stop:.{digits}f}",
                              f"{self._pips(sig, sig.stop):.0f} pips. "
                              f"Full 1R loss, as planned.",
                              {"side": "", "kind": "stop"})
                    del self.active[key]

            elif sig.status == "tp1":
                if (best >= sig.tp2) if sig.is_long else (best <= sig.tp2):
                    self.emit("UPDATE", sig.symbol,
                              f"{sig.symbol} TP2 HIT @ {sig.tp2:.{digits}f}",
                              f"+{self._pips(sig, sig.tp2):.0f} pips. Trade complete.",
                              {"side": "", "kind": "tp2"})
                    del self.active[key]
                elif (worst <= sig.entry) if sig.is_long else (worst >= sig.entry):
                    self.emit("UPDATE", sig.symbol,
                              f"{sig.symbol} back to BREAKEVEN",
                              f"Price returned to entry {sig.entry:.{digits}f} "
                              f"after TP1. If your stop is at entry you are flat, "
                              f"with TP1 banked.",
                              {"side": "", "kind": "breakeven"})
                    del self.active[key]

    def _pips(self, sig: ActiveSignal, price: float) -> float:
        return (price - sig.entry) * sig.direction / pip_size(sig.symbol)

    # -- scanning for new setups ----------------------------------------------

    def scan(self, symbol: str, now: int) -> None:
        cfg = self.cfg.for_symbol(symbol)
        ex = cfg.execution

        entry = self.feed.bars(symbol, ex.entry_timeframe, BARS_ENTRY)
        if entry is None or len(entry) < 60:
            return

        # Examine EVERY entry bar that has closed since the last scan, not just
        # the newest one.
        #
        # A scheduled runner does not get to choose when it wakes: GitHub
        # delivers a 20-minute cron every 90-140 minutes under load, by which
        # point six to nine M15 bars have closed. Looking only at the latest
        # would leave the bot blind to the rest, and a zone touched by a wick
        # two bars ago would never be seen. The backtester walks every bar, so
        # this is also what parity requires.
        latest = int(entry.time[-1])
        last_seen = self._last_bar.get(symbol)
        if last_seen is None:
            start = len(entry) - 1          # first ever scan: just the newest
        elif last_seen >= latest:
            return                           # nothing new has closed
        else:
            start = int(np.searchsorted(entry.time, last_seen, "right"))
        # After a long outage, do not replay days of history looking for setups
        # that have long since resolved.
        start = max(start, len(entry) - MAX_CATCHUP_BARS)
        if start >= len(entry):
            return
        self._last_bar[symbol] = latest

        zone_bars = self.feed.bars(symbol, ex.zone_timeframe, BARS_ZONE)
        bias_bars = self.feed.bars(symbol, ex.bias_timeframe, BARS_BIAS)
        if zone_bars is None or bias_bars is None:
            return
        if len(zone_bars) < 120 or len(bias_bars) < 60:
            return

        zone_secs = tfmod.seconds(ex.zone_timeframe)
        bias_secs = tfmod.seconds(ex.bias_timeframe)

        # Zones are derived from zone-timeframe bars, so they only change when a
        # new one closes. Detect then, and age the result forward with each new
        # entry bar -- the same incremental model the backtester uses, and far
        # cheaper than rebuilding the map every 15 minutes.
        cached = self._analysis.get(symbol)
        zone_stamp = int(zone_bars.time[-1])
        bias_stamp = int(bias_bars.time[-1])

        # Zones are aged with every bar *before* the ones we are about to
        # examine. The bar that first reaches a zone is the entry, not a used-up
        # test -- the backtester evaluates entries against pre-bar state for
        # exactly this reason, and ageing through a bar before judging it would
        # retire every setup one moment before it could be signalled.
        prior = entry.slice(0, start)
        prior_stamp = int(prior.time[-1]) if len(prior) else 0

        if (cached is None or cached["zone_stamp"] != zone_stamp
                or cached["bias_stamp"] != bias_stamp):
            zone_struct = Structure(zone_bars, cfg.structure.swing_lookback)
            bias_struct = Structure(bias_bars, cfg.structure.swing_lookback)
            pool = settle_on(
                find_zones(zone_bars, cfg.zone, zone_struct), prior, zone_secs
            )
            bias_pool = settle_on(
                find_zones(bias_bars, cfg.zone, bias_struct), prior, bias_secs
            )
            # Zones are fresh objects; restore what we already spent.
            for z in pool:
                if self._zone_key(symbol, z) in self.signalled_keys:
                    z.traded = True
            cached = {
                "zone_stamp": zone_stamp, "bias_stamp": bias_stamp,
                "zone_struct": zone_struct, "bias_struct": bias_struct,
                "pool": pool, "bias_pool": bias_pool,
                "aged_to": prior_stamp,
            }
            self._analysis[symbol] = cached
        else:
            # Same zone map, newer entry bars: age it forward only.
            zone_struct = cached["zone_struct"]
            bias_struct = cached["bias_struct"]
            pool, bias_pool = cached["pool"], cached["bias_pool"]
            fresh = (prior.time > cached["aged_to"]) if len(prior) else None
            if fresh is not None and fresh.any():
                for k in np.flatnonzero(fresh):
                    update_all(pool, float(prior.high[k]), float(prior.low[k]),
                               float(prior.close[k]))
                    update_all(bias_pool, float(prior.high[k]), float(prior.low[k]),
                               float(prior.close[k]))
                cached["aged_to"] = prior_stamp

        last = len(zone_bars) - 1
        bias_live = [z for z in bias_pool if not z.invalidated]
        htf_trend = int(bias_struct.trend[-1])

        spread_points = self.feed.spread_points(symbol)
        spread_price = spread_points * sources.spec_for(symbol).point
        digits = sources.spec_for(symbol).digits
        pip = pip_size(symbol)

        # Same spread gate the backtester applies.
        if spread_points > ex.max_spread_points:
            return

        atr_series = atr_fn(entry.high, entry.low, entry.close)
        tradeable = in_session(now, ex) and not is_week_close(now, ex)

        # Walk each unexamined bar: judge it against pre-bar zone state, then
        # age the zones with it. Identical ordering to the backtester's loop.
        for k in range(start, len(entry)):
            self._evaluate_bar(
                symbol, cfg, entry, k, pool, bias_pool, bias_live, zone_struct,
                htf_trend, last, atr_series, spread_price, digits, pip, tradeable,
            )
            h, lo_, c = (float(entry.high[k]), float(entry.low[k]),
                         float(entry.close[k]))
            update_all(pool, h, lo_, c)
            update_all(bias_pool, h, lo_, c)
        cached["aged_to"] = latest

    def _evaluate_bar(
        self, symbol, cfg, entry, k, pool, bias_pool, bias_live, zone_struct,
        htf_trend, last, atr_series, spread_price, digits, pip, tradeable,
    ) -> None:
        """Look for a signal on one entry bar, using zone state as it stood
        before that bar printed."""
        # The backtest allows max_trades_per_symbol concurrent positions; mirror
        # that here so the signal count cannot drift above what was tested.
        live_here = sum(1 for s in self.active.values() if s.symbol == symbol)
        if live_here >= cfg.risk.max_trades_per_symbol:
            return

        live = [z for z in pool if z.is_live(cfg.zone, last)]
        if not live:
            return
        # The profit-margin rule needs every zone that could still block price,
        # not just the tradeable ones. A zone that has already been tested is
        # spent as an entry but still stands in the way as resistance, and the
        # backtester counts it -- so this list must match, or the two engines
        # disagree about which setups have room to run.
        obstacles = [z for z in pool if not z.invalidated and not z.traded]
        for z in live:
            apply_score(z, cfg, bias_live, htf_trend)

        price = float(entry.close[k])
        # The backtest fills a resting limit order at the proximal line, so a
        # setup counts as triggered when the bar's range *reached* the entry --
        # not only when it closed beyond it. Using the close would silently drop
        # every wick-fill and make live results diverge from the backtest.
        bar_high = float(entry.high[k])
        bar_low = float(entry.low[k])
        atr_value = float(atr_series[k])

        for zone in sorted(live, key=lambda z: -z.score):
            plan, why = build_plan(
                zone, cfg, k, atr_value,
                obstacles + bias_live, spread_price
            )
            if plan is None:
                continue

            key = self._zone_key(symbol, zone)
            if key in self.active:
                continue
            if self.cooldown.get(key, 0) > time.time():
                continue

            distance = abs(price - plan.entry)
            # Limit-order semantics, matching the backtester exactly.
            triggered = (
                bar_low <= plan.entry if plan.is_long else bar_high >= plan.entry
            )

            if triggered and tradeable:
                sig = ActiveSignal(
                    key=key, symbol=symbol, direction=plan.direction,
                    entry=plan.entry, stop=plan.stop, tp1=plan.tp1, tp2=plan.tp2,
                    score=zone.score, zone_note=plan.zone_note,
                    created=datetime.now(timezone.utc).isoformat(),
                    # The bar's range reached the entry, so a resting limit
                    # order is filled -- exactly as the backtester models it.
                    # Leaving this "pending" would keep the signal open forever
                    # whenever price never closed back through the level.
                    status="filled",
                    created_ts=now,
                    # Anchor the exit search here. Without it, targets could be
                    # reported hit by price that traded before we were filled.
                    filled_ts=now,
                )
                self.active[key] = sig
                self.all_signalled.append(sig)
                # Spent for good, matching zone.traded in the backtester.
                self.signalled_keys.add(key)
                zone.traded = True
                self.cooldown[key] = time.time() + cfg.alerts.cooldown_minutes * 60
                self.emit(
                    "TRIGGERED", symbol,
                    f"{'BUY' if plan.is_long else 'SELL'} {symbol} @ "
                    f"{plan.entry:.{digits}f}",
                    "Your pending order at this level has filled.\n"
                    "(If you did not place one, the level has now been reached "
                    "and the entry is gone -- do not chase it.)\n\n"
                    + self._full_alert(cfg, symbol, zone, plan, htf_trend,
                                       zone_struct, last, digits, pip),
                    {"side": "BUY" if plan.is_long else "SELL", "kind": "triggered",
                     "score": round(zone.score), "symbol": symbol},
                )
                return  # one new signal per symbol per bar

            if (cfg.alerts.alert_on_approach
                    and distance <= cfg.alerts.approach_atr * atr_value):
                if key in self.ordered_keys:
                    continue          # already told you about this level
                self.ordered_keys.add(key)
                side = "BUY LIMIT" if plan.is_long else "SELL LIMIT"
                lots, note = self._size(cfg, symbol, plan.risk_distance)
                self.emit(
                    "SET ORDER", symbol,
                    f"{side} {symbol} @ {plan.entry:.{digits}f}",
                    f"Place this now as a PENDING order. It fills by itself when\n"
                    f"price arrives -- which is how the strategy was tested.\n\n"
                    f"TYPE    {side}\n"
                    f"ENTRY   {plan.entry:.{digits}f}   "
                    f"({distance / pip:.0f} pips away)\n"
                    f"STOP    {plan.stop:.{digits}f}   "
                    f"({plan.risk_distance / pip:.0f} pips risk)\n"
                    f"TP1     {plan.tp1:.{digits}f}   close "
                    f"{cfg.signal.tp1_fraction:.0%} at {cfg.signal.tp1_r:.2f}R\n"
                    f"TP2     {plan.tp2:.{digits}f}   ({plan.rr2:.1f}R)\n\n"
                    f"SIZE    {lots}\n"
                    f"        {note}\n\n"
                    f"ZONE    {zone.pattern} grade {grade(zone.score)} "
                    f"({zone.score:.0f}/100), departure "
                    f"{zone.departure_ratio:.1f}x\n\n"
                    f"If price never reaches it, cancel the order. You will get a\n"
                    f"follow-up either way."
                    + ("" if tradeable else
                       "\n\nNOTE: outside session hours right now."),
                    {"side": "BUY" if plan.is_long else "SELL",
                     "kind": "set_order", "symbol": symbol,
                     "score": round(zone.score)},
                )
                return

    # -- message building -----------------------------------------------------

    def _full_alert(self, cfg, symbol, zone, plan, htf_trend,
                    zone_struct, last, digits, pip) -> str:
        risk_pips = plan.risk_distance / pip
        tp1_pips = abs(plan.tp1 - plan.entry) / pip
        tp2_pips = abs(plan.tp2 - plan.entry) / pip
        curve = zone_struct.curve_position(last, plan.entry)

        lots, note = self._size(cfg, symbol, plan.risk_distance)
        why = [
            f"{zone.pattern} zone, grade {grade(zone.score)} ({zone.score:.0f}/100)",
            f"departure {zone.departure_ratio:.1f}x zone height",
            f"{cfg.execution.bias_timeframe} trend {trend_name(htf_trend)}",
            f"{'discount' if curve < 0.5 else 'premium'} ({curve:.0%} of range)",
        ]
        if zone.caused_bos:
            why.append("departure broke structure")
        if zone.has_imbalance:
            why.append("unfilled FVG in the leg out")
        if zone.tests == 0:
            why.append("zone untested")

        return (
            f"ENTRY   {plan.entry:.{digits}f}\n"
            f"STOP    {plan.stop:.{digits}f}   ({risk_pips:.0f} pips risk)\n"
            f"TP1     {plan.tp1:.{digits}f}   (+{tp1_pips:.0f} pips, "
            f"{cfg.signal.tp1_r:.1f}R) close {cfg.signal.tp1_fraction:.0%}\n"
            f"TP2     {plan.tp2:.{digits}f}   (+{tp2_pips:.0f} pips, "
            f"{plan.rr2:.1f}R)\n\n"
            f"SIZE    {lots}\n"
            f"        {note}\n\n"
            f"WHY\n" + "\n".join(f"  - {w}" for w in why) + "\n\n"
            f"Manage: bank {cfg.signal.tp1_fraction:.0%} at TP1"
            + (", then stop to entry." if cfg.signal.breakeven_after_tp1
               else ", let the rest run to TP2.")
        )

    def _size(self, cfg, symbol: str, risk_distance: float) -> tuple[str, str]:
        equity = cfg.backtest.initial_balance
        pct = cfg.risk.risk_per_trade_pct
        spec = sources.spec_for(symbol)
        loss_per_lot = spec.money_per_lot(risk_distance)
        if loss_per_lot <= 0:
            return "n/a", ""
        amount = equity * pct / 100.0
        lots = amount / loss_per_lot
        step = spec.volume_step or 0.01
        lots = max(step, int(lots / step) * step)
        return (
            f"{lots:.2f} lots",
            f"risks {amount:,.0f} ({pct}% of {equity:,.0f}). "
            f"Scale to your own balance.",
        )

    def _levels(self, sig: ActiveSignal, digits: int) -> str:
        return (
            f"ENTRY   {sig.entry:.{digits}f}\n"
            f"STOP    {sig.stop:.{digits}f}\n"
            f"TP1     {sig.tp1:.{digits}f}\n"
            f"TP2     {sig.tp2:.{digits}f}\n\n"
            f"You are in the trade. Manage to plan."
        )

    # -- housekeeping ---------------------------------------------------------

    def emit(self, kind: str, symbol: str, subject: str, body: str,
             meta: dict | None = None) -> None:
        """Send an alert. Raises if it did not reach the user.

        Console and file always "succeed" -- they are a local log, not a way of
        reaching someone who is not at the machine. Treating them as delivery
        let a broken Telegram token look like a working bot: alerts were
        recorded as sent, the zone was marked as already-ordered, and nothing
        ever arrived. Silence is the one failure mode this system cannot afford,
        so a remote channel failing is now an error, not a log line.
        """
        meta = {"kind": kind.lower(), "symbol": symbol, **(meta or {})}
        results = self.out.send(f"[{kind}] {subject}", body, meta)

        local = {"console", "file"}
        remote = {n: ok for n, ok in results.items() if n not in local}
        dead = [n for n, ok in results.items() if not ok]
        if dead:
            print(f"[{_now()}] delivery failed on: {', '.join(dead)}", flush=True)

        # If remote channels are configured, at least one has to have worked.
        if remote and not any(remote.values()):
            raise DeliveryError(
                f"alert '{subject}' reached no remote channel "
                f"({', '.join(sorted(remote))} all failed). "
                f"Check TELEGRAM_TOKEN / TELEGRAM_CHAT_ID."
            )

    def startup_report(self) -> None:
        ex = self.cfg.execution
        print(f"\n[{_now()}] signal scanner starting")
        print(f"  feed        {self.feed.backend}")
        print(f"  watching    {', '.join(ex.symbols)}")
        print(f"  timeframes  bias={ex.bias_timeframe} zones={ex.zone_timeframe} "
              f"entry={ex.entry_timeframe}")
        print(f"  session     {min(ex.session_hours_utc):02d}:00-"
              f"{max(ex.session_hours_utc):02d}:59 UTC")
        print(f"  poll        every {self.cfg.alerts.poll_seconds}s")
        print("  channels:")
        for line in self.out.preflight():
            print(line)
        if self.active:
            print(f"  resumed {len(self.active)} signal(s) already in flight")
        print(flush=True)

    def heartbeat(self) -> None:
        hours = self.cfg.alerts.heartbeat_hours
        if hours <= 0:
            return
        if time.time() - self._last_heartbeat < hours * 3600:
            return
        self._last_heartbeat = time.time()
        lines = [f"watching {len(self.cfg.execution.symbols)} instruments"]
        if self.active:
            for sig in self.active.values():
                lines.append(f"  {sig.symbol} {'BUY' if sig.is_long else 'SELL'} "
                             f"{sig.status}")
        else:
            lines.append("no live signals")
        self.emit("STATUS", "", "Signal bot alive", "\n".join(lines),
                  {"kind": "heartbeat"})


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")
