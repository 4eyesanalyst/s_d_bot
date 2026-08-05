"""Live / demo execution against a MetaTrader 5 terminal.

Cadence mirrors how the strategy is actually traded by hand:

* On each **closed entry-timeframe bar**, redraw the map -- structure, zones,
  scores -- and rest a limit order at the proximal line of any zone that earns
  one. Set and forget.
* On **every poll**, manage what is already open: bank the first partial at
  TP1, move the stop to breakeven, then trail behind structure.

Nothing is decided on a bar that is still forming.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from . import broker
from . import timeframes as tfmod
from .config import Config
from .data import recent
from .indicators import atr as atr_fn
from .journal import Journal
from .risk import RiskManager
from .scoring import apply_score, explain
from .sessions import in_session, is_week_close
from .signals import build_plan
from .structure import Structure, trend_name
from .trades import LONG, SHORT, TradePlan
from .zones import Zone, find_zones, settle

# How much history each timeframe needs for structure and zones to be stable.
BARS_ENTRY = 600
BARS_ZONE = 800
BARS_BIAS = 400


class LiveTrader:
    def __init__(self, cfg: Config, journal: Journal, dry_run: bool = False):
        self.cfg = cfg
        self.journal = journal
        self.dry_run = dry_run
        broker.ensure_initialized()
        self.account = broker.account_summary()
        self.risk = RiskManager(cfg.risk, self.account["equity"])
        self._last_bar: dict[str, int] = {}
        self._plans: dict[int, TradePlan] = {}
        self._specs: dict[str, object] = {}

    def spec(self, symbol: str):
        """Cached contract spec -- these do not change during a session."""
        if symbol not in self._specs:
            self._specs[symbol] = broker.spec_for(symbol)
        return self._specs[symbol]

    # -- reporting -----------------------------------------------------------

    def announce(self) -> None:
        a = self.account
        mode = "DRY RUN (no orders sent)" if self.dry_run else (
            "DEMO" if a["is_demo"] else "*** LIVE MONEY ***"
        )
        self.journal.log(
            f"connected: {a['login']}@{a['server']} {a['currency']} "
            f"equity={a['equity']:,.2f} leverage=1:{a['leverage']} [{mode}]"
        )
        ex = self.cfg.execution
        self.journal.log(
            f"symbols={','.join(ex.symbols)} bias={ex.bias_timeframe} "
            f"zones={ex.zone_timeframe} entry={ex.entry_timeframe} "
            f"risk={self.cfg.risk.risk_per_trade_pct}%/trade "
            f"min_score={self.cfg.zone.min_score} minRR={self.cfg.signal.min_risk_reward}"
        )
        if not a["trade_allowed"] and not self.dry_run:
            self.journal.log(
                "WARNING: algo trading is disabled in the terminal. Enable the "
                "'Algo Trading' button in MT5 or no order will go through."
            )

    # -- main loop -----------------------------------------------------------

    def run(self) -> None:
        self.announce()
        try:
            while True:
                try:
                    self.step()
                except Exception as exc:  # keep the session alive on transient errors
                    self.journal.log(f"ERROR in step: {exc!r}")
                time.sleep(self.cfg.execution.poll_seconds)
        except KeyboardInterrupt:
            self.journal.log("stopped by user")
        finally:
            broker.shutdown()

    def step(self) -> None:
        equity = broker.account_equity()
        now = int(datetime.now(timezone.utc).timestamp())
        self.risk.on_time(now, equity)

        self.manage_open_positions()
        self.prune_stale_orders()

        if is_week_close(now, self.cfg.execution):
            self.journal.log("weekend window: flattening and standing down")
            self.flatten_all("weekend")
            return

        for symbol in self.cfg.execution.symbols:
            if self._new_bar(symbol):
                self.scan(symbol)

    def _new_bar(self, symbol: str) -> bool:
        """True once per closed entry-timeframe bar, per symbol."""
        bars = recent(symbol, self.cfg.execution.entry_timeframe, 2)
        if len(bars) == 0:
            return False
        latest = int(bars.time[-1])
        if self._last_bar.get(symbol) == latest:
            return False
        self._last_bar[symbol] = latest
        return True

    # -- analysis ------------------------------------------------------------

    def scan(self, symbol: str) -> None:
        cfg = self.cfg
        ex = cfg.execution

        entry = recent(symbol, ex.entry_timeframe, BARS_ENTRY)
        zone_bars = recent(symbol, ex.zone_timeframe, BARS_ZONE)
        bias_bars = recent(symbol, ex.bias_timeframe, BARS_BIAS)
        if len(zone_bars) < 100 or len(bias_bars) < 60:
            self.journal.log(f"{symbol}: not enough history yet, skipping")
            return

        zone_struct = Structure(zone_bars, cfg.structure.swing_lookback)
        bias_struct = Structure(bias_bars, cfg.structure.swing_lookback)

        zone_pool = settle(find_zones(zone_bars, cfg.zone, zone_struct), zone_bars)
        bias_pool = settle(find_zones(bias_bars, cfg.zone, bias_struct), bias_bars)

        last_zi = len(zone_bars) - 1
        live_zones = [z for z in zone_pool if z.is_live(cfg.zone, last_zi)]
        bias_live = [z for z in bias_pool if not z.invalidated]
        htf_trend = int(bias_struct.trend[-1])

        for z in live_zones:
            apply_score(z, cfg, bias_live, htf_trend)
        live_zones.sort(key=lambda z: -z.score)

        best = f" best={live_zones[0].score:.0f}" if live_zones else ""
        self.journal.log(
            f"{symbol} {ex.bias_timeframe} trend={trend_name(htf_trend)} "
            f"curve={zone_struct.curve_position(last_zi, float(entry.close[-1])):.0%} "
            f"| {len(live_zones)} live zones{best}"
        )

        # Trail any open position on this symbol, on the same cadence the
        # backtester uses.
        self.trail(symbol, zone_struct, len(zone_bars) - 1, entry)

        now = int(datetime.now(timezone.utc).timestamp())
        if not in_session(now, ex):
            return

        # One working order or position per symbol.
        positions = broker.open_positions(ex.magic_number)
        if any(p.symbol == symbol for p in positions):
            return
        if any(o.symbol == symbol for o in broker.pending_orders(ex.magic_number)):
            return

        spread_points = broker.current_spread_points(symbol)
        if spread_points > ex.max_spread_points:
            self.journal.log(
                f"{symbol}: spread {spread_points:.0f}pts over "
                f"{ex.max_spread_points}, standing aside"
            )
            return

        spec = self.spec(symbol)
        atr_value = float(atr_fn(entry.high, entry.low, entry.close)[-1])
        spread_price = spread_points * spec.point

        for zone in live_zones:
            plan, why = build_plan(
                zone, cfg, len(entry) - 1, atr_value,
                live_zones + bias_live, spread_price,
            )
            if plan is None:
                continue

            veto = self.risk.veto(
                symbol, plan.direction, self._positions_as_objects()
            )
            if veto:
                self.journal.log(f"{symbol}: setup found but vetoed -- {veto}")
                return

            volume, risk_amount = self.risk.volume_for(plan.risk_distance, spec)
            if volume <= 0:
                self.journal.log(
                    f"{symbol}: correct size is below the {spec.volume_min} lot "
                    f"minimum at {self.risk.risk_pct():.2f}% risk -- skipping "
                    f"rather than over-risking"
                )
                return

            self.place(symbol, plan, zone, volume, risk_amount)
            return

    # -- execution -----------------------------------------------------------

    def place(
        self, symbol: str, plan: TradePlan, zone: Zone, volume: float, risk_amount: float
    ) -> None:
        ex = self.cfg.execution
        expiry = datetime.now(timezone.utc) + timedelta(
            minutes=tfmod.minutes(ex.entry_timeframe) * self.cfg.signal.pending_expiry_bars
        )
        summary = (
            f"{symbol} {'BUY' if plan.is_long else 'SELL'} LIMIT {volume:.2f} @ "
            f"{plan.entry:.5f} sl={plan.stop:.5f} tp1={plan.tp1:.5f} "
            f"tp2={plan.tp2:.5f} ({plan.rr2:.1f}R) risk={risk_amount:,.2f}"
        )
        self.journal.log(f"SETUP {summary}")
        self.journal.log(f"      {explain(zone)}")

        if self.dry_run:
            self.journal.event("dry_run_order", symbol=symbol, plan=vars(plan))
            return

        ok, message, ticket = broker.send_pending_limit(
            symbol=symbol,
            direction=plan.direction,
            volume=volume,
            price=plan.entry,
            stop=plan.stop,
            take_profit=plan.tp2,
            magic=ex.magic_number,
            comment=f"{ex.comment} {zone.pattern}",
            expiry=expiry,
        )
        self.journal.log(("PLACED " + message) if ok else ("REJECTED " + message))
        if ok:
            self._plans[ticket] = plan
            self.journal.event(
                "order_placed", ticket=ticket, symbol=symbol,
                plan=vars(plan), volume=volume, risk=risk_amount,
                zone=zone.describe(),
            )

    def manage_open_positions(self) -> None:
        """Bank TP1, move to breakeven, then trail behind structure."""
        ex = self.cfg.execution
        sig = self.cfg.signal
        for p in broker.open_positions(ex.magic_number):
            import MetaTrader5 as mt5

            direction = LONG if p.type == mt5.POSITION_TYPE_BUY else SHORT
            if p.sl == 0:
                continue  # no stop: nothing safe to infer, leave it alone
            risk = abs(p.price_open - p.sl)
            if risk <= 0:
                continue

            plan = self._plans.get(p.ticket)
            tp1 = (
                plan.tp1 if plan
                else p.price_open + direction * sig.tp1_r * risk
            )
            reached = (
                p.price_current >= tp1 if direction == LONG
                else p.price_current <= tp1
            )
            beyond_be = (
                (p.sl - p.price_open) * direction >= 0
            )
            if not reached or beyond_be:
                continue

            # First partial.
            chunk = round(p.volume * sig.tp1_fraction, 2)
            spec = self.spec(p.symbol)
            if chunk >= spec.volume_min and p.volume - chunk >= spec.volume_min:
                ok, msg = broker.close_partial(
                    p.ticket, chunk, ex.magic_number, ex.deviation_points,
                    f"{ex.comment} tp1",
                )
                self.journal.log(("TP1 " + msg) if ok else ("TP1 failed: " + msg))

            if sig.breakeven_after_tp1:
                initial_risk = abs(p.price_open - p.sl)
                be = p.price_open + direction * sig.breakeven_offset_r * initial_risk
                ok, msg = broker.modify_stop(p.ticket, be, p.tp)
                self.journal.log(
                    (f"{p.symbol} breakeven: " + msg) if ok else ("BE failed: " + msg)
                )

    def trail(self, symbol: str, zone_struct: Structure, zi: int, entry) -> None:
        """Ratchet the stop behind structure once the first partial is banked.

        Only ever moves the stop in our favour, and only after TP1 -- trailing
        a full-size position out of a good zone is how a 5R trade becomes 0.3R.
        """
        import MetaTrader5 as mt5

        sig = self.cfg.signal
        if not sig.trail_after_tp1:
            return

        buffer = sig.stop_buffer_atr * float(
            atr_fn(entry.high, entry.low, entry.close)[-1]
        )
        for p in broker.open_positions(self.cfg.execution.magic_number):
            if p.symbol != symbol or p.sl == 0:
                continue
            direction = LONG if p.type == mt5.POSITION_TYPE_BUY else SHORT
            # Only trail once the stop is already at or past breakeven.
            if (p.sl - p.price_open) * direction < 0:
                continue

            if direction == LONG:
                swing = zone_struct.recent_swing_low_price(zi)
                new_stop = swing - buffer if swing is not None else None
                better = new_stop is not None and new_stop > p.sl
            else:
                swing = zone_struct.recent_swing_high_price(zi)
                new_stop = swing + buffer if swing is not None else None
                better = new_stop is not None and new_stop < p.sl

            # Never trail past the current price.
            if better and (p.price_current - new_stop) * direction <= 0:
                better = False

            if better:
                ok, msg = broker.modify_stop(p.ticket, new_stop, p.tp)
                self.journal.log(
                    f"{symbol} trail: {msg}" if ok else f"{symbol} trail failed: {msg}"
                )

    def prune_stale_orders(self) -> None:
        """Cancel resting orders whose zone has since been blown through."""
        ex = self.cfg.execution
        for o in broker.pending_orders(ex.magic_number):
            import MetaTrader5 as mt5

            tick = mt5.symbol_info_tick(o.symbol)
            if tick is None or o.sl == 0:
                continue
            is_buy = o.type == mt5.ORDER_TYPE_BUY_LIMIT
            price = tick.bid if is_buy else tick.ask
            dead = price < o.sl if is_buy else price > o.sl
            if dead:
                ok, msg = broker.cancel_order(o.ticket)
                self.journal.log(
                    f"{o.symbol}: zone invalidated before entry -- {msg}"
                    if ok else msg
                )
                self._plans.pop(o.ticket, None)

    def flatten_all(self, reason: str) -> None:
        ex = self.cfg.execution
        for o in broker.pending_orders(ex.magic_number):
            broker.cancel_order(o.ticket)
        for p in broker.open_positions(ex.magic_number):
            ok, msg = broker.close_partial(
                p.ticket, p.volume, ex.magic_number, ex.deviation_points,
                f"{ex.comment} {reason}",
            )
            self.journal.log(f"flatten {p.symbol}: {msg}")

    # -- helpers -------------------------------------------------------------

    def _positions_as_objects(self) -> list:
        """Adapt MT5 positions to what RiskManager expects."""
        import MetaTrader5 as mt5

        from .trades import Position

        out = []
        for p in broker.open_positions(self.cfg.execution.magic_number):
            direction = LONG if p.type == mt5.POSITION_TYPE_BUY else SHORT
            risk = abs(p.price_open - p.sl) if p.sl else 0.0
            spec = self.spec(p.symbol)
            plan = self._plans.get(p.ticket)
            out.append(
                Position(
                    symbol=p.symbol,
                    direction=direction,
                    entry_price=p.price_open,
                    entry_time=int(p.time),
                    entry_index=0,
                    volume=p.volume,
                    stop=p.sl,
                    tp1=plan.tp1 if plan else p.tp,
                    tp2=p.tp,
                    initial_stop=p.sl,
                    risk_amount=spec.money_per_lot(risk) * p.volume,
                    plan=plan or TradePlan(
                        symbol=p.symbol, direction=direction, entry=p.price_open,
                        stop=p.sl, tp1=p.tp, tp2=p.tp, risk_distance=risk,
                        rr1=0, rr2=0, score=0, zone_id=0, zone_note="recovered",
                        created_index=0, expires_index=0,
                    ),
                    remaining=p.volume,
                )
            )
        return out
