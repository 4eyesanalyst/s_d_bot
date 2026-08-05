"""Event-driven, portfolio-level backtester.

Design commitments, because a backtest that flatters the strategy is worse than
no backtest at all:

* **No lookahead.** Higher-timeframe context is aligned with
  :func:`sd_bot.data.align`, which only exposes bars that had already closed.
  Zones only become tradeable once their departure leg has completed *and* that
  bar has closed.
* **Pessimistic fills.** When a bar contains both the stop and a target, the
  stop wins. Same-bar stop-outs on freshly opened trades are checked.
* **Real costs.** Recorded spread, configurable slippage, and round-turn
  commission are charged on every unit of volume closed.
* **One account.** All symbols share equity, so the risk caps and the currency
  exposure limit bite exactly as they would live.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from . import timeframes as tfmod
from .broker import estimate_spec
from .config import Config
from .data import Bars, align
from .indicators import atr as atr_fn
from .risk import RiskManager
from .scoring import apply_score
from .sessions import in_session, is_week_close
from .signals import build_plan, confirmed_entry
from .structure import Structure
from .trades import LONG, ClosedTrade, Position, SymbolSpec
from .zones import DEMAND, Zone, find_zones, update_all


@dataclass
class SymbolState:
    symbol: str
    cfg: Config          # base config with this symbol's overrides applied
    spec: SymbolSpec
    entry: Bars
    zone: Bars
    bias: Bars
    zone_struct: Structure
    bias_struct: Structure
    zone_pool: list[Zone]
    bias_pool: list[Zone]
    entry_atr: np.ndarray
    e2z: np.ndarray
    e2b: np.ndarray
    zone_ptr: int = 0
    bias_ptr: int = 0
    active: list[Zone] = field(default_factory=list)
    bias_active: list[Zone] = field(default_factory=list)
    last_close: float = 0.0


@dataclass
class Result:
    trades: list[ClosedTrade]
    equity_curve: list[tuple[int, float]]
    starting_balance: float
    ending_balance: float
    veto_counts: dict[str, int]
    rejected_plans: dict[str, int]
    zones_found: dict[str, int]
    bars_processed: int


def analysis_key(cfg: Config) -> tuple:
    """Identity of everything that affects structure and zone *detection*.

    Two configs sharing this key produce identical zones, so a parameter sweep
    over entry/risk settings can reuse the expensive analysis. Scoring weights
    are excluded on purpose: scores are recomputed at decision time.
    """
    z, st, ex = cfg.zone, cfg.structure, cfg.execution
    return (
        z.min_base_candles, z.max_base_candles, z.base_body_atr, z.base_body_ratio,
        z.leg_body_atr, z.leg_body_ratio, z.leg_lookahead, z.min_departure_ratio,
        z.max_zone_atr, z.min_zone_atr,
        st.swing_lookback,
        ex.zone_timeframe, ex.bias_timeframe, ex.entry_timeframe,
    )


class Backtester:
    def __init__(
        self,
        cfg: Config,
        specs: dict[str, SymbolSpec] | None = None,
        analysis_cache: dict | None = None,
    ):
        self.cfg = cfg
        self.specs = specs or {}
        # Optional cross-run cache of structures and zone pools. Zones carry
        # mutable per-run state, so cached pools are always deep-copied out.
        self.analysis_cache = analysis_cache
        self.balance = cfg.backtest.initial_balance
        self.risk = RiskManager(cfg.risk, self.balance)
        self.positions: list[Position] = []
        self._by_symbol: dict[str, list[Position]] = {}
        self.closed: list[ClosedTrade] = []
        self.equity_curve: list[tuple[int, float]] = []
        self.rejected: dict[str, int] = {}
        self._states: dict[str, SymbolState] = {}

    # -- setup ---------------------------------------------------------------

    def prepare(self, symbol: str, series: dict[str, Bars]) -> SymbolState:
        # Volatility-dependent settings differ per instrument; resolve them once.
        cfg = self.cfg.for_symbol(symbol)
        ex = cfg.execution
        entry = series[ex.entry_timeframe]
        zone = series[ex.zone_timeframe]
        bias = series[ex.bias_timeframe]

        cached = None
        if self.analysis_cache is not None:
            key = (symbol, id(entry), analysis_key(cfg))
            cached = self.analysis_cache.get(key)

        if cached is None:
            zone_struct = Structure(zone, cfg.structure.swing_lookback)
            bias_struct = Structure(bias, cfg.structure.swing_lookback)
            zone_pool = find_zones(zone, cfg.zone, zone_struct)
            bias_pool = find_zones(bias, cfg.zone, bias_struct)
            entry_atr = atr_fn(entry.high, entry.low, entry.close)
            e2z, e2b = align(zone, entry), align(bias, entry)
            if self.analysis_cache is not None:
                # Store a pristine template. The run below mutates its zones
                # (tests, invalidated, traded), so the cache must never hold the
                # same objects the run is about to consume.
                self.analysis_cache[key] = (
                    zone_struct, bias_struct,
                    copy.deepcopy(zone_pool), copy.deepcopy(bias_pool),
                    entry_atr, e2z, e2b,
                )
        else:
            zone_struct, bias_struct, zone_pool, bias_pool, entry_atr, e2z, e2b = cached
            # Structures are immutable, zones are not: hand this run its own.
            zone_pool = copy.deepcopy(zone_pool)
            bias_pool = copy.deepcopy(bias_pool)

        state = SymbolState(
            symbol=symbol,
            cfg=cfg,
            spec=self.specs.get(symbol) or estimate_spec(symbol),
            entry=entry,
            zone=zone,
            bias=bias,
            zone_struct=zone_struct,
            bias_struct=bias_struct,
            zone_pool=zone_pool,
            bias_pool=bias_pool,
            entry_atr=entry_atr,
            e2z=e2z,
            e2b=e2b,
            last_close=float(entry.close[0]) if len(entry) else 0.0,
        )
        self._states[symbol] = state
        return state

    # -- main loop -----------------------------------------------------------

    def run(self, data: dict[str, dict[str, Bars]]) -> Result:
        for symbol, series in data.items():
            self.prepare(symbol, series)

        events: list[tuple[int, str, int]] = []
        for symbol, st in self._states.items():
            for i in range(len(st.entry)):
                events.append((int(st.entry.time[i]), symbol, i))
        events.sort()

        entry_secs = tfmod.seconds(self.cfg.execution.entry_timeframe)
        last_sample = -1

        for ts, symbol, i in events:
            st = self._states[symbol]
            st.last_close = float(st.entry.close[i])
            decision_time = ts + entry_secs

            equity = self.equity()
            self.risk.on_time(ts, equity)
            # One equity sample per timestamp, not per symbol per timestamp.
            if ts != last_sample:
                self.equity_curve.append((ts, equity))
                last_sample = ts

            # 1. existing trades meet this bar first.
            self._manage(st, i, ts)

            # 2. zones whose departure has completed become visible.
            self._activate(st, decision_time)

            # 3. look for a new entry using zone state as it stood *before*
            #    this bar -- the first touch is the entry, not a used-up test.
            self._try_entry(st, i, ts)

            # 4. now age the zones with this bar.
            bar_h = float(st.entry.high[i])
            bar_l = float(st.entry.low[i])
            bar_c = float(st.entry.close[i])
            update_all(st.active, bar_h, bar_l, bar_c)
            update_all(st.bias_active, bar_h, bar_l, bar_c)
            zi = int(st.e2z[i])
            max_age = st.cfg.zone.max_zone_age_bars
            st.active = [
                z for z in st.active
                if not z.invalidated and not z.traded and zi - z.created <= max_age
            ]
            st.bias_active = [z for z in st.bias_active if not z.invalidated]

            # 5. no weekend gap risk.
            if is_week_close(ts, st.cfg.execution):
                self._flatten(st, i, ts, "weekend")

        # Close anything still open at the end of the data.
        for symbol, st in list(self._states.items()):
            self._flatten(st, len(st.entry) - 1, int(st.entry.time[-1]), "end of data")

        return Result(
            trades=self.closed,
            equity_curve=self.equity_curve,
            starting_balance=self.cfg.backtest.initial_balance,
            ending_balance=self.balance,
            veto_counts=dict(self.risk.veto_counts),
            rejected_plans=dict(self.rejected),
            zones_found={s: len(st.zone_pool) for s, st in self._states.items()},
            bars_processed=len(events),
        )

    # -- accounting ----------------------------------------------------------

    def equity(self) -> float:
        floating = 0.0
        for p in self.positions:
            st = self._states[p.symbol]
            diff = (st.last_close - p.entry_price) * p.direction
            floating += diff / st.spec.tick_size * st.spec.tick_value * p.remaining
        return self.balance + floating

    def _chunk_cost(self, spec: SymbolSpec, volume: float, spread_points: float) -> float:
        bt = self.cfg.backtest
        points = spread_points + 2.0 * bt.slippage_points
        spread_cost = points * spec.point / spec.tick_size * spec.tick_value * volume
        return spread_cost + bt.commission_per_lot * volume

    # -- zone lifecycle ------------------------------------------------------

    def _activate(self, st: SymbolState, decision_time: int) -> None:
        zone_secs = tfmod.seconds(st.zone.timeframe)
        while st.zone_ptr < len(st.zone_pool):
            z = st.zone_pool[st.zone_ptr]
            if z.created_time + zone_secs > decision_time:
                break
            st.active.append(z)
            st.zone_ptr += 1

        bias_secs = tfmod.seconds(st.bias.timeframe)
        while st.bias_ptr < len(st.bias_pool):
            z = st.bias_pool[st.bias_ptr]
            if z.created_time + bias_secs > decision_time:
                break
            st.bias_active.append(z)
            st.bias_ptr += 1

    # -- entries -------------------------------------------------------------

    def _try_entry(self, st: SymbolState, i: int, ts: int) -> None:
        cfg = st.cfg
        if not in_session(ts, cfg.execution) or is_week_close(ts, cfg.execution):
            return
        spread_points = float(st.entry.spread[i])
        if spread_points > cfg.execution.max_spread_points:
            self._reject("spread too wide")
            return

        zi = int(st.e2z[i])
        bi = int(st.e2b[i])
        if zi < 0 or bi < 0:
            return

        o = float(st.entry.open[i])
        h = float(st.entry.high[i])
        low = float(st.entry.low[i])
        c = float(st.entry.close[i])
        atr_value = float(st.entry_atr[i])
        spread_price = spread_points * st.spec.point
        htf_trend = int(st.bias_struct.trend[bi])

        # Best-scoring live zone that this bar actually reached.
        candidates: list[tuple[float, Zone]] = []
        for z in st.active:
            # Two float compares reject the overwhelming majority of bars.
            if low > z.proximal if z.kind == DEMAND else h < z.proximal:
                continue
            if not z.is_live(cfg.zone, zi):
                continue
            apply_score(z, cfg, st.bias_active, htf_trend)
            candidates.append((z.score, z))

        if not candidates:
            return
        candidates.sort(key=lambda kv: -kv[0])

        for _, zone in candidates:
            plan, why = build_plan(
                zone, cfg, i, atr_value, st.active + st.bias_active, spread_price
            )
            if plan is None:
                self._reject(why)
                continue

            if cfg.signal.entry_mode == "limit":
                fill = plan.entry
                reached = low <= fill if plan.is_long else h >= fill
                if not reached:
                    continue
            else:
                if not confirmed_entry(zone, o, h, low, c):
                    continue
                fill = c
                # Re-derive risk from the actual fill: a confirmation entry is
                # further from the stop than the resting-limit price.
                plan.risk_distance = abs(fill - plan.stop)
                if plan.risk_distance <= 0:
                    continue
                rr2 = (plan.tp2 - fill) * plan.direction / plan.risk_distance
                if rr2 < cfg.signal.min_risk_reward:
                    self._reject(f"confirmation entry leaves only {rr2:.1f}R")
                    continue
                plan.rr2 = rr2
                plan.entry = fill
                plan.tp1 = fill + plan.direction * cfg.signal.tp1_r * plan.risk_distance

            veto = self.risk.veto(st.symbol, plan.direction, self.positions)
            if veto:
                continue

            volume, risk_amount = self.risk.volume_for(
                abs(fill - plan.stop), st.spec
            )
            if volume <= 0:
                self._reject("size below broker minimum lot")
                continue

            pos = Position(
                symbol=st.symbol,
                direction=plan.direction,
                entry_price=fill,
                entry_time=ts,
                entry_index=i,
                volume=volume,
                stop=plan.stop,
                tp1=plan.tp1,
                tp2=plan.tp2,
                initial_stop=plan.stop,
                risk_amount=risk_amount,
                plan=plan,
            )
            self.positions.append(pos)
            self._by_symbol.setdefault(st.symbol, []).append(pos)
            zone.traded = True

            # Entry cost is charged when the position finally closes, together
            # with the exit cost, so a scale-out is not double-charged.
            self._process(st, pos, i, ts, opened_this_bar=True)
            return  # one new position per symbol per bar

    def _reject(self, reason: str) -> None:
        key = reason.split(" (")[0]
        # Collapse the numeric detail so the summary stays readable.
        for prefix in ("only ", "target only ", "score ", "stop only ",
                       "confirmation entry leaves "):
            if key.startswith(prefix):
                key = prefix.strip() + " <n>"
                break
        if key.startswith("stop ") and "too wide" in key:
            key = "stop too wide"
        self.rejected[key] = self.rejected.get(key, 0) + 1

    # -- position management -------------------------------------------------

    def _manage(self, st: SymbolState, i: int, ts: int) -> None:
        book = self._by_symbol.get(st.symbol)
        if not book:
            return
        for pos in list(book):
            self._process(st, pos, i, ts)

    def _process(
        self,
        st: SymbolState,
        pos: Position,
        i: int,
        ts: int,
        opened_this_bar: bool = False,
    ) -> None:
        cfg = st.cfg.signal
        h = float(st.entry.high[i])
        low = float(st.entry.low[i])
        r = pos.risk_distance
        if r <= 0:
            return

        pos.mae_r = min(pos.mae_r, ((low if pos.is_long else h) - pos.entry_price)
                        * pos.direction / r)
        pos.mfe_r = max(pos.mfe_r, ((h if pos.is_long else low) - pos.entry_price)
                        * pos.direction / r)

        hit_stop = low <= pos.stop if pos.is_long else h >= pos.stop
        hit_tp1 = (h >= pos.tp1 if pos.is_long else low <= pos.tp1) and not pos.tp1_filled
        hit_tp2 = h >= pos.tp2 if pos.is_long else low <= pos.tp2

        if hit_stop and (self.cfg.backtest.pessimistic_fills or not (hit_tp1 or hit_tp2)):
            self._close(st, pos, pos.stop, ts, i, pos.stop_reason)
            return

        if hit_tp1:
            chunk = self._round_volume(st.spec, pos.volume * cfg.tp1_fraction)
            if 0 < chunk < pos.remaining:
                self._book(st, pos, pos.tp1, chunk)
                pos.tp1_filled = True
                if cfg.breakeven_after_tp1:
                    pos.stop = pos.entry_price + pos.direction * cfg.breakeven_offset_r * r
                    pos.stop_reason = "breakeven"
            elif chunk >= pos.remaining:
                pos.tp1_filled = True

        if hit_tp2 and pos.remaining > 0:
            self._close(st, pos, pos.tp2, ts, i, "target")
            return

        if pos.tp1_filled and cfg.trail_after_tp1 and not opened_this_bar:
            zi = int(st.e2z[i])
            if zi >= 0:
                self._trail(st, pos, zi)

    def _trail(self, st: SymbolState, pos: Position, zi: int) -> None:
        """Ratchet the stop behind the most recent opposite swing."""
        buffer = st.cfg.signal.stop_buffer_atr * float(st.entry_atr[pos.entry_index])
        if pos.is_long:
            swing = st.zone_struct.recent_swing_low_price(zi)
            if swing is not None and swing - buffer > pos.stop:
                pos.stop = swing - buffer
                pos.stop_reason = "trail"
        else:
            swing = st.zone_struct.recent_swing_high_price(zi)
            if swing is not None and swing + buffer < pos.stop:
                pos.stop = swing + buffer
                pos.stop_reason = "trail"

    def _round_volume(self, spec: SymbolSpec, volume: float) -> float:
        step = spec.volume_step if spec.volume_step > 0 else 0.01
        return round(int(volume / step) * step, 8)

    def _book(
        self, st: SymbolState, pos: Position, price: float, volume: float
    ) -> None:
        """Realize ``volume`` lots at ``price`` without closing the position."""
        spec = st.spec
        diff = (price - pos.entry_price) * pos.direction
        gross = diff / spec.tick_size * spec.tick_value * volume
        cost = self._chunk_cost(spec, volume, float(st.entry.spread[pos.entry_index]))
        net = gross - cost

        pos.realized += net
        pos.exit_notional += price * volume
        pos.closed_volume += volume
        pos.remaining = round(pos.remaining - volume, 8)
        self.balance += net

    def _close(
        self,
        st: SymbolState,
        pos: Position,
        price: float,
        ts: int,
        i: int,
        reason: str,
    ) -> None:
        if pos.remaining > 0:
            self._book(st, pos, price, pos.remaining)

        trade = ClosedTrade(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=ts,
            entry_price=pos.entry_price,
            exit_price=pos.average_exit,
            volume=pos.volume,
            profit=pos.realized,
            r_multiple=pos.realized / pos.risk_amount if pos.risk_amount else 0.0,
            reason=reason,
            score=pos.plan.score,
            zone_note=pos.plan.zone_note,
            mae_r=pos.mae_r,
            mfe_r=pos.mfe_r,
            bars_held=i - pos.entry_index,
            risk_distance=pos.risk_distance,
        )
        self.closed.append(trade)
        self.risk.on_close(trade)
        if pos in self.positions:
            self.positions.remove(pos)
        book = self._by_symbol.get(pos.symbol)
        if book and pos in book:
            book.remove(pos)

    def _flatten(self, st: SymbolState, i: int, ts: int, reason: str) -> None:
        if i < 0:
            return
        for pos in list(self._by_symbol.get(st.symbol, ())):
            self._close(st, pos, float(st.entry.close[i]), ts, i, reason)


def load_series(cfg: Config, symbol: str, cache_dir: str = "data", refresh: bool = False):
    """Fetch the three timeframes this strategy needs for one symbol."""
    from .data import load

    bt = cfg.backtest
    start = datetime.fromisoformat(bt.start).replace(tzinfo=timezone.utc)
    end = (
        datetime.fromisoformat(bt.end).replace(tzinfo=timezone.utc)
        if bt.end
        else datetime.now(timezone.utc)
    )
    ex = cfg.execution
    out = {}
    for tf in {ex.entry_timeframe, ex.zone_timeframe, ex.bias_timeframe}:
        out[tf] = load(symbol, tf, start, end, cache_dir=cache_dir, refresh=refresh)
    return out
