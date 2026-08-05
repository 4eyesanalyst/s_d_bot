"""Risk management.

Entries decide how often you are right. This module decides whether you are
still trading when it matters. Every check here can veto a signal, and a vetoed
signal is logged with its reason so the journal explains the quiet days.
"""

from __future__ import annotations

import math
from .config import RiskConfig
from .sessions import day_index, week_index
from .trades import LONG, ClosedTrade, Position, SymbolSpec

# Instruments that are not six-letter FX pairs still need an exposure bucket.
_METALS = {"XAU", "XAG", "XPT", "XPD"}


def split_currencies(symbol: str) -> tuple[str, str] | None:
    """``EURUSD`` -> ``("EUR", "USD")``. Returns None for indices, CFDs, etc."""
    core = "".join(ch for ch in symbol.upper() if ch.isalpha())
    if len(core) != 6:
        return None
    base, quote = core[:3], core[3:]
    if not (base.isalpha() and quote.isalpha()):
        return None
    return base, quote


def currency_exposure(positions: list[Position]) -> dict[str, float]:
    """Net directional exposure per currency across all open positions.

    Long EURUSD is long EUR and short USD. Three long-EUR trades against three
    different quotes are one trade with three times the size, and this is what
    catches that.
    """
    exposure: dict[str, float] = {}
    for p in positions:
        pair = split_currencies(p.symbol)
        sign = 1.0 if p.direction == LONG else -1.0
        if pair is None:
            exposure[p.symbol.upper()] = exposure.get(p.symbol.upper(), 0.0) + sign
            continue
        base, quote = pair
        exposure[base] = exposure.get(base, 0.0) + sign
        exposure[quote] = exposure.get(quote, 0.0) - sign
    return exposure


class RiskManager:
    """Tracks account state and vetoes trades that breach the risk rules."""

    def __init__(self, cfg: RiskConfig, starting_equity: float):
        self.cfg = cfg
        self.equity = starting_equity
        self.peak_equity = starting_equity
        self.day_start_equity = starting_equity
        self.week_start_equity = starting_equity
        self.consecutive_losses = 0
        self._day: int | None = None
        self._week: int | None = None
        self.veto_counts: dict[str, int] = {}

    # -- clock ---------------------------------------------------------------

    def on_time(self, ts: int, equity: float) -> None:
        """Advance the clock and roll the daily/weekly loss budgets."""
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        # Integer day/week identity: this runs once per bar per symbol.
        day = day_index(ts)
        week = week_index(ts)

        if self._day != day:
            self._day = day
            self.day_start_equity = equity
        if self._week != week:
            self._week = week
            self.week_start_equity = equity

    # -- budgets -------------------------------------------------------------

    def day_loss_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (self.day_start_equity - self.equity) / self.day_start_equity * 100.0

    def week_loss_pct(self) -> float:
        if self.week_start_equity <= 0:
            return 0.0
        return (self.week_start_equity - self.equity) / self.week_start_equity * 100.0

    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100.0

    def risk_pct(self) -> float:
        """Risk for the next trade, reduced while a losing streak is running."""
        pct = self.cfg.risk_per_trade_pct
        if self.consecutive_losses >= self.cfg.losing_streak_trigger:
            pct *= self.cfg.losing_streak_risk_factor
        return pct

    def open_risk_pct(self, positions: list[Position]) -> float:
        if self.equity <= 0:
            return 0.0
        at_risk = 0.0
        for p in positions:
            # Once the stop is at or beyond entry the trade risks nothing more.
            exposed = (p.entry_price - p.stop) * p.direction
            if exposed <= 0:
                continue
            at_risk += p.risk_amount * (p.remaining / p.volume if p.volume else 0.0)
        return at_risk / self.equity * 100.0

    # -- the gate ------------------------------------------------------------

    def veto(
        self,
        symbol: str,
        direction: int,
        positions: list[Position],
    ) -> str | None:
        """Return a reason string if this trade must not be taken, else None."""
        c = self.cfg

        if self.day_loss_pct() >= c.daily_loss_limit_pct:
            return self._count(f"daily loss limit ({c.daily_loss_limit_pct}%) hit")
        if self.week_loss_pct() >= c.weekly_loss_limit_pct:
            return self._count(f"weekly loss limit ({c.weekly_loss_limit_pct}%) hit")
        if self.drawdown_pct() >= c.max_drawdown_pct:
            return self._count(f"max drawdown ({c.max_drawdown_pct}%) breached")
        if len(positions) >= c.max_open_trades:
            return self._count("max open trades")
        if sum(1 for p in positions if p.symbol == symbol) >= c.max_trades_per_symbol:
            return self._count("already in this symbol")

        projected = self.open_risk_pct(positions) + self.risk_pct()
        if projected > c.max_total_risk_pct + 1e-9:
            return self._count("total open risk cap")

        if c.enforce_currency_exposure:
            reason = self._currency_veto(symbol, direction, positions)
            if reason:
                return self._count(reason)

        return None

    def _currency_veto(
        self, symbol: str, direction: int, positions: list[Position]
    ) -> str | None:
        exposure = currency_exposure(positions)
        pair = split_currencies(symbol)
        sign = 1.0 if direction == LONG else -1.0
        if pair is None:
            deltas = {symbol.upper(): sign}
        else:
            base, quote = pair
            deltas = {base: sign, quote: -sign}

        for ccy, delta in deltas.items():
            net = exposure.get(ccy, 0.0) + delta
            if abs(net) > self.cfg.max_currency_exposure + 1e-9:
                return f"{ccy} exposure would reach {net:+.0f}"
        return None

    def _count(self, reason: str) -> str:
        key = reason.split(" (")[0]
        self.veto_counts[key] = self.veto_counts.get(key, 0) + 1
        return reason

    # -- sizing --------------------------------------------------------------

    def volume_for(
        self, stop_distance: float, spec: SymbolSpec, risk_pct: float | None = None
    ) -> tuple[float, float]:
        """Lot size for the configured risk. Returns ``(volume, risk_amount)``.

        Returns ``(0.0, 0.0)`` when the correct size is below the broker's
        minimum lot -- taking the minimum anyway would silently over-risk the
        account, which is exactly the failure this system exists to avoid.
        """
        pct = self.risk_pct() if risk_pct is None else risk_pct
        risk_amount = self.equity * pct / 100.0
        if stop_distance <= 0 or risk_amount <= 0:
            return 0.0, 0.0

        loss_per_lot = spec.money_per_lot(stop_distance)
        if loss_per_lot <= 0:
            return 0.0, 0.0

        raw = risk_amount / loss_per_lot
        step = spec.volume_step if spec.volume_step > 0 else 0.01
        volume = math.floor(raw / step) * step
        volume = round(volume, 8)

        if volume < spec.volume_min:
            return 0.0, 0.0
        volume = min(volume, spec.volume_max)
        return volume, volume * loss_per_lot

    # -- feedback ------------------------------------------------------------

    def on_close(self, trade: ClosedTrade) -> None:
        if trade.profit > 0:
            self.consecutive_losses = 0
        elif trade.profit < 0:
            self.consecutive_losses += 1

    def status(self) -> str:
        return (
            f"equity={self.equity:,.2f} dd={self.drawdown_pct():.2f}% "
            f"day={-self.day_loss_pct():+.2f}% week={-self.week_loss_pct():+.2f}% "
            f"streak={self.consecutive_losses} risk={self.risk_pct():.2f}%"
        )
