"""Trade objects shared by the backtester, the live loop and the journal."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

LONG = 1
SHORT = -1


def side_name(direction: int) -> str:
    return "BUY" if direction == LONG else "SELL"


def pip_size(symbol: str) -> float:
    """One pip, by market convention -- what traders mean when they say "pips".

    Deliberately convention-based rather than derived from digits: gold quotes to
    two decimals but a "pip" of gold is 0.10, not 0.01.
    """
    s = symbol.upper()
    if s.startswith("XAU"):
        return 0.1
    if s.startswith("XAG"):
        return 0.01
    if s.endswith("JPY"):
        return 0.01
    return 0.0001


@dataclass
class SymbolSpec:
    """Contract details needed to size a position correctly."""

    name: str
    digits: int = 5
    point: float = 0.00001
    tick_size: float = 0.00001
    tick_value: float = 1.0      # account-currency value of one tick per 1.0 lot
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    contract_size: float = 100_000.0

    @property
    def pip(self) -> float:
        """One pip. For 5/3-digit quotes that is ten points."""
        return self.point * 10 if self.digits in (3, 5) else self.point

    def money_per_lot(self, price_distance: float) -> float:
        """Account-currency P/L of a 1.0-lot move of ``price_distance``."""
        if self.tick_size <= 0:
            return 0.0
        return abs(price_distance) / self.tick_size * self.tick_value


@dataclass
class TradePlan:
    """A fully specified setup, before any capital is committed."""

    symbol: str
    direction: int
    entry: float
    stop: float
    tp1: float
    tp2: float
    risk_distance: float
    rr1: float
    rr2: float
    score: float
    zone_id: int
    zone_note: str
    created_index: int
    expires_index: int
    created_time: int = 0

    @property
    def is_long(self) -> bool:
        return self.direction == LONG


@dataclass
class Position:
    """An open trade, including its partial-exit state."""

    symbol: str
    direction: int
    entry_price: float
    entry_time: int
    entry_index: int
    volume: float
    stop: float
    tp1: float
    tp2: float
    initial_stop: float
    risk_amount: float
    plan: TradePlan
    remaining: float = 0.0
    tp1_filled: bool = False
    realized: float = 0.0
    ticket: int = 0

    # Accumulated exit state, so a position that scales out still reports as a
    # single trade with one blended result.
    closed_volume: float = 0.0
    exit_notional: float = 0.0
    mae_r: float = 0.0
    mfe_r: float = 0.0
    stop_reason: str = "stop"

    def __post_init__(self) -> None:
        if self.remaining == 0.0:
            self.remaining = self.volume

    @property
    def average_exit(self) -> float:
        if self.closed_volume <= 0:
            return self.entry_price
        return self.exit_notional / self.closed_volume

    @property
    def is_long(self) -> bool:
        return self.direction == LONG

    @property
    def risk_distance(self) -> float:
        return abs(self.entry_price - self.initial_stop)

    def r_multiple(self, exit_price: float) -> float:
        """Result of ``exit_price`` expressed in units of initial risk."""
        d = self.risk_distance
        if d <= 0:
            return 0.0
        return (exit_price - self.entry_price) * self.direction / d


@dataclass
class ClosedTrade:
    """One completed trade, written to the journal and fed to the stats."""

    symbol: str
    direction: int
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    volume: float
    profit: float
    r_multiple: float
    reason: str
    score: float
    zone_note: str
    mae_r: float = 0.0
    mfe_r: float = 0.0
    bars_held: int = 0
    risk_distance: float = 0.0   # price distance from entry to the initial stop

    @property
    def won(self) -> bool:
        return self.profit > 0

    @property
    def move_distance(self) -> float:
        """Price travelled from entry to the blended exit, unsigned."""
        return abs(self.exit_price - self.entry_price)

    def move_pips(self, pip: float) -> float:
        """Signed move in pips: positive when the trade went our way."""
        if pip <= 0:
            return 0.0
        return (self.exit_price - self.entry_price) * self.direction / pip

    def risk_pips(self, pip: float) -> float:
        return self.risk_distance / pip if pip > 0 else 0.0

    def as_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": side_name(self.direction),
            "entry_time": datetime.fromtimestamp(
                self.entry_time, tz=timezone.utc
            ).isoformat(),
            "exit_time": datetime.fromtimestamp(
                self.exit_time, tz=timezone.utc
            ).isoformat(),
            "entry": round(self.entry_price, 6),
            "exit": round(self.exit_price, 6),
            "volume": self.volume,
            "profit": round(self.profit, 2),
            "R": round(self.r_multiple, 3),
            "mae_R": round(self.mae_r, 3),
            "mfe_R": round(self.mfe_r, 3),
            "bars": self.bars_held,
            "reason": self.reason,
            "score": round(self.score, 1),
            "zone": self.zone_note,
        }
