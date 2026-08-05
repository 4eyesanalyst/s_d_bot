"""Strategy, risk and execution configuration.

Defaults encode the rule set the bot trades. They are deliberately conservative:
a supply/demand system makes its money by *declining* most zones, so nearly every
threshold here is a filter that removes trades rather than one that adds them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ZoneConfig:
    """How a supply/demand zone is found and drawn."""

    # --- Base (the consolidation where orders were left behind) ---------------
    min_base_candles: int = 1
    max_base_candles: int = 6
    # A candle is "base" when its body is small relative to current volatility.
    base_body_atr: float = 0.50
    # ...and when the body does not dominate the candle's range.
    base_body_ratio: float = 0.60

    # --- Legs (the impulsive departure that proves institutional intent) ------
    # A candle is "impulse" when its body is large relative to volatility...
    leg_body_atr: float = 1.00
    # ...and mostly body rather than wick (conviction, not rejection).
    leg_body_ratio: float = 0.50
    # How far forward we measure the departure move.
    leg_lookahead: int = 12
    # Candles looked back over to label the approach into the base (DBR vs RBR).
    leg_in_lookback: int = 3
    # The departure must also be this many ATRs in absolute terms, so a
    # razor-thin base cannot clear the ratio test on a trivial move.
    min_leg_out_atr: float = 1.0
    # Demand a single explosive candle either side of the base. Off by default:
    # real departures often build over two or three candles, and requiring one
    # heroic bar discards roughly 90% of genuine order blocks.
    require_impulse_leg_in: bool = False
    require_impulse_leg_out: bool = False

    # --- Hard rejection filters ----------------------------------------------
    # Departure move must be this many multiples of the zone's own height.
    # This is the single best proxy for "price left in a hurry".
    min_departure_ratio: float = 3.0
    # Reject zones wider than this (bad risk:reward, ambiguous boundaries).
    max_zone_atr: float = 2.0
    # Reject zones thinner than this (noise, stop sits inside the spread).
    min_zone_atr: float = 0.08
    # A zone is dead after this many touches. Fresh zones carry the edge.
    max_tests: int = 1
    # Drop zones older than this many bars of their origin timeframe.
    max_zone_age_bars: int = 500

    # --- Scoring -------------------------------------------------------------
    # Weights sum to 100. See scoring.py for how each is earned.
    w_freshness: float = 22.0
    w_departure: float = 18.0
    w_base_tightness: float = 8.0
    w_bos: float = 14.0
    w_htf_trend: float = 14.0
    w_curve: float = 10.0
    w_htf_confluence: float = 9.0
    w_imbalance: float = 5.0
    # Minimum score a zone needs before it may be traded.
    min_score: float = 65.0


@dataclass
class StructureConfig:
    """Swing detection and market-structure reading."""

    # Fractal half-width: a swing high needs N lower highs on each side.
    swing_lookback: int = 2
    # Structure break needs a *close* beyond the level, not just a wick.
    require_close_break: bool = True
    # Zone must sit on the correct side of the range midpoint by this margin
    # (fraction of range height) to earn full curve credit.
    equilibrium_tolerance: float = 0.10


@dataclass
class SignalConfig:
    """Entry, stop and target construction."""

    # "limit"        -> resting order at the proximal line (best R, set & forget)
    # "confirmation" -> wait for a lower-timeframe shift inside the zone
    entry_mode: str = "limit"
    # Enter at this fraction into the zone. 0.0 = proximal edge, 0.5 = mid.
    entry_depth: float = 0.0
    # Stop goes beyond the distal line by max(this * zone height, buffer_atr * ATR).
    stop_buffer_zone: float = 0.15
    stop_buffer_atr: float = 0.20
    # Skip any setup that cannot pay this multiple of risk to its first target.
    min_risk_reward: float = 3.0
    # Stop-distance band, in pips. Below the floor, spread and noise dominate the
    # trade and costs eat the edge; above the ceiling the zone is too vague to be
    # an order block and the position size shrinks to nothing. This band is also
    # what sets how far the average trade travels: target move ~= stop x rr.
    min_stop_pips: float = 0.0      # 0 disables
    max_stop_pips: float = 0.0      # 0 disables
    # Partial exit plan.
    tp1_r: float = 2.0
    tp1_fraction: float = 0.5
    # Second target: "opposing_zone" walks to the next opposing zone's proximal,
    # "fixed_r" simply uses tp2_r.
    tp2_mode: str = "opposing_zone"
    tp2_r: float = 5.0
    # Move stop to entry once TP1 is banked.
    breakeven_after_tp1: bool = True
    breakeven_offset_r: float = 0.1
    # After TP1, trail behind the most recent opposite swing point.
    trail_after_tp1: bool = True
    trail_swing_lookback: int = 2
    # Abandon an untouched pending setup after this many entry-timeframe bars.
    pending_expiry_bars: int = 96


@dataclass
class RiskConfig:
    """Capital preservation. The part that actually decides survival."""

    risk_per_trade_pct: float = 0.5
    max_open_trades: int = 3
    max_trades_per_symbol: int = 1
    # Total risk of all open positions combined.
    max_total_risk_pct: float = 2.0
    # Halt for the rest of the day / week once these are hit.
    daily_loss_limit_pct: float = 3.0
    weekly_loss_limit_pct: float = 6.0
    # Stop adding risk once the account is this far below its high-water mark.
    max_drawdown_pct: float = 15.0
    # After N losses in a row, cut size by the given factor until a win.
    losing_streak_trigger: int = 3
    losing_streak_risk_factor: float = 0.5
    # Currency exposure cap: long EURUSD + long GBPUSD + long AUDUSD is one
    # oversized short-USD bet, not three independent trades.
    max_currency_exposure: float = 2.0
    enforce_currency_exposure: bool = True


@dataclass
class ExecutionConfig:
    """Broker-side and session mechanics."""

    symbols: list[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "XAUUSD"])
    # The three-screen workflow: bias / zones / entry.
    bias_timeframe: str = "D1"
    zone_timeframe: str = "H4"
    entry_timeframe: str = "M15"
    # Also harvest zones from the entry timeframe (nested, lower-R setups).
    use_entry_tf_zones: bool = False

    # Skip when the spread blows out (news, rollover, thin liquidity).
    max_spread_points: int = 30
    # Only trade these UTC hours. Default covers London + New York.
    session_hours_utc: list[int] = field(
        default_factory=lambda: list(range(7, 21))
    )
    trade_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    # Flatten before the weekend gap.
    friday_close_hour_utc: int = 20

    # Live order plumbing.
    magic_number: int = 771001
    deviation_points: int = 20
    comment: str = "sd_bot"
    # Poll interval for the live loop, in seconds.
    poll_seconds: int = 20


@dataclass
class AlertConfig:
    """Real-time signal delivery."""

    # console and file are always on; add "telegram", "discord", "desktop".
    channels: list[str] = field(default_factory=lambda: ["telegram"])
    directory: str = "signals"
    # Prefer environment variables over writing secrets into config.yaml.
    telegram_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook: str = ""

    # How often to look for new setups, in seconds.
    poll_seconds: int = 60
    # Re-alert the same zone at most this often (minutes). Stops a price that
    # hovers on a proximal line from firing the same signal repeatedly.
    cooldown_minutes: int = 240
    # Warn when a setup is this close to its entry, before it triggers.
    approach_atr: float = 0.5
    alert_on_approach: bool = True
    # Send a short summary of what is being watched, every N hours (0 = never).
    heartbeat_hours: int = 8


@dataclass
class BacktestConfig:
    """Simulation assumptions."""

    initial_balance: float = 10_000.0
    # Extra cost per side, in points, on top of the recorded spread.
    slippage_points: float = 2.0
    # Round-turn commission per lot, in account currency.
    commission_per_lot: float = 7.0
    # When a bar contains both the stop and the target, assume the stop.
    pessimistic_fills: bool = True
    start: str = "2022-01-01"
    end: str = ""  # empty -> now


@dataclass
class Config:
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    # Per-symbol overrides, e.g. {"EURUSD": {"signal": {"min_stop_pips": 12}}}.
    # A 40 pip stop floor is right for gold and would reject almost every FX
    # setup, so anything volatility-dependent belongs here.
    symbols: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        """Build a config from YAML, falling back to defaults for absent keys."""
        cfg = cls()
        if path is None:
            return cfg
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        _apply(cfg, raw, path=p.name)
        cfg.validate()
        return cfg

    def for_symbol(self, symbol: str) -> "Config":
        """This config with any per-symbol overrides applied."""
        overrides = self.symbols.get(symbol) or self.symbols.get(symbol.upper())
        if not overrides:
            return self
        import copy

        cfg = copy.deepcopy(self)
        cfg.symbols = {}
        _apply(cfg, overrides, path=f"symbols.{symbol}")
        cfg.symbols = self.symbols
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Fail loudly on settings that would silently misbehave."""
        from . import timeframes as tfmod

        ex = self.execution
        for label, tf in (
            ("bias_timeframe", ex.bias_timeframe),
            ("zone_timeframe", ex.zone_timeframe),
            ("entry_timeframe", ex.entry_timeframe),
        ):
            tfmod.minutes(tf)  # raises on unknown labels

        if not tfmod.is_higher(ex.bias_timeframe, ex.zone_timeframe):
            raise ValueError(
                f"bias_timeframe ({ex.bias_timeframe}) must be higher than "
                f"zone_timeframe ({ex.zone_timeframe})"
            )
        if not tfmod.is_higher(ex.zone_timeframe, ex.entry_timeframe):
            raise ValueError(
                f"zone_timeframe ({ex.zone_timeframe}) must be higher than "
                f"entry_timeframe ({ex.entry_timeframe})"
            )
        if self.zone.min_base_candles > self.zone.max_base_candles:
            raise ValueError("zone.min_base_candles exceeds zone.max_base_candles")
        if not 0.0 <= self.signal.entry_depth < 1.0:
            raise ValueError("signal.entry_depth must be in [0, 1)")
        if not 0.0 < self.signal.tp1_fraction <= 1.0:
            raise ValueError("signal.tp1_fraction must be in (0, 1]")
        if self.signal.entry_mode not in ("limit", "confirmation"):
            raise ValueError("signal.entry_mode must be 'limit' or 'confirmation'")
        if self.signal.tp2_mode not in ("opposing_zone", "fixed_r"):
            raise ValueError("signal.tp2_mode must be 'opposing_zone' or 'fixed_r'")
        if self.risk.risk_per_trade_pct <= 0:
            raise ValueError("risk.risk_per_trade_pct must be positive")
        if self.risk.risk_per_trade_pct > 2.0:
            raise ValueError(
                "risk.risk_per_trade_pct above 2% is not survivable over a normal "
                "losing streak; lower it or edit this check deliberately"
            )
        if not self.execution.symbols:
            raise ValueError("execution.symbols is empty")


def _apply(obj: Any, raw: dict, path: str, prefix: str = "") -> None:
    """Recursively overlay ``raw`` onto dataclass ``obj``, rejecting stray keys."""
    known = {f.name: f for f in fields(obj)}
    for key, value in raw.items():
        if key not in known:
            raise ValueError(
                f"{path}: unknown setting {prefix + key!r}. "
                f"Valid keys here: {sorted(known)}"
            )
        current = getattr(obj, key)
        if key == "symbols" and isinstance(current, dict):
            setattr(obj, key, value)   # per-symbol override blocks, free-form
            continue
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value, path, prefix=f"{prefix}{key}.")
        else:
            setattr(obj, key, value)
