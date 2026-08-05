"""Performance measurement.

Deliberately R-centric. Currency P/L flatters a system that happened to size up
into its winners; expectancy per unit of risk is the number that tells you
whether the rules have an edge.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .backtest import Result
from .scoring import grade
from .sources import pip_size
from .trades import ClosedTrade


@dataclass
class Stats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    win_rate: float = 0.0
    net_profit: float = 0.0
    return_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    total_r: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    best_r: float = 0.0
    worst_r: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_money: float = 0.0
    longest_loss_streak: int = 0
    avg_bars_held: float = 0.0
    sharpe: float = 0.0
    # Frequency and size of the average move, which is what a trader feels.
    trading_days: int = 0
    trades_per_day: float = 0.0
    calendar_weeks: float = 0.0
    trades_per_week: float = 0.0
    avg_move_pips: float = 0.0      # signed, across all trades
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    avg_risk_pips: float = 0.0
    median_win_pips: float = 0.0
    by_reason: dict[str, int] = field(default_factory=dict)
    by_symbol: dict[str, tuple[int, float]] = field(default_factory=dict)
    by_grade: dict[str, tuple[int, float]] = field(default_factory=dict)


def compute(result: Result) -> Stats:
    s = Stats()
    trades = result.trades
    s.trades = len(trades)
    s.net_profit = result.ending_balance - result.starting_balance
    s.return_pct = (
        s.net_profit / result.starting_balance * 100.0
        if result.starting_balance
        else 0.0
    )
    s.max_drawdown_pct, s.max_drawdown_money = _drawdown(result.equity_curve)

    if not trades:
        return s

    wins = [t for t in trades if t.profit > 0]
    losses = [t for t in trades if t.profit < 0]
    s.wins, s.losses = len(wins), len(losses)
    s.scratches = s.trades - s.wins - s.losses
    s.win_rate = s.wins / s.trades * 100.0

    s.gross_profit = sum(t.profit for t in wins)
    s.gross_loss = abs(sum(t.profit for t in losses))
    s.profit_factor = (
        s.gross_profit / s.gross_loss if s.gross_loss > 0 else math.inf
    )

    rs = [t.r_multiple for t in trades]
    s.total_r = sum(rs)
    s.expectancy_r = s.total_r / s.trades
    s.avg_win_r = sum(t.r_multiple for t in wins) / len(wins) if wins else 0.0
    s.avg_loss_r = sum(t.r_multiple for t in losses) / len(losses) if losses else 0.0
    s.best_r, s.worst_r = max(rs), min(rs)
    s.avg_bars_held = sum(t.bars_held for t in trades) / s.trades

    streak = best = 0
    for t in trades:
        streak = streak + 1 if t.profit < 0 else 0
        best = max(best, streak)
    s.longest_loss_streak = best

    by_reason: dict[str, int] = defaultdict(int)
    for t in trades:
        by_reason[t.reason] += 1
    s.by_reason = dict(sorted(by_reason.items(), key=lambda kv: -kv[1]))

    s.by_symbol = _bucket(trades, lambda t: t.symbol)
    s.by_grade = _bucket(trades, lambda t: grade(t.score))
    s.sharpe = _sharpe(result.equity_curve)

    # Frequency, measured against days the market was actually open.
    days = {
        datetime.fromtimestamp(ts, tz=timezone.utc).date()
        for ts, _ in result.equity_curve
    }
    s.trading_days = len(days)
    s.trades_per_day = s.trades / s.trading_days if s.trading_days else 0.0

    # Per-week is measured against elapsed calendar time, not bar days: it is
    # the number you actually plan around.
    span = result.equity_curve[-1][0] - result.equity_curve[0][0]
    s.calendar_weeks = span / (7 * 86_400)
    s.trades_per_week = s.trades / s.calendar_weeks if s.calendar_weeks else 0.0

    pips = [t.move_pips(pip_size(t.symbol)) for t in trades]
    s.avg_move_pips = sum(pips) / len(pips)
    win_pips = [t.move_pips(pip_size(t.symbol)) for t in wins]
    loss_pips = [t.move_pips(pip_size(t.symbol)) for t in losses]
    s.avg_win_pips = sum(win_pips) / len(win_pips) if win_pips else 0.0
    s.avg_loss_pips = sum(loss_pips) / len(loss_pips) if loss_pips else 0.0
    s.median_win_pips = _median(win_pips)
    risks = [t.risk_pips(pip_size(t.symbol)) for t in trades if t.risk_distance > 0]
    s.avg_risk_pips = sum(risks) / len(risks) if risks else 0.0
    return s


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _bucket(trades: list[ClosedTrade], key) -> dict[str, tuple[int, float]]:
    counts: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        counts[key(t)].append(t.r_multiple)
    return {
        k: (len(v), sum(v)) for k, v in sorted(counts.items(), key=lambda kv: -sum(kv[1]))
    }


def _drawdown(curve: list[tuple[int, float]]) -> tuple[float, float]:
    peak = -math.inf
    worst_pct = worst_money = 0.0
    for _, equity in curve:
        peak = max(peak, equity)
        if peak <= 0:
            continue
        drop = peak - equity
        worst_money = max(worst_money, drop)
        worst_pct = max(worst_pct, drop / peak * 100.0)
    return worst_pct, worst_money


def _sharpe(curve: list[tuple[int, float]], periods_per_year: int = 252) -> float:
    """Annualised Sharpe on daily closing equity, zero risk-free rate."""
    daily: dict[tuple[int, int, int], float] = {}
    for ts, equity in curve:
        d = datetime.fromtimestamp(ts, tz=timezone.utc)
        daily[(d.year, d.month, d.day)] = equity
    values = [v for _, v in sorted(daily.items())]
    if len(values) < 3:
        return 0.0

    returns = []
    for prev, cur in zip(values, values[1:]):
        if prev > 0:
            returns.append(cur / prev - 1.0)
    if len(returns) < 2:
        return 0.0

    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0
    return mean / sd * math.sqrt(periods_per_year)


def report(result: Result, s: Stats) -> str:
    """Human-readable summary, including *why trades were not taken*."""
    pf = "inf" if s.profit_factor == math.inf else f"{s.profit_factor:.2f}"
    lines = [
        "=" * 66,
        "  SUPPLY & DEMAND BACKTEST",
        "=" * 66,
        f"  Balance       {result.starting_balance:>12,.2f} -> {result.ending_balance:>12,.2f}",
        f"  Net profit    {s.net_profit:>12,.2f}   ({s.return_pct:+.2f}%)",
        f"  Max drawdown  {s.max_drawdown_money:>12,.2f}   ({s.max_drawdown_pct:.2f}%)",
        "",
        f"  Trades        {s.trades:>6}      Win rate     {s.win_rate:>6.1f}%",
        f"  Wins/Losses   {s.wins:>3}/{s.losses:<3}      Profit factor{pf:>7}",
        f"  Expectancy    {s.expectancy_r:>6.3f}R     Total        {s.total_r:>6.1f}R",
        f"  Avg win       {s.avg_win_r:>6.2f}R     Avg loss     {s.avg_loss_r:>6.2f}R",
        f"  Best          {s.best_r:>6.2f}R     Worst        {s.worst_r:>6.2f}R",
        f"  Sharpe        {s.sharpe:>6.2f}      Loss streak  {s.longest_loss_streak:>6}",
        f"  Avg hold      {s.avg_bars_held:>6.1f} bars",
        "",
        f"  Trades/week   {s.trades_per_week:>6.2f}      over {s.calendar_weeks:.0f} weeks",
        f"  Trades/day    {s.trades_per_day:>6.2f}      over {s.trading_days} session days",
        f"  Avg risk      {s.avg_risk_pips:>6.1f} pips  (entry to initial stop)",
        f"  Avg move      {s.avg_move_pips:>+6.1f} pips  (all trades, signed)",
        f"  Avg winner    {s.avg_win_pips:>+6.1f} pips  median {s.median_win_pips:>+6.1f}",
        f"  Avg loser     {s.avg_loss_pips:>+6.1f} pips",
    ]

    if s.by_symbol:
        lines += ["", "  By symbol            trades      total R"]
        for sym, (n, total) in s.by_symbol.items():
            lines.append(f"    {sym:<18} {n:>6}     {total:>8.1f}R")

    if s.by_grade:
        lines += ["", "  By zone grade        trades      total R"]
        for g, (n, total) in s.by_grade.items():
            lines.append(f"    {g:<18} {n:>6}     {total:>8.1f}R")

    if s.by_reason:
        lines += ["", "  Exit reason          count"]
        for reason, n in s.by_reason.items():
            lines.append(f"    {reason:<18} {n:>6}")

    zones = sum(result.zones_found.values())
    lines += [
        "",
        f"  Zones detected: {zones}  "
        + "  ".join(f"{k}={v}" for k, v in result.zones_found.items()),
    ]

    if result.rejected_plans:
        lines += ["", "  Setups declined (zone reached but plan refused):"]
        for reason, n in sorted(result.rejected_plans.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {reason:<44} {n:>6}")

    if result.veto_counts:
        lines += ["", "  Risk vetoes:"]
        for reason, n in sorted(result.veto_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {reason:<44} {n:>6}")

    lines.append("=" * 66)
    return "\n".join(lines)
