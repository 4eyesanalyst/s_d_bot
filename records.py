"""Full performance record: monthly returns, drawdowns, and a trade ledger.

Yearly totals hide what a month actually feels like. This prints the record a
trader would want before committing capital -- every month, the worst run of
losses, how long drawdowns lasted, and whether recent performance still matches
the backtest.

    python records.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sd_bot import sources
from sd_bot.backtest import Backtester
from sd_bot.config import Config
from sd_bot.stats import compute
from sd_bot.trades import LONG, pip_size

TIMEFRAMES = ("M15", "H1", "H4")


def load(symbol: str, start: str, end: str) -> dict:
    lo = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    hi = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    out = {}
    for tf in TIMEFRAMES:
        b = sources.load(symbol, tf, lo, hi)
        a = int(np.searchsorted(b.time, lo.timestamp(), "left"))
        z = int(np.searchsorted(b.time, hi.timestamp(), "right"))
        out[tf] = b.slice(a, z)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="XAUUSD,EURUSD,USDJPY")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-01")
    p.add_argument("--balance", type=float, default=10_000)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cfg = Config.load("config.yaml")
    cfg.execution.symbols = symbols
    cfg.backtest.initial_balance = args.balance
    cfg.validate()

    print("loading...", flush=True)
    data = {s: load(s, args.start, args.end) for s in symbols}
    result = Backtester(cfg, {s: sources.spec_for(s) for s in data}).run(data)
    st = compute(result)
    trades = sorted(result.trades, key=lambda t: t.exit_time)

    print("\n" + "=" * 78)
    print("  PERFORMANCE RECORD")
    print("=" * 78)
    print(f"  instruments    {', '.join(symbols)}")
    print(f"  period         {args.start} -> {args.end}")
    print(f"  risk/trade     {cfg.risk.risk_per_trade_pct}%")
    print(f"  sessions       {min(cfg.execution.session_hours_utc):02d}:00-"
          f"{max(cfg.execution.session_hours_utc):02d}:59 UTC, Mon-Fri")
    print()
    print(f"  starting       {args.balance:>12,.2f}")
    print(f"  ending         {result.ending_balance:>12,.2f}")
    print(f"  net profit     {st.net_profit:>12,.2f}   ({st.return_pct:+.1f}%)")
    years = (datetime.fromisoformat(args.end)
             - datetime.fromisoformat(args.start)).days / 365.25
    cagr = ((result.ending_balance / args.balance) ** (1 / years) - 1) * 100
    print(f"  CAGR           {cagr:>12.1f}%   over {years:.1f} years")
    print(f"  max drawdown   {st.max_drawdown_money:>12,.2f}   ({st.max_drawdown_pct:.2f}%)")
    print(f"  return/DD      {st.return_pct / max(st.max_drawdown_pct, 1e-9):>12.2f}")
    print()
    print(f"  trades         {st.trades:>12}   ({st.trades_per_week:.2f}/week)")
    print(f"  win rate       {st.win_rate:>11.1f}%")
    print(f"  profit factor  {st.profit_factor:>12.2f}")
    print(f"  expectancy     {st.expectancy_r:>+12.3f}R")
    print(f"  avg win/loss   {st.avg_win_r:>+7.2f}R / {st.avg_loss_r:+.2f}R")
    print(f"  Sharpe         {st.sharpe:>12.2f}")
    print(f"  worst streak   {st.longest_loss_streak:>12} losses")

    # -- monthly ----------------------------------------------------------
    monthly: dict[str, float] = defaultdict(float)
    m_count: dict[str, int] = defaultdict(int)
    for t in trades:
        key = datetime.fromtimestamp(t.exit_time, tz=timezone.utc).strftime("%Y-%m")
        monthly[key] += t.profit
        m_count[key] += 1

    print("\n" + "-" * 78)
    print("  MONTHLY (currency, on a fixed 10k base -- not compounded)")
    print("-" * 78)
    years_seen = sorted({k[:4] for k in monthly})
    print(f"  {'year':<6}" + "".join(f"{m:>6}" for m in
          ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
          + f"{'total':>9}")
    for y in years_seen:
        row = f"  {y:<6}"
        total = 0.0
        for m in range(1, 13):
            v = monthly.get(f"{y}-{m:02d}")
            total += v or 0.0
            row += f"{v:>6.0f}" if v else f"{'.':>6}"
        row += f"{total:>9.0f}"
        print(row)

    values = [v for _, v in sorted(monthly.items())]
    wins = [v for v in values if v > 0]
    print(f"\n  months traded  {len(values)}")
    print(f"  profitable     {len(wins)} ({len(wins)/len(values):.0%})")
    print(f"  best month     {max(values):>+9,.0f}")
    print(f"  worst month    {min(values):>+9,.0f}")
    print(f"  average month  {sum(values)/len(values):>+9,.0f}")
    streak = worst = 0
    for v in values:
        streak = streak + 1 if v < 0 else 0
        worst = max(worst, streak)
    print(f"  worst run      {worst} consecutive losing months")

    # -- per symbol -------------------------------------------------------
    print("\n" + "-" * 78)
    print("  BY INSTRUMENT")
    print("-" * 78)
    print(f"  {'symbol':<10}{'trades':>8}{'win%':>8}{'pf':>8}{'expR':>9}{'profit':>12}")
    for sym in symbols:
        sel = [t for t in trades if t.symbol == sym]
        if not sel:
            continue
        w = [t for t in sel if t.profit > 0]
        gl = abs(sum(t.profit for t in sel if t.profit < 0)) or 1e-9
        pf = sum(t.profit for t in w) / gl
        print(f"  {sym:<10}{len(sel):>8}{len(w)/len(sel)*100:>8.1f}{pf:>8.2f}"
              f"{sum(t.r_multiple for t in sel)/len(sel):>+9.3f}"
              f"{sum(t.profit for t in sel):>12,.0f}")

    # -- direction --------------------------------------------------------
    print("\n" + "-" * 78)
    print("  BY DIRECTION")
    print("-" * 78)
    for name, sel in (("LONG", [t for t in trades if t.direction == LONG]),
                      ("SHORT", [t for t in trades if t.direction != LONG])):
        if not sel:
            continue
        w = [t for t in sel if t.profit > 0]
        print(f"  {name:<10}{len(sel):>8}{len(w)/len(sel)*100:>8.1f}%"
              f"{sum(t.r_multiple for t in sel)/len(sel):>+9.3f}R"
              f"{sum(t.profit for t in sel):>12,.0f}")

    # -- ledger -----------------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "ledger.csv"
    running = args.balance
    with ledger.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["#", "symbol", "side", "entry_time", "exit_time", "entry",
                    "exit", "pips", "R", "profit", "balance", "reason", "grade"])
        for n, t in enumerate(trades, 1):
            running += t.profit
            w.writerow([
                n, t.symbol, "BUY" if t.direction == LONG else "SELL",
                datetime.fromtimestamp(t.entry_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                datetime.fromtimestamp(t.exit_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                round(t.entry_price, 5), round(t.exit_price, 5),
                round(t.move_pips(pip_size(t.symbol)), 1), round(t.r_multiple, 3),
                round(t.profit, 2), round(running, 2), t.reason,
                t.zone_note.split()[1] if len(t.zone_note.split()) > 1 else "",
            ])
    print(f"\n  full ledger -> {ledger}  ({len(trades)} trades)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
