"""Year-by-year validation of a single configuration.

A strategy that makes all its money in one year is a bet on that year. This
prints each calendar year separately so a lucky regime cannot hide inside a
flattering total.

    python validate.py --symbol XAUUSD
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from sd_bot import sources
from sd_bot.backtest import Backtester
from sd_bot.config import Config
from sd_bot.stats import compute, report

TIMEFRAMES = ("M15", "H1", "H4", "D1")


def load(symbol: str, start: str, end: str):
    lo = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    hi = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    return {tf: sources.load(symbol, tf, lo, hi) for tf in TIMEFRAMES}


def window(series, start: str, end: str):
    lo = datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp()
    hi = datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp()
    cut = {}
    for tf, bars in series.items():
        a = int(np.searchsorted(bars.time, lo, "left"))
        b = int(np.searchsorted(bars.time, hi, "right"))
        cut[tf] = bars.slice(a, b)
    return cut


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-01")
    p.add_argument("--full", action="store_true", help="print the full report too")
    args = p.parse_args()

    cfg = Config.load(args.config)
    cfg.execution.symbols = [args.symbol]
    cfg.validate()
    specs = {args.symbol: sources.spec_for(args.symbol)}

    series = load(args.symbol, args.start, args.end)
    print(f"{args.symbol}  "
          + "  ".join(f"{tf}={len(b)}" for tf, b in series.items()))
    print(f"zones={cfg.execution.zone_timeframe} bias={cfg.execution.bias_timeframe} "
          f"entry={cfg.execution.entry_timeframe}  min_score={cfg.zone.min_score} "
          f"minRR={cfg.signal.min_risk_reward} "
          f"stop>={cfg.signal.min_stop_pips}p BE={cfg.signal.breakeven_after_tp1}\n")

    cache: dict = {}
    header = (f"{'period':<14}{'trds':>6}{'t/wk':>7}{'win%':>7}{'pf':>7}"
              f"{'expR':>8}{'ret%':>8}{'dd%':>7}{'wpips':>7}{'risk':>7}")
    print(header)
    print("-" * len(header))

    years = range(
        datetime.fromisoformat(args.start).year,
        datetime.fromisoformat(args.end).year + 1,
    )
    for year in years:
        lo = f"{year}-01-01"
        hi = min(f"{year + 1}-01-01", args.end)
        if lo >= hi:
            continue
        cut = window(series, lo, hi)
        if any(len(b) < 100 for b in cut.values()):
            continue
        res = Backtester(cfg, specs, analysis_cache=cache).run({args.symbol: cut})
        s = compute(res)
        pf = "  inf" if s.profit_factor == float("inf") else f"{s.profit_factor:6.2f}"
        print(f"{year:<14}{s.trades:>6}{s.trades_per_week:>7.2f}{s.win_rate:>7.1f}"
              f"{pf}{s.expectancy_r:>+8.2f}{s.return_pct:>+8.1f}"
              f"{s.max_drawdown_pct:>7.1f}{s.avg_win_pips:>7.0f}"
              f"{s.avg_risk_pips:>7.0f}")

    print("-" * len(header))
    res = Backtester(cfg, specs, analysis_cache=cache).run({args.symbol: series})
    s = compute(res)
    pf = "  inf" if s.profit_factor == float("inf") else f"{s.profit_factor:6.2f}"
    print(f"{'ALL':<14}{s.trades:>6}{s.trades_per_week:>7.2f}{s.win_rate:>7.1f}"
          f"{pf}{s.expectancy_r:>+8.2f}{s.return_pct:>+8.1f}"
          f"{s.max_drawdown_pct:>7.1f}{s.avg_win_pips:>7.0f}{s.avg_risk_pips:>7.0f}")

    if args.full:
        print()
        print(report(res, s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
