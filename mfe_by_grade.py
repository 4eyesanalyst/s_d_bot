"""How far does each grade actually run in our favour?

Cross-tabulates maximum favourable excursion against zone grade for every
backtested trade. Read-only: it re-runs the same backtest and reports on the
trades it produced, changing nothing.

    python mfe_by_grade.py
    python mfe_by_grade.py --threshold 70
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone

from sd_bot import sources
from sd_bot.backtest import Backtester
from sd_bot.config import Config
from sd_bot.scoring import grade
from sd_bot.trades import pip_size

SYMBOLS = ["XAUUSD", "EURUSD", "USDJPY"]
TIMEFRAMES = ("M15", "H1", "H4")
ORDER = ["A+", "A", "B", "C", "D"]


def load(symbol: str, start: str, end: str) -> dict:
    lo = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    hi = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    return {tf: sources.load(symbol, tf, lo, hi) for tf in TIMEFRAMES}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=50.0,
                   help="pips in our favour to count as 'moved'")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-01")
    args = p.parse_args()

    cfg = Config.load("config.yaml")
    cfg.execution.symbols = SYMBOLS
    cfg.backtest.start, cfg.backtest.end = args.start, args.end
    cfg.validate()

    print("loading...", flush=True)
    data = {s: load(s, args.start, args.end) for s in SYMBOLS}
    result = Backtester(
        cfg, {s: sources.spec_for(s) for s in SYMBOLS}
    ).run(data)

    t = args.threshold
    buckets: dict[str, list] = defaultdict(list)
    for trade in result.trades:
        pip = pip_size(trade.symbol)
        # mfe_r is in units of initial risk; convert back to price, then pips.
        mfe_pips = trade.mfe_r * trade.risk_distance / pip if pip else 0.0
        buckets[grade(trade.score)].append((mfe_pips, trade))

    print()
    print("=" * 78)
    print(f"  HOW MANY TRADES RAN AT LEAST {t:.0f} PIPS IN OUR FAVOUR, BY GRADE")
    print("=" * 78)
    print(f"  {len(result.trades)} trades, {args.start} to {args.end}")
    print()
    print(f"  {'grade':<8}{'trades':>8}{'reached ' + str(int(t)) + 'p':>14}"
          f"{'share':>9}{'median MFE':>13}{'won':>8}{'expR':>9}")
    print("  " + "-" * 74)

    for g in ORDER:
        rows = buckets.get(g)
        if not rows:
            continue
        n = len(rows)
        hit = sum(1 for m, _ in rows if m >= t)
        med = sorted(m for m, _ in rows)[n // 2]
        won = sum(1 for _, tr in rows if tr.profit > 0)
        exp = sum(tr.r_multiple for _, tr in rows) / n
        print(f"  {g:<8}{n:>8}{hit:>14}{hit / n:>8.0%}"
              f"{med:>12.0f}p{won / n:>7.0%}{exp:>+9.3f}")

    allrows = [r for rows in buckets.values() for r in rows]
    n = len(allrows)
    hit = sum(1 for m, _ in allrows if m >= t)
    med = sorted(m for m, _ in allrows)[n // 2]
    won = sum(1 for _, tr in allrows if tr.profit > 0)
    exp = sum(tr.r_multiple for _, tr in allrows) / n
    print("  " + "-" * 74)
    print(f"  {'ALL':<8}{n:>8}{hit:>14}{hit / n:>8.0%}"
          f"{med:>12.0f}p{won / n:>7.0%}{exp:>+9.3f}")

    # Per instrument, because a "pip" means very different things across them.
    print()
    print("  Same question, per instrument (a gold pip is 0.10, so 50 gold")
    print("  pips is only $5 -- the threshold is not comparable across rows):")
    print()
    print(f"  {'symbol':<10}{'trades':>8}{'reached ' + str(int(t)) + 'p':>14}"
          f"{'share':>9}{'median MFE':>13}")
    print("  " + "-" * 60)
    per: dict[str, list] = defaultdict(list)
    for m, tr in allrows:
        per[tr.symbol].append(m)
    for sym in SYMBOLS:
        vals = per.get(sym)
        if not vals:
            continue
        hit = sum(1 for m in vals if m >= t)
        med = sorted(vals)[len(vals) // 2]
        print(f"  {sym:<10}{len(vals):>8}{hit:>14}{hit / len(vals):>8.0%}"
              f"{med:>12.0f}p")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
