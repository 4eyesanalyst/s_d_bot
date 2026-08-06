"""Read-only: how far does each trade run in our favour before it turns?

Answers "did it move 50-70 pips my way first?" by measuring maximum favourable
excursion (MFE) in pips for every backtested trade. Changes nothing -- it runs
the same backtest and only reports on the trades it produced.

    python analyse_mfe.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from sd_bot import sources
from sd_bot.backtest import Backtester
from sd_bot.config import Config
from sd_bot.trades import pip_size

TIMEFRAMES = ("M15", "H1", "H4")
BANDS = [(0, 10), (10, 25), (25, 50), (50, 70), (70, 100),
         (100, 200), (200, 500), (500, 10 ** 9)]


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
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cfg = Config.load("config.yaml")
    cfg.execution.symbols = symbols
    cfg.validate()

    print("loading...", flush=True)
    data = {s: load(s, args.start, args.end) for s in symbols}
    result = Backtester(cfg, {s: sources.spec_for(s) for s in data}).run(data)
    trades = result.trades

    # MFE is stored in R; convert to pips using each trade's own stop distance.
    rows = []
    for t in trades:
        pip = pip_size(t.symbol)
        mfe_pips = t.mfe_r * t.risk_distance / pip if pip > 0 else 0.0
        mae_pips = t.mae_r * t.risk_distance / pip if pip > 0 else 0.0
        rows.append((t, mfe_pips, mae_pips))

    print("\n" + "=" * 76)
    print("  HOW FAR DOES A TRADE RUN IN OUR FAVOUR BEFORE TURNING?")
    print("  (MFE = best unrealised move, in pips, at any point in the trade)")
    print("=" * 76)

    for sym in symbols + ["ALL"]:
        sel = [r for r in rows if sym == "ALL" or r[0].symbol == sym]
        if not sel:
            continue
        n = len(sel)
        mfes = sorted(r[1] for r in sel)
        pip = pip_size(sel[0][0].symbol) if sym != "ALL" else None

        print(f"\n  {sym}   {n} trades", end="")
        if pip:
            print(f"   (1 pip = {pip};  50-70 pips = "
                  f"{50 * pip:g}-{70 * pip:g} of price)")
        else:
            print()
        print("  " + "-" * 72)
        print(f"  {'MFE reached':<18}{'trades':>8}{'share':>9}   {'cumulative >=':>14}")
        running = n
        for lo, hi in BANDS:
            k = sum(1 for m in mfes if lo <= m < hi)
            label = f"{lo}-{hi} pips" if hi < 10 ** 9 else f"{lo}+ pips"
            print(f"  {label:<18}{k:>8}{k / n:>8.1%}   {running:>8} ({running / n:>5.1%})")
            running -= k

        reach50 = sum(1 for m in mfes if m >= 50)
        band = sum(1 for m in mfes if 50 <= m < 70)
        reach70 = sum(1 for m in mfes if m >= 70)
        med = mfes[n // 2]
        print(f"\n    median MFE          {med:>8.1f} pips")
        print(f"    reached >= 50 pips  {reach50:>8}  ({reach50 / n:.1%})")
        print(f"    landed in 50-70     {band:>8}  ({band / n:.1%})")
        print(f"    reached >= 70 pips  {reach70:>8}  ({reach70 / n:.1%})")

    # The question behind the question: of trades that eventually lost, how many
    # showed a decent profit first?
    print("\n" + "=" * 76)
    print("  OF THE TRADES THAT LOST, HOW MANY WENT OUR WAY FIRST?")
    print("=" * 76)
    print(f"  {'symbol':<9}{'losers':>8}{'>=25p first':>13}{'>=50p first':>13}{'>=70p first':>13}")
    print("  " + "-" * 72)
    for sym in symbols + ["ALL"]:
        sel = [r for r in rows
               if (sym == "ALL" or r[0].symbol == sym) and r[0].profit < 0]
        if not sel:
            continue
        n = len(sel)
        a = sum(1 for r in sel if r[1] >= 25)
        b = sum(1 for r in sel if r[1] >= 50)
        c = sum(1 for r in sel if r[1] >= 70)
        print(f"  {sym:<9}{n:>8}{a:>8} {a/n:>4.0%}{b:>8} {b/n:>4.0%}{c:>8} {c/n:>4.0%}")

    print("\n  (A high number here means losers gave back open profit -- which is")
    print("   what a wider stop or an earlier partial would have captured.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
