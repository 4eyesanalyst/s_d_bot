"""Fetch one symbol's M15 history in yearly chunks.

A single multi-year M15 request can stall for a long time with no progress. Year
chunks are quick, restartable, and show where they got to.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

import numpy as np

from sd_bot import sources
from sd_bot.data import Bars, cache_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--timeframe", default="M15")
    p.add_argument("--start", type=int, default=2022)
    p.add_argument("--end", type=int, default=2026)
    args = p.parse_args()

    pieces: list[Bars] = []
    for year in range(args.start, args.end + 1):
        lo = datetime(year, 1, 1, tzinfo=timezone.utc)
        hi = min(
            datetime(year + 1, 1, 1, tzinfo=timezone.utc),
            datetime.now(timezone.utc),
        )
        if lo >= hi:
            break
        t0 = time.time()
        try:
            bars = sources.fetch(args.symbol, args.timeframe, lo, hi)
        except Exception as exc:
            print(f"  {year}: FAILED {exc}", flush=True)
            continue
        pieces.append(bars)
        print(f"  {year}: {len(bars):>7} bars  {time.time() - t0:5.1f}s  "
              f"{bars.datetimes()[0]:%Y-%m-%d} -> {bars.datetimes()[-1]:%Y-%m-%d}",
              flush=True)

    if not pieces:
        print("nothing fetched")
        return 1

    merged = Bars(
        symbol=args.symbol,
        timeframe=args.timeframe,
        time=np.concatenate([b.time for b in pieces]),
        open=np.concatenate([b.open for b in pieces]),
        high=np.concatenate([b.high for b in pieces]),
        low=np.concatenate([b.low for b in pieces]),
        close=np.concatenate([b.close for b in pieces]),
        volume=np.concatenate([b.volume for b in pieces]),
        spread=np.concatenate([b.spread for b in pieces]),
    )
    path = cache_path("data", args.symbol, args.timeframe)
    merged.to_csv(path)
    print(f"\n{len(merged)} bars -> {path}")
    print(f"{merged.datetimes()[0]} .. {merged.datetimes()[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
