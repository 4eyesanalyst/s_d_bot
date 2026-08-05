"""Bulk-download the backtest basket from Dukascopy into data/.

    python download.py                     # default basket, default range
    python download.py --symbols EURUSD,GBPUSD --start 2020-01-01
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from sd_bot import sources


def one(symbol: str, tf: str, start: datetime, end: datetime, refresh: bool):
    t0 = time.time()
    bars = sources.load(symbol, tf, start, end, refresh=refresh)
    return symbol, tf, len(bars), time.time() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=",".join(sources.DEFAULT_BASKET))
    p.add_argument("--timeframes", default="M15,H4,D1")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tfs = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = (
        datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        if args.end else datetime.now(timezone.utc)
    )

    jobs = [(s, tf) for s in symbols for tf in tfs]
    print(f"{len(jobs)} downloads: {len(symbols)} symbols x {len(tfs)} timeframes")
    print(f"range {start:%Y-%m-%d} -> {end:%Y-%m-%d}, {args.workers} workers\n")

    done = failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(one, s, tf, start, end, args.refresh): (s, tf)
            for s, tf in jobs
        }
        for fut in as_completed(futures):
            s, tf = futures[fut]
            try:
                symbol, timeframe, n, secs = fut.result()
                done += 1
                print(f"[{done + failed:>3}/{len(jobs)}] {symbol:<8} {timeframe:<4} "
                      f"{n:>7} bars  {secs:>5.1f}s", flush=True)
            except Exception as exc:
                failed += 1
                print(f"[{done + failed:>3}/{len(jobs)}] {s:<8} {tf:<4} FAILED: {exc}",
                      flush=True)

    print(f"\n{done} ok, {failed} failed in {time.time() - t0:.0f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
