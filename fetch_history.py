"""Download history for a watchlist, in yearly chunks.

Dukascopy stalls badly on multi-year intraday requests. Year chunks are quick
and restartable, and each symbol/timeframe is written as soon as it completes so
an interrupted run keeps everything it already had.

    python fetch_history.py --symbols EURUSD,GBPUSD --timeframes M15,H1,H4,D1
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

from sd_bot import sources
from sd_bot.data import Bars, cache_path


def fetch_years(symbol: str, timeframe: str, start: int, end: int) -> Bars | None:
    pieces: list[Bars] = []
    for year in range(start, end + 1):
        lo = datetime(year, 1, 1, tzinfo=timezone.utc)
        hi = min(datetime(year + 1, 1, 1, tzinfo=timezone.utc),
                 datetime.now(timezone.utc))
        if lo >= hi:
            break
        for attempt in range(3):
            try:
                pieces.append(sources.fetch(symbol, timeframe, lo, hi))
                break
            except Exception:
                if attempt == 2:
                    return None
                time.sleep(2)
    if not pieces:
        return None

    merged = Bars(
        symbol=symbol,
        timeframe=timeframe,
        time=np.concatenate([b.time for b in pieces]),
        open=np.concatenate([b.open for b in pieces]),
        high=np.concatenate([b.high for b in pieces]),
        low=np.concatenate([b.low for b in pieces]),
        close=np.concatenate([b.close for b in pieces]),
        volume=np.concatenate([b.volume for b in pieces]),
        spread=np.concatenate([b.spread for b in pieces]),
    )
    merged.to_csv(cache_path("data", symbol, timeframe))
    return merged


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True)
    p.add_argument("--timeframes", default="M15,H1,H4,D1")
    p.add_argument("--start", type=int, default=2022)
    p.add_argument("--end", type=int, default=2026)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tfs = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    jobs = [(s, tf) for s in symbols for tf in tfs]
    print(f"{len(jobs)} downloads ({len(symbols)} symbols x {len(tfs)} timeframes), "
          f"{args.start}-{args.end}\n", flush=True)

    done = failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_years, s, tf, args.start, args.end): (s, tf)
            for s, tf in jobs
        }
        for fut in as_completed(futures):
            s, tf = futures[fut]
            try:
                bars = fut.result()
            except Exception as exc:
                bars = None
                print(f"  {s} {tf}: error {exc}", flush=True)
            if bars is None:
                failed += 1
                print(f"[{done + failed:>3}/{len(jobs)}] {s:<8} {tf:<4} FAILED",
                      flush=True)
            else:
                done += 1
                print(f"[{done + failed:>3}/{len(jobs)}] {s:<8} {tf:<4} "
                      f"{len(bars):>7} bars", flush=True)

    print(f"\n{done} ok, {failed} failed in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
