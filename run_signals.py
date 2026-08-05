"""Supervisor for unattended operation.

Keeps the scanner alive across crashes, network outages and broken pipes, with
exponential backoff so a persistent failure does not hammer the data provider.
Use this as the entry point on a server; use `python -m sd_bot signals` when you
are sitting in front of it.

    python run_signals.py
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone

MIN_BACKOFF = 30
MAX_BACKOFF = 900

_stop = False


def _handle(signum, frame):
    global _stop
    _stop = True
    print(f"\n[{_stamp()}] signal {signum} received, shutting down cleanly", flush=True)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--symbols")
    p.add_argument("--feed", default="auto",
                   choices=["auto", "mt5", "dukascopy"])
    p.add_argument("--max-restarts", type=int, default=0,
                   help="0 = unlimited")
    args = p.parse_args()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    from sd_bot.config import Config
    from sd_bot.feed import Feed
    from sd_bot.notify import build
    from sd_bot.scanner import SignalScanner

    restarts = 0
    backoff = MIN_BACKOFF

    while not _stop:
        try:
            cfg = Config.load(args.config)
            if args.symbols:
                cfg.execution.symbols = [
                    s.strip() for s in args.symbols.split(",") if s.strip()
                ]
            cfg.validate()

            out = build(cfg)
            for line in out.preflight():
                print(line, flush=True)

            scanner = SignalScanner(
                cfg, out, Feed(args.feed), state_dir=cfg.alerts.directory
            )
            if restarts:
                scanner.emit(
                    "STATUS", "", "Signal bot restarted",
                    f"Recovered after a failure (restart #{restarts}).\n"
                    f"Watching {', '.join(cfg.execution.symbols)} again.",
                    {"kind": "restart"},
                )
            backoff = MIN_BACKOFF   # a clean start resets the penalty
            scanner.run()
            return 0                # run() only returns on a deliberate stop

        except KeyboardInterrupt:
            print(f"[{_stamp()}] stopped by user", flush=True)
            return 0
        except Exception:
            restarts += 1
            print(f"[{_stamp()}] crashed (restart #{restarts}):", flush=True)
            traceback.print_exc()
            if args.max_restarts and restarts >= args.max_restarts:
                print(f"[{_stamp()}] giving up after {restarts} restarts", flush=True)
                return 1
            print(f"[{_stamp()}] retrying in {backoff}s", flush=True)
            for _ in range(backoff):
                if _stop:
                    return 0
                time.sleep(1)
            backoff = min(backoff * 2, MAX_BACKOFF)

    return 0


if __name__ == "__main__":
    sys.exit(main())
