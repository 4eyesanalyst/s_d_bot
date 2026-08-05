"""One scan, then exit. The entry point for scheduled (CI) hosting.

`run_signals.py` loops forever, which suits a server. GitHub Actions instead
starts a fresh container on a schedule, so the bot must do exactly one pass and
leave -- carrying its state in and out through the repository.

Exit codes: 0 = scanned fine, 1 = something failed (the run shows red).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone

from sd_bot.config import Config
from sd_bot.feed import Feed
from sd_bot.notify import build
from sd_bot.scanner import SignalScanner
from sd_bot.sessions import in_session, is_week_close


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--symbols")
    p.add_argument("--force", action="store_true",
                   help="scan even outside session hours")
    args = p.parse_args()

    now = int(datetime.now(timezone.utc).timestamp())
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    cfg = Config.load(args.config)
    if args.symbols:
        cfg.execution.symbols = [
            s.strip() for s in args.symbols.split(",") if s.strip()
        ]
    cfg.validate()

    # A scheduled runner has no memory of "did I already say this", so the
    # heartbeat is driven by the workflow instead of a timer inside the process.
    cfg.alerts.heartbeat_hours = 0

    print(f"[{stamp}Z] scanning {', '.join(cfg.execution.symbols)}")

    if not args.force and (not in_session(now, cfg.execution)
                           or is_week_close(now, cfg.execution)):
        print("outside session hours -- nothing to do")
        return 0

    out = build(cfg)
    for line in out.preflight():
        print(line)

    scanner = SignalScanner(cfg, out, Feed("dukascopy"),
                            state_dir=cfg.alerts.directory)
    before = len(scanner.active)

    try:
        scanner.poll()
    except Exception:
        traceback.print_exc()
        return 1

    after = len(scanner.active)
    print(f"live signals: {before} -> {after}")
    for sig in scanner.active.values():
        print(f"  {sig.symbol} {'BUY' if sig.is_long else 'SELL'} "
              f"{sig.status} entry={sig.entry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
