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
    p.add_argument("--heartbeat", action="store_true",
                   help="send a status message instead of scanning")
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

    if not args.force and not args.heartbeat and (
            not in_session(now, cfg.execution)
            or is_week_close(now, cfg.execution)):
        print("outside session hours -- nothing to do")
        return 0

    out = build(cfg)
    for line in out.preflight():
        print(line)

    scanner = SignalScanner(cfg, out, Feed("dukascopy"),
                            state_dir=cfg.alerts.directory)

    if args.heartbeat:
        # This strategy signals roughly 2-3 times a week, so silence is normal
        # and a dead bot looks exactly like a quiet market. The daily heartbeat
        # is the only thing that tells them apart.
        lines = [f"Watching {', '.join(cfg.execution.symbols)}",
                 f"Session {min(cfg.execution.session_hours_utc):02d}:00-"
                 f"{max(cfg.execution.session_hours_utc):02d}:59 UTC, Mon-Fri",
                 ""]
        if scanner.active:
            lines.append(f"{len(scanner.active)} live signal(s):")
            for s in scanner.active.values():
                lines.append(
                    f"  {s.symbol} {'BUY' if s.is_long else 'SELL'} "
                    f"{s.status} @ {s.entry}  stop {s.stop}  tp2 {s.tp2}"
                )
        else:
            lines.append("No live signals. Waiting for price to reach a zone.")
        lines += ["", "If this message stops arriving, the bot has stopped."]
        scanner.emit("STATUS", "", "Signal bot alive", "\n".join(lines),
                     {"kind": "heartbeat"})
        print("heartbeat sent")
        return 0

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
