"""Long-running watcher for GitHub Actions.

``github_poll.py`` scans once and exits, so latency is whatever GitHub's
scheduler decides -- measured at 60-90 minutes, and occasionally a whole
weekend. That dominates every other delay in the system.

This runs a polling loop *inside* one Actions job instead. Public repositories
get unlimited Actions minutes and a job may run for six hours, so a handful of
long jobs covers the session with the scheduler involved only at handover.

Latency drops from "up to 90 minutes" to "one poll interval".

    python github_watch.py --max-minutes 320 --interval 90
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import traceback
from datetime import datetime, timezone

from sd_bot.config import Config
from sd_bot.feed import Feed
from sd_bot.notify import build
from sd_bot.scanner import SignalScanner
from sd_bot.sessions import in_session, is_week_close

_stop = False


def _handle(signum, frame):
    """GitHub sends SIGTERM when it cancels a job or the timeout hits."""
    global _stop
    _stop = True
    print(f"\n[{_now()}] signal {signum}: finishing this pass and saving state",
          flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--max-minutes", type=float, default=110.0,
                   help="exit cleanly before the Actions job timeout")
    p.add_argument("--interval", type=int, default=90,
                   help="seconds between passes")
    args = p.parse_args()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    cfg = Config.load(args.config)
    cfg.validate()
    cfg.alerts.heartbeat_hours = 0      # the daily heartbeat is handled below

    out = build(cfg)
    preflight = out.preflight()
    for line in preflight:
        print(line, flush=True)

    # A watcher that cannot deliver is worse than no watcher: it consumes the
    # setups, marks them alerted, and tells nobody.
    if [ln for ln in preflight if ln.strip().startswith("DEAD")]:
        print("\nFATAL: a configured alert channel is not working. "
              "Check the repository secrets.", flush=True)
        return 1

    scanner = SignalScanner(cfg, out, Feed("dukascopy"),
                            state_dir=cfg.alerts.directory)

    print(f"[{_now()}] watching {', '.join(cfg.execution.symbols)} "
          f"every {args.interval}s for up to {args.max_minutes:.0f} min",
          flush=True)

    deadline = time.time() + args.max_minutes * 60
    passes = 0
    failures = 0

    while not _stop and time.time() < deadline:
        now = int(datetime.now(timezone.utc).timestamp())

        # The heartbeat is a liveness signal, so it goes out regardless of
        # whether the market is open to us.
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if scanner.last_heartbeat_day != today:
            try:
                from github_poll import send_heartbeat

                send_heartbeat(cfg, scanner)
                scanner.last_heartbeat_day = today
                scanner._save()
                print(f"[{_now()}] daily heartbeat sent", flush=True)
            except Exception:
                traceback.print_exc()

        if in_session(now, cfg.execution) and not is_week_close(now, cfg.execution):
            try:
                scanner.poll()
                passes += 1
                failures = 0
            except Exception:
                failures += 1
                traceback.print_exc()
                # Back off on repeated failures rather than hammering the feed.
                if failures >= 3:
                    time.sleep(min(300, 30 * failures))
        else:
            # Outside the window there is nothing to do but stay alive; the next
            # job takes over long before the session reopens.
            print(f"[{_now()}] outside session, idling", flush=True)
            time.sleep(min(600, args.interval * 5))
            continue

        remaining = (deadline - time.time()) / 60
        if passes % 20 == 1:
            print(f"[{_now()}] pass {passes}, {remaining:.0f} min left, "
                  f"{len(scanner.active)} live signal(s)", flush=True)

        for _ in range(args.interval):
            if _stop or time.time() >= deadline:
                break
            time.sleep(1)

    scanner._save()
    print(f"[{_now()}] finished after {passes} passes; state saved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
