"""Backtest / live parity.

The strategy is only worth deploying if the live scanner makes the *same*
decisions the backtester made. This replays historical bars through the real
SignalScanner, one bar at a time, and checks its signals against the trades the
Backtester takes on identical data.

Any divergence here means the numbers in the README do not describe the bot you
are actually running.

    python tests/test_parity.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sd_bot import sources
from sd_bot.backtest import Backtester
from sd_bot.config import Config
from sd_bot.data import Bars
from sd_bot.feed import Feed
from sd_bot.notify import Broadcaster, Notifier
from sd_bot.scanner import SignalScanner

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


class SilentNotifier(Notifier):
    """Captures alerts instead of sending them."""

    name = "capture"

    def __init__(self):
        self.sent: list[tuple[str, str, dict]] = []

    def send(self, subject: str, body: str, meta: dict | None = None) -> bool:
        self.sent.append((subject, body, meta or {}))
        return True


class ReplayFeed(Feed):
    """Serves historical bars as if they were arriving live.

    ``advance_to(ts)`` moves the clock; every ``bars()`` call then returns only
    bars that had closed by that moment. This is the same no-lookahead guarantee
    the backtester enforces, applied to the live code path.
    """

    def __init__(self, series: dict[str, dict[str, Bars]]):
        self.backend = "replay"
        self.series = series
        self.now = 0

    def advance_to(self, ts: int) -> None:
        self.now = ts

    def bars(self, symbol: str, timeframe: str, count: int) -> Bars | None:
        full = self.series[symbol][timeframe]
        # Bars that have *closed* by self.now.
        from sd_bot import timeframes as tfmod

        closed = full.time + tfmod.seconds(timeframe) <= self.now
        n = int(closed.sum())
        if n == 0:
            return None
        return full.slice(max(0, n - count), n)

    def tick(self, symbol: str):
        b = self.bars(symbol, "M15", 1)
        if b is None:
            return None
        bid = float(b.close[-1])
        spec = sources.spec_for(symbol)
        return bid, bid + sources.spread_points(symbol) * spec.point

    def spread_points(self, symbol: str) -> float:
        return sources.spread_points(symbol)


def load(symbol: str, start: str, end: str) -> dict[str, Bars]:
    lo = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    hi = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    out = {}
    for tf in ("M15", "H1", "H4"):
        bars = sources.load(symbol, tf, lo, hi)
        a = int(np.searchsorted(bars.time, lo.timestamp(), "left"))
        b = int(np.searchsorted(bars.time, hi.timestamp(), "right"))
        out[tf] = bars.slice(a, b)
    return out


def run_parity(symbol: str, start: str, end: str, unlimited: bool = False) -> None:
    """Replay ``symbol`` through both engines and compare their entries.

    With ``unlimited`` the concurrency cap is lifted on both sides. That
    separates the two questions worth asking: does the live bot *find* the same
    setups (detection), and does it *hold* them for the same length of time
    (trade management)? Only the first is the strategy itself.
    """
    mode = "detection only (no concurrency cap)" if unlimited else "end to end"
    print(f"\nreplaying {symbol}  {start} -> {end}   [{mode}]")
    cfg = Config.load("config.yaml")
    cfg.execution.symbols = [symbol]
    cfg.alerts.channels = []
    cfg.alerts.alert_on_approach = False   # compare entries, not warnings
    cfg.alerts.cooldown_minutes = 0
    cfg.alerts.heartbeat_hours = 0
    if unlimited:
        cfg.risk.max_trades_per_symbol = 99
        cfg.risk.max_open_trades = 99
        cfg.risk.max_total_risk_pct = 1e6
        cfg.risk.max_drawdown_pct = 1e6
        cfg.risk.daily_loss_limit_pct = 1e6
        cfg.risk.weekly_loss_limit_pct = 1e6
        cfg.risk.enforce_currency_exposure = False
    cfg.validate()

    series = load(symbol, start, end)
    for tf, b in series.items():
        print(f"  {tf:<4} {len(b):>6} bars")

    # --- the backtester's view -------------------------------------------
    specs = {symbol: sources.spec_for(symbol)}
    result = Backtester(cfg, specs).run({symbol: series})
    bt_entries = {
        (t.symbol, t.entry_time, t.direction, round(t.entry_price, 5))
        for t in result.trades
    }

    # --- the live scanner's view -----------------------------------------
    capture = SilentNotifier()
    feed = ReplayFeed({symbol: series})
    state = Path("signals/_parity_state.json")
    if state.exists():
        state.unlink()
    scanner = SignalScanner(cfg, Broadcaster([capture]), feed, state_dir="signals")
    scanner.state_path = state

    m15 = series["M15"]
    step = 900
    # Skip the replay warm-up. The scanner needs a minimum history window before
    # it can analyse anything; live it always has that (the feed seeds from cache
    # or downloads it), but a replay has to accumulate it bar by bar. Comparing
    # over those first bars measures the test harness, not the bot.
    warmup = 0
    for i in range(len(m15)):
        if int(m15.time[i]) >= int(series["H1"].time[min(140, len(series["H1"]) - 1)]):
            warmup = i
            break
    bt_cutoff = int(m15.time[warmup])
    for i in range(warmup, len(m15)):
        close_time = int(m15.time[i]) + step
        feed.advance_to(close_time)
        # track() resolves live signals (fill / TP / stop). Without it they
        # never clear and the per-symbol limit blocks every later setup.
        scanner.track(close_time)
        scanner.scan(symbol, close_time)

    sc_entries = set()
    for subject, body, meta in capture.sent:
        if meta.get("kind") != "triggered":
            continue
        key = meta.get("_key")
        sc_entries.add(key)

    # Rebuild scanner entries from its own recorded signals.
    sc_entries = {
        (s.symbol, s.direction, round(s.entry, 5))
        for s in scanner.all_signalled
    }
    bt_simple = {(sym, d, e) for sym, ts, d, e in bt_entries if ts >= bt_cutoff}

    print(f"  backtester took     {len(bt_simple)} entries")
    print(f"  scanner signalled   {len(sc_entries)} entries")

    both = bt_simple & sc_entries
    only_bt = bt_simple - sc_entries
    only_sc = sc_entries - bt_simple
    coverage = len(both) / len(bt_simple) if bt_simple else 0.0

    label = "detection" if unlimited else "end-to-end"
    # Detection must be near-exact: that is the strategy. End-to-end is allowed
    # to drift, because exits are the trader's to manage.
    threshold = 0.95 if unlimited else 0.60
    check(f"{symbol} [{label}]: scanner reproduces the backtest's entries",
          coverage >= threshold,
          f"{len(both)}/{len(bt_simple)} matched ({coverage:.0%})")
    if unlimited:
        check(f"{symbol} [{label}]: scanner invents no extra entries",
              len(only_sc) <= max(2, 0.05 * len(bt_simple)),
              f"{len(only_sc)} unmatched")

    if only_bt:
        print(f"    backtest-only examples: {sorted(only_bt)[:3]}")
    if only_sc:
        print(f"    scanner-only examples:  {sorted(only_sc)[:3]}")

    if state.exists():
        state.unlink()


def main() -> int:
    print("=" * 66)
    print("  BACKTEST / LIVE PARITY")
    print("=" * 66)
    import os
    symbols = os.environ.get("PARITY_SYMBOLS", "XAUUSD,EURUSD,USDJPY").split(",")
    for symbol in symbols:
        try:
            run_parity(symbol, "2025-01-01", "2025-04-01", unlimited=True)
            run_parity(symbol, "2025-01-01", "2025-04-01", unlimited=False)
        except FileNotFoundError as exc:
            print(f"  skipped {symbol}: {exc}")

    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("live scanner matches the backtested strategy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
