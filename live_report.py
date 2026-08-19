"""Forward performance tracking. Read-only.

Reconstructs live trades from the alert log the bot already writes and compares
them against the backtest ledger. Touches nothing the bot runs: it reads
``signals/*.jsonl`` and ``results/ledger.csv``, and writes no state.

The point is not to judge the strategy on a handful of trades -- it is to build
the record that makes a judgement possible later. Adapting the rules before that
record exists is fitting to noise, which is how a positive-expectancy system
gets optimised into a negative one.

    python live_report.py              # read the local signals/ directory
    python live_report.py --github     # read the log from the repository
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

REPO = "4eyesanalyst/s_d_bot"

# A trade is over when one of these arrives.
CLOSERS = {"stop", "tp2", "trail_stop", "weekend_close", "breakeven", "invalidated"}

_ENTRY = re.compile(r"ENTRY\s+([0-9.]+)")
_RISK = re.compile(r"\(([0-9.]+)\s+pips risk\)")
_PIPS = re.compile(r"([+-]?[0-9.]+)\s+pips")
_GRADE = re.compile(r"grade\s+([A-D]\+?)")


@dataclass
class LiveTrade:
    symbol: str
    side: str
    opened: str
    entry: float = 0.0
    risk_pips: float = 0.0
    grade: str = "?"
    tp1_pips: float | None = None
    closed: str = ""
    reason: str = ""
    final_pips: float | None = None
    events: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.reason) and self.final_pips is not None

    def r_multiple(self, tp1_fraction: float = 0.5) -> float | None:
        """Blended R across the banked partial and the runner."""
        if not self.risk_pips or self.final_pips is None:
            return None
        runner = self.final_pips / self.risk_pips
        if self.tp1_pips is None:
            return runner
        first = self.tp1_pips / self.risk_pips
        return tp1_fraction * first + (1.0 - tp1_fraction) * runner


def load_alerts(use_github: bool) -> list[dict]:
    rows: list[dict] = []
    if use_github:
        url = "https://api.github.com/repos/" + REPO + "/contents/signals"
        try:
            with urllib.request.urlopen(url, timeout=30) as fh:
                listing = json.load(fh)
        except Exception as exc:
            print("could not list signals/ on GitHub: " + repr(exc))
            return rows
        for item in listing:
            if not item.get("name", "").endswith(".jsonl"):
                continue
            try:
                with urllib.request.urlopen(item["download_url"], timeout=30) as fh:
                    rows += _parse(fh.read().decode().splitlines())
            except Exception:
                continue
    else:
        for path in sorted(glob.glob(os.path.join("signals", "*.jsonl"))):
            with open(path, encoding="utf-8") as fh:
                rows += _parse(fh)
    rows.sort(key=lambda r: r.get("time", ""))
    return rows


def _parse(lines) -> list[dict]:
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def build_trades(alerts: list[dict]) -> list[LiveTrade]:
    """Stitch alerts into trades: triggered -> (tp1) -> terminal event."""
    open_by_symbol: dict[str, LiveTrade] = {}
    plans: dict[str, str] = {}
    trades: list[LiveTrade] = []

    for a in alerts:
        kind = a.get("kind")
        symbol = a.get("symbol") or ""
        body = a.get("body") or ""

        # The grade only appears on the SET ORDER alert, so remember it.
        if kind == "set_order" and symbol:
            g = _GRADE.search(body)
            plans[symbol] = g.group(1) if g else "?"
            continue

        if kind == "triggered" and symbol:
            t = LiveTrade(symbol=symbol, side=a.get("side", ""),
                          opened=a.get("time", ""), grade=plans.get(symbol, "?"))
            m = _ENTRY.search(body)
            if m:
                t.entry = float(m.group(1))
            m = _RISK.search(body)
            if m:
                t.risk_pips = float(m.group(1))
            t.events.append(kind)
            open_by_symbol[symbol] = t
            trades.append(t)
            continue

        t = open_by_symbol.get(symbol)
        if t is None:
            continue

        if kind == "tp1":
            m = _PIPS.search(body)
            if m:
                t.tp1_pips = abs(float(m.group(1)))
            t.events.append(kind)
        elif kind in CLOSERS:
            m = _PIPS.search(body)
            if m:
                t.final_pips = float(m.group(1))
            elif kind == "invalidated":
                t.final_pips = 0.0
            t.reason = kind
            t.closed = a.get("time", "")
            t.events.append(kind)
            open_by_symbol.pop(symbol, None)

    return trades


def ledger_baseline() -> dict:
    path = os.path.join("results", "ledger.csv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rs = [float(r["R"]) for r in rows if r.get("R")]
    if not rs:
        return {}
    out = {
        "trades": len(rs),
        "win_rate": sum(1 for r in rs if r > 0) / len(rs) * 100.0,
        "expectancy": sum(rs) / len(rs),
    }
    by_grade: dict[str, list] = {}
    for r in rows:
        if r.get("grade") and r.get("R"):
            by_grade.setdefault(r["grade"], []).append(float(r["R"]))
    out["by_grade"] = {g: (len(v), sum(v) / len(v)) for g, v in by_grade.items()}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--github", action="store_true",
                   help="read the alert log from the repository")
    p.add_argument("--tp1-fraction", type=float, default=0.5)
    args = p.parse_args()

    alerts = load_alerts(args.github)
    trades = build_trades(alerts)
    done = [t for t in trades if t.complete]
    base = ledger_baseline()

    print("=" * 74)
    print("  LIVE PERFORMANCE vs BACKTEST")
    print("=" * 74)
    print("  alerts read        %d" % len(alerts))
    print("  entries triggered  %d" % len(trades))
    print("  completed trades   %d" % len(done))
    if base:
        print("  backtest baseline  %d trades, %.1f%% win, %+.3fR"
              % (base["trades"], base["win_rate"], base["expectancy"]))
    print()

    rs = [r for r in (t.r_multiple(args.tp1_fraction) for t in done)
          if r is not None]

    if not rs:
        print("  No completed trades with a usable risk figure yet.")
        _open_positions(trades)
        _guidance(0)
        return 0

    wins = sum(1 for r in rs if r > 0)
    exp = sum(rs) / len(rs)
    live_win = wins / len(rs) * 100.0

    print("  %-20s%14s%14s%12s" % ("", "LIVE", "BACKTEST", "gap"))
    print("  " + "-" * 60)
    print("  %-20s%14d%14d" % ("trades", len(rs), base.get("trades", 0)))
    print("  %-20s%13.1f%%%13.1f%%%+11.1f"
          % ("win rate", live_win, base.get("win_rate", 0.0),
             live_win - base.get("win_rate", 0.0)))
    print("  %-20s%13.3fR%13.3fR%+11.3f"
          % ("expectancy", exp, base.get("expectancy", 0.0),
             exp - base.get("expectancy", 0.0)))
    print()

    print("  Trade by trade:")
    print("    %-12s%-9s%-6s%-7s%-14s%8s%8s"
          % ("opened", "symbol", "side", "grade", "reason", "pips", "R"))
    print("    " + "-" * 62)
    for t in done:
        r = t.r_multiple(args.tp1_fraction)
        if r is None:
            continue
        print("    %-12s%-9s%-6s%-7s%-14s%8.0f%8.2f"
              % (t.opened[:10], t.symbol, t.side or "-", t.grade,
                 t.reason, t.final_pips, r))

    by_grade: dict[str, list] = {}
    for t in done:
        r = t.r_multiple(args.tp1_fraction)
        if r is not None:
            by_grade.setdefault(t.grade, []).append(r)
    if by_grade:
        print()
        print("  By grade (live vs backtest):")
        print("    %-8s%5s%13s%13s" % ("grade", "n", "live expR", "backtest"))
        print("    " + "-" * 40)
        for g in ("A+", "A", "B", "C", "D", "?"):
            v = by_grade.get(g)
            if not v:
                continue
            b = base.get("by_grade", {}).get(g)
            btxt = ("%+.3fR" % b[1]) if b else "-"
            print("    %-8s%5d%+12.3fR%13s"
                  % (g, len(v), sum(v) / len(v), btxt))

    _open_positions(trades)
    _guidance(len(rs))
    return 0


def _open_positions(trades: list[LiveTrade]) -> None:
    live = [t for t in trades if not t.complete]
    if not live:
        return
    print()
    print("  Still open:")
    for t in live:
        print("    %s %s entry %s  (%s)"
              % (t.symbol, t.side or "-", t.entry, ", ".join(t.events)))


def _guidance(n: int) -> None:
    print()
    print("  " + "-" * 60)
    if n < 30:
        print("  %d completed trades. Far too few to conclude anything: at a" % n)
        print("  47% win rate with fat-tailed winners, even 30 trades can land")
        print("  anywhere between roughly -0.3R and +0.9R by chance alone.")
        print("  Keep collecting. Change nothing.")
    elif n < 100:
        print("  %d trades. Enough to catch a gross divergence -- live" % n)
        print("  expectancy far below zero would be worth investigating -- but")
        print("  not enough to justify tuning any parameter.")
    else:
        print("  %d trades. A comparison now means something. If live still" % n)
        print("  tracks the ledger, the edge is real forward. If it does not,")
        print("  re-run prove.py and the parity suite before changing a rule.")
    print("=" * 74)


if __name__ == "__main__":
    raise SystemExit(main())
