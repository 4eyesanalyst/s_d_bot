"""Parameter experiments with an out-of-sample guard.

Every variant is measured twice: on the in-sample period it may be tuned
against, and on a held-out period it never saw. A variant that only works
in-sample is curve fitting, and the report is laid out to make that obvious.

    python sweep.py --list baseline
    python sweep.py --grid frequency
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from datetime import datetime, timezone

import numpy as np

from sd_bot import sources
from sd_bot.backtest import Backtester
from sd_bot.config import Config
from sd_bot.data import Bars
from sd_bot.stats import compute

IS_START, IS_END = "2022-01-01", "2025-01-01"   # in sample: tune here
OOS_START, OOS_END = "2025-01-01", "2026-08-01"  # out of sample: judge here


# Every timeframe any variant might ask for, loaded once up front.
TIMEFRAMES = ("M15", "H1", "H4", "D1")


def load_all(symbols: list[str], cfg: Config) -> dict[str, dict[str, Bars]]:
    from datetime import datetime, timezone

    start = datetime.fromisoformat(IS_START).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(OOS_END).replace(tzinfo=timezone.utc)
    data = {}
    for sym in symbols:
        series = {}
        for tf in TIMEFRAMES:
            try:
                series[tf] = sources.load(sym, tf, start, end)
            except Exception as exc:
                print(f"  {sym} {tf}: unavailable ({exc})")
        if series:
            data[sym] = series
    return data


def window(data: dict[str, dict[str, Bars]], start: str, end: str):
    """Slice every series to a date range, without re-reading from disk."""
    lo = datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp()
    hi = datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp()
    out = {}
    for symbol, series in data.items():
        cut = {}
        for tf, bars in series.items():
            a = int(np.searchsorted(bars.time, lo, "left"))
            b = int(np.searchsorted(bars.time, hi, "right"))
            cut[tf] = bars.slice(a, b)
        if all(len(b) > 200 for b in cut.values()):
            out[symbol] = cut
    return out


def apply(cfg: Config, overrides: dict) -> Config:
    cfg = copy.deepcopy(cfg)
    for path, value in overrides.items():
        section, key = path.split(".")
        setattr(getattr(cfg, section), key, value)
    cfg.validate()
    return cfg


def measure(cfg: Config, data, specs, cache) -> dict:
    result = Backtester(cfg, specs, analysis_cache=cache).run(data)
    s = compute(result)
    return {
        "trades": s.trades,
        "per_day": s.trades_per_day,
        "per_week": s.trades_per_week,
        "win": s.win_rate,
        "pf": s.profit_factor,
        "exp": s.expectancy_r,
        "ret": s.return_pct,
        "dd": s.max_drawdown_pct,
        "risk_pips": s.avg_risk_pips,
        "win_pips": s.avg_win_pips,
        "move_pips": s.avg_move_pips,
        "result": result,
        "stats": s,
    }


def row(label: str, a: dict, b: dict) -> str:
    def fmt(m):
        pf = "  inf" if m["pf"] == float("inf") else f"{m['pf']:5.2f}"
        return (f"{m['trades']:>5} {m['per_week']:>5.2f} {m['win']:>5.1f} {pf} "
                f"{m['exp']:>+6.2f} {m['ret']:>+7.1f} {m['dd']:>5.1f} "
                f"{m['win_pips']:>6.0f}")
    return f"{label:<30} | {fmt(a)} | {fmt(b)}"


HEADER = (
    f"{'variant':<30} | "
    f"{'trds':>5} {'t/wk':>5} {'win%':>5} {'  pf':>5} {'expR':>6} "
    f"{'ret%':>7} {'dd%':>5} {'wpips':>6} | "
    f"{'trds':>5} {'t/wk':>5} {'win%':>5} {'  pf':>5} {'expR':>6} "
    f"{'ret%':>7} {'dd%':>5} {'wpips':>6}"
)


GRIDS: dict[str, list[tuple[str, dict]]] = {
    "baseline": [
        ("baseline (as shipped)", {}),
    ],
    # Does constraining the stop band lift the average move toward 100 pips?
    "stopband": [
        ("no band", {}),
        ("stop 15-40p", {"signal.min_stop_pips": 15, "signal.max_stop_pips": 40}),
        ("stop 20-50p", {"signal.min_stop_pips": 20, "signal.max_stop_pips": 50}),
        ("stop 25-60p", {"signal.min_stop_pips": 25, "signal.max_stop_pips": 60}),
        ("stop 30-70p", {"signal.min_stop_pips": 30, "signal.max_stop_pips": 70}),
    ],
    # How much frequency can we buy without weakening the zone rules?
    "frequency": [
        ("3 open, score 65", {}),
        ("5 open", {"risk.max_open_trades": 5, "risk.max_total_risk_pct": 2.5}),
        ("8 open", {"risk.max_open_trades": 8, "risk.max_total_risk_pct": 3.0}),
        ("8 open, ccy cap 3", {"risk.max_open_trades": 8,
                               "risk.max_total_risk_pct": 3.0,
                               "risk.max_currency_exposure": 3}),
        ("8 open, all hours", {"risk.max_open_trades": 8,
                               "risk.max_total_risk_pct": 3.0,
                               "risk.max_currency_exposure": 3,
                               "execution.session_hours_utc": list(range(24))}),
    ],
    # Which quality bar actually pays?
    "quality": [
        ("score 55", {"zone.min_score": 55}),
        ("score 60", {"zone.min_score": 60}),
        ("score 65 (default)", {}),
        ("score 70", {"zone.min_score": 70}),
        ("score 75", {"zone.min_score": 75}),
        ("score 80", {"zone.min_score": 80}),
    ],
    "filters": [
        ("score 70 (ref)", {"zone.min_score": 70}),
        ("+ RR 2.5", {"zone.min_score": 70, "signal.min_risk_reward": 2.5}),
        ("+ RR 3.5", {"zone.min_score": 70, "signal.min_risk_reward": 3.5}),
        ("+ departure 4x", {"zone.min_score": 70, "zone.min_departure_ratio": 4.0}),
        ("+ departure 5x", {"zone.min_score": 70, "zone.min_departure_ratio": 5.0}),
        ("+ 2 tests allowed", {"zone.min_score": 70, "zone.max_tests": 2}),
        ("+ base<=3", {"zone.min_score": 70, "zone.max_base_candles": 3}),
        ("+ leg-out 1.5atr", {"zone.min_score": 70, "zone.min_leg_out_atr": 1.5}),
    ],
    # Single-instrument frequency levers.
    "throughput": [
        ("1 trade at a time (ref)", {"zone.min_score": 70}),
        ("2 concurrent", {"zone.min_score": 70, "risk.max_trades_per_symbol": 2,
                          "risk.max_open_trades": 2}),
        ("3 concurrent", {"zone.min_score": 70, "risk.max_trades_per_symbol": 3,
                          "risk.max_open_trades": 3}),
        ("3 concurrent, all hours", {"zone.min_score": 70,
                                     "risk.max_trades_per_symbol": 3,
                                     "risk.max_open_trades": 3,
                                     "execution.session_hours_utc": list(range(24))}),
        ("H1 zones, 3 concurrent", {"zone.min_score": 70,
                                    "execution.zone_timeframe": "H1",
                                    "execution.bias_timeframe": "H4",
                                    "risk.max_trades_per_symbol": 3,
                                    "risk.max_open_trades": 3}),
    ],
    # How many trades can this instrument physically produce? The last rows
    # dismantle the rule set entirely -- not as a proposal, but to establish the
    # ceiling that frequency targets have to live under.
    "ceiling": [
        ("shipped rules", {}),
        ("score 0 (no grading)", {"zone.min_score": 0}),
        ("+ 3 concurrent", {"zone.min_score": 0, "risk.max_trades_per_symbol": 3,
                            "risk.max_open_trades": 3}),
        ("+ all hours", {"zone.min_score": 0, "risk.max_trades_per_symbol": 3,
                         "risk.max_open_trades": 3,
                         "execution.session_hours_utc": list(range(24))}),
        ("+ RR 1.5, 3 tests", {"zone.min_score": 0, "risk.max_trades_per_symbol": 3,
                               "risk.max_open_trades": 3,
                               "execution.session_hours_utc": list(range(24)),
                               "signal.min_risk_reward": 1.5, "zone.max_tests": 3}),
        ("+ departure 1.5x (all zones)",
         {"zone.min_score": 0, "risk.max_trades_per_symbol": 3,
          "risk.max_open_trades": 3,
          "execution.session_hours_utc": list(range(24)),
          "signal.min_risk_reward": 1.5, "zone.max_tests": 3,
          "zone.min_departure_ratio": 1.5, "zone.min_leg_out_atr": 0.4}),
        ("H1 zones, everything open",
         {"zone.min_score": 0, "execution.zone_timeframe": "H1",
          "execution.bias_timeframe": "H4",
          "risk.max_trades_per_symbol": 3, "risk.max_open_trades": 3,
          "execution.session_hours_utc": list(range(24)),
          "signal.min_risk_reward": 1.5, "zone.max_tests": 3,
          "zone.min_departure_ratio": 1.5, "zone.min_leg_out_atr": 0.4}),
    ],
    # Best honest configuration for gold: H1 zones (the highest-frequency
    # timeframe that still yields ~100 pip moves), gold's real 23/5 session,
    # and the quality bar swept across.
    "final": [
        ("H1 24h, score 0", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 0}),
        ("H1 24h, score 55", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 55}),
        ("H1 24h, score 65", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 65}),
        ("H1 24h, score 70", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 70}),
        ("H1 24h, score 75", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 75}),
        ("H1 24h, s65, no BE", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 65,
                                 "signal.breakeven_after_tp1": False}),
        ("H1 24h, s65, tp1 1.5R", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 65,
                                    "signal.tp1_r": 1.5}),
        ("H1 24h, s65, RR 2.5", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 65,
                                  "signal.min_risk_reward": 2.5}),
        ("H1 24h, s65, min stop 30p", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 65,
                                        "signal.min_stop_pips": 30}),
    ],
    # Target the ~100 pip move explicitly. Move size ~= stop x reward ratio, so
    # the stop floor is the lever; each notch up trades frequency for size.
    "hundred": [
        ("s55, no BE, no floor", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 55,
                                   "signal.breakeven_after_tp1": False}),
        ("s55, no BE, stop>=20p", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 55,
                                    "signal.breakeven_after_tp1": False,
                                    "signal.min_stop_pips": 20}),
        ("s55, no BE, stop>=30p", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 55,
                                    "signal.breakeven_after_tp1": False,
                                    "signal.min_stop_pips": 30}),
        ("s55, no BE, stop>=40p", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 55,
                                    "signal.breakeven_after_tp1": False,
                                    "signal.min_stop_pips": 40}),
        ("s0, no BE, stop>=30p", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 0,
                                   "signal.breakeven_after_tp1": False,
                                   "signal.min_stop_pips": 30}),
        ("s55, no BE, 30p, tp1 2.5R", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 55,
                                        "signal.breakeven_after_tp1": False,
                                        "signal.min_stop_pips": 30,
                                        "signal.tp1_r": 2.5}),
        ("s55, no BE, 30p, run tp2", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24)), "zone.min_score": 55,
                                       "signal.breakeven_after_tp1": False,
                                       "signal.min_stop_pips": 30,
                                       "signal.tp1_fraction": 1.0}),
    ],
    # Which trading window? London opens 07:00 UTC, New York closes ~21:00 UTC.
    "sessions": [
        ("24h (all hours)", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(24))}),
        ("London+NY 07-21", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(7, 21))}),
        ("London+NY 07-22", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(7, 22))}),
        ("London+NY 08-22", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(8, 22))}),
        ("London only 07-16", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(7, 16))}),
        ("New York only 12-21", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(12, 21))}),
        ("overlap only 12-16", {"execution.zone_timeframe": "H1", "execution.bias_timeframe": "H4", "execution.session_hours_utc": list(range(12, 16))}),
    ],
    # tp1 1.5R beat baseline in BOTH the gold-only and portfolio proof runs.
    # Before adopting it, confirm it holds out of sample rather than being the
    # one lucky cell in a grid.
    "tp1check": [
        ("tp1 2.0R (current)", {}),
        ("tp1 1.5R", {"signal.tp1_r": 1.5}),
        ("tp1 1.25R", {"signal.tp1_r": 1.25}),
        ("tp1 1.75R", {"signal.tp1_r": 1.75}),
    ],
    # Exit management: does breakeven help or cost?
    "exits": [
        ("tp1 2R + BE (default)", {}),
        ("tp1 2R, no BE", {"signal.breakeven_after_tp1": False}),
        ("tp1 1.5R + BE", {"signal.tp1_r": 1.5}),
        ("tp1 3R + BE", {"signal.tp1_r": 3.0}),
        ("no partial, run to tp2", {"signal.tp1_fraction": 1.0}),
        ("fixed 4R target", {"signal.tp2_mode": "fixed_r", "signal.tp2_r": 4.0}),
    ],
    "entry": [
        ("limit at proximal", {}),
        ("limit at 50% depth", {"signal.entry_depth": 0.5}),
        ("confirmation", {"signal.entry_mode": "confirmation"}),
    ],
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="baseline", choices=sorted(GRIDS))
    p.add_argument("--symbols", default="")
    p.add_argument("--base", default="config.yaml")
    args = p.parse_args()

    base = Config.load(args.base)
    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        or sources.DEFAULT_BASKET
    )
    base.execution.symbols = symbols
    base.backtest.start, base.backtest.end = IS_START, OOS_END

    print(f"loading {len(symbols)} symbols...", flush=True)
    data = load_all(symbols, base)
    is_data = window(data, IS_START, IS_END)
    oos_data = window(data, OOS_START, OOS_END)
    specs = {s: sources.spec_for(s) for s in data}
    print(f"in-sample  {IS_START} -> {IS_END}   {len(is_data)} symbols")
    print(f"out-sample {OOS_START} -> {OOS_END}   {len(oos_data)} symbols\n")

    cache: dict = {}
    variants = GRIDS[args.grid]
    print(" " * 31 + "|" + " IN SAMPLE (tuned on)".center(48)
          + "|" + " OUT OF SAMPLE (held back)".center(48))
    print(HEADER)
    print("-" * len(HEADER))

    for label, overrides in variants:
        t0 = time.time()
        cfg = apply(base, overrides)
        a = measure(cfg, is_data, specs, cache)
        b = measure(cfg, oos_data, specs, cache)
        print(row(label, a, b) + f"   {time.time() - t0:4.0f}s", flush=True)

    print("\nwpips = average pips gained on winning trades.")
    print("A variant that looks good only on the left half is curve fitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
