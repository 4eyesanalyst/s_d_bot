"""Does the strategy actually have an edge, or is it luck and trend?

A backtest that made money proves nothing on its own. Four questions have to be
answered before the answer is "yes":

1. NULL HYPOTHESIS -- do supply/demand entries beat *random* entries that get
   the same stops, the same targets, the same session filter and the same
   long/short mix? If not, the zones are decoration and what you actually have
   is trend exposure plus exit management. This is the test that matters most.

2. CONFIDENCE -- bootstrap the trade population. If the 5th percentile of
   expectancy is below zero, the result is not distinguishable from luck.

3. SENSITIVITY -- perturb each parameter. An edge that only exists at one exact
   setting is curve fitting, not a strategy.

4. COSTS -- how much spread and slippage does the edge absorb before it dies?
   Your broker is not Dukascopy.

    python prove.py --symbols XAUUSD,EURUSD,USDJPY
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from datetime import datetime, timezone

import numpy as np

from sd_bot import sources
from sd_bot.backtest import Backtester
from sd_bot.config import Config
from sd_bot.sessions import in_session, is_week_close
from sd_bot.stats import compute
from sd_bot.trades import LONG, ClosedTrade, pip_size

TIMEFRAMES = ("M15", "H1", "H4")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load(symbol: str, start: str, end: str) -> dict:
    lo = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    hi = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    out = {}
    for tf in TIMEFRAMES:
        bars = sources.load(symbol, tf, lo, hi)
        a = int(np.searchsorted(bars.time, lo.timestamp(), "left"))
        b = int(np.searchsorted(bars.time, hi.timestamp(), "right"))
        out[tf] = bars.slice(a, b)
    return out


def run(cfg: Config, data: dict, cache: dict | None = None):
    specs = {s: sources.spec_for(s) for s in data}
    return Backtester(cfg, specs, analysis_cache=cache).run(data)


# --------------------------------------------------------------------------
# 1. null hypothesis: random entries, matched in every other respect
# --------------------------------------------------------------------------

def simulate_random_entries(
    data: dict,
    real_trades: list[ClosedTrade],
    cfg: Config,
    rng: random.Random,
) -> list[float]:
    """Trade the same instruments at random times, everything else matched.

    Each simulated trade copies a real trade's direction and stop distance, then
    picks a random entry bar inside the trading session. Exits use the same
    rules: TP1 at tp1_r closing a fraction, then TP2, stop otherwise, evaluated
    bar by bar. Returns the R outcome of each simulated trade.

    Matching the direction mix is deliberate. Gold rose 250% over this period,
    so purely random directions would lose to any long-biased system and flatter
    the strategy for the wrong reason. Holding direction fixed isolates the only
    thing being tested: whether *where* the zone put the entry mattered.
    """
    by_symbol: dict[str, list[ClosedTrade]] = {}
    for t in real_trades:
        by_symbol.setdefault(t.symbol, []).append(t)

    results: list[float] = []
    for symbol, trades in by_symbol.items():
        bars = data[symbol]["M15"]
        n = len(bars)
        if n < 500:
            continue
        high, low, close = bars.high, bars.low, bars.close
        tp1_r = cfg.signal.tp1_r
        frac = cfg.signal.tp1_fraction
        # Cost as a fraction of risk, mirroring what the backtester charges.
        spec = sources.spec_for(symbol)
        pts = sources.spread_points(symbol) + 2 * cfg.backtest.slippage_points
        cost_price = pts * spec.point

        for t in trades:
            risk = t.risk_distance
            if risk <= 0:
                continue
            # Random entry bar that is inside a tradeable session.
            for _ in range(40):
                i = rng.randrange(200, n - 200)
                ts = int(bars.time[i])
                if in_session(ts, cfg.execution) and not is_week_close(ts, cfg.execution):
                    break
            else:
                continue

            direction = t.direction
            entry = float(close[i])
            stop = entry - direction * risk
            tp1 = entry + direction * tp1_r * risk
            # Match the real trade's realised reward ratio ceiling.
            rr2 = max(abs(t.mfe_r), cfg.signal.min_risk_reward)
            tp2 = entry + direction * rr2 * risk

            banked = 0.0
            remaining = 1.0
            outcome = None
            for k in range(i + 1, min(i + 400, n)):
                h, lo_ = float(high[k]), float(low[k])
                hit_stop = lo_ <= stop if direction == LONG else h >= stop
                hit_tp1 = (h >= tp1 if direction == LONG else lo_ <= tp1) and remaining == 1.0
                hit_tp2 = h >= tp2 if direction == LONG else lo_ <= tp2

                if hit_stop:                      # pessimistic, as in the backtest
                    outcome = banked - remaining
                    break
                if hit_tp1:
                    banked += frac * tp1_r
                    remaining -= frac
                if hit_tp2:
                    outcome = banked + remaining * rr2
                    break
            if outcome is None:
                exit_price = float(close[min(i + 400, n - 1)])
                outcome = banked + remaining * (exit_price - entry) * direction / risk

            results.append(outcome - cost_price / risk)
    return results


# --------------------------------------------------------------------------
# 2. bootstrap confidence
# --------------------------------------------------------------------------

def bootstrap(values: list[float], iterations: int = 10_000,
              rng: random.Random | None = None) -> tuple[float, float, float]:
    """Resample with replacement. Returns (5th pct, median, 95th pct) of mean R."""
    rng = rng or random.Random(11)
    n = len(values)
    if n < 5:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.05 * iterations)],
            means[iterations // 2],
            means[int(0.95 * iterations)])


# --------------------------------------------------------------------------
# report helpers
# --------------------------------------------------------------------------

def hr(title: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="XAUUSD,EURUSD,USDJPY")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-01")
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    base = Config.load(args.config)
    base.execution.symbols = symbols
    base.validate()

    print("loading...", flush=True)
    data = {s: load(s, args.start, args.end) for s in symbols}
    cache: dict = {}

    result = run(base, data, cache)
    stats = compute(result)
    real_R = [t.r_multiple for t in result.trades]

    hr("BASELINE")
    print(f"  symbols       {', '.join(symbols)}")
    print(f"  period        {args.start} -> {args.end}")
    print(f"  trades        {stats.trades}")
    print(f"  win rate      {stats.win_rate:.1f}%")
    print(f"  profit factor {stats.profit_factor:.2f}")
    print(f"  expectancy    {stats.expectancy_r:+.3f}R")
    print(f"  return        {stats.return_pct:+.1f}%")
    print(f"  max drawdown  {stats.max_drawdown_pct:.2f}%")

    # -- 1. null hypothesis ------------------------------------------------
    hr("1. NULL HYPOTHESIS -- do the zones actually matter?")
    print("  Random entry times. Same instruments, same direction mix, same stop")
    print("  sizes, same targets, same sessions, same costs. The ONLY difference")
    print("  is that entries ignore supply and demand.\n")
    rng = random.Random(7)
    sim_means = []
    for s in range(args.sims):
        outcomes = simulate_random_entries(data, result.trades, base, rng)
        if outcomes:
            sim_means.append(sum(outcomes) / len(outcomes))
        if (s + 1) % 50 == 0:
            print(f"    {s + 1}/{args.sims} simulations", flush=True)

    sim_means.sort()
    actual = stats.expectancy_r
    better = sum(1 for m in sim_means if m >= actual)
    pvalue = (better + 1) / (len(sim_means) + 1)
    mean_random = sum(sim_means) / len(sim_means) if sim_means else 0.0

    print(f"\n  random entries    {mean_random:+.3f}R  "
          f"(5th {sim_means[int(0.05*len(sim_means))]:+.3f}, "
          f"95th {sim_means[int(0.95*len(sim_means))]:+.3f})")
    print(f"  strategy          {actual:+.3f}R")
    print(f"  edge over random  {actual - mean_random:+.3f}R")
    print(f"  p-value           {pvalue:.4f}   "
          f"({better} of {len(sim_means)} random runs matched or beat it)")
    verdict = ("PASS -- the zones add something beyond trend and exits"
               if pvalue < 0.05 else
               "FAIL -- indistinguishable from entering at random")
    print(f"\n  {verdict}")

    # -- 2. bootstrap ------------------------------------------------------
    hr("2. CONFIDENCE -- is the expectancy real or noise?")
    lo, mid, hi = bootstrap(real_R)
    print(f"  bootstrap of {len(real_R)} trades, 10,000 resamples")
    print(f"    5th percentile   {lo:+.3f}R")
    print(f"    median           {mid:+.3f}R")
    print(f"    95th percentile  {hi:+.3f}R")
    print(f"\n  {'PASS -- lower bound above zero' if lo > 0 else 'FAIL -- lower bound below zero; cannot rule out luck'}")

    # -- 3. sensitivity ----------------------------------------------------
    hr("3. SENSITIVITY -- does it survive being nudged?")
    print(f"  {'variant':<28}{'trades':>8}{'expR':>9}{'pf':>8}{'ret%':>9}")
    print("  " + "-" * 60)
    perturbations = [
        ("baseline", {}),
        ("min_score 45", {"zone.min_score": 45}),
        ("min_score 65", {"zone.min_score": 65}),
        ("min RR 2.5", {"signal.min_risk_reward": 2.5}),
        ("min RR 3.5", {"signal.min_risk_reward": 3.5}),
        ("departure 2.5x", {"zone.min_departure_ratio": 2.5}),
        ("departure 3.5x", {"zone.min_departure_ratio": 3.5}),
        ("tp1 1.5R", {"signal.tp1_r": 1.5}),
        ("tp1 2.5R", {"signal.tp1_r": 2.5}),
        ("swing lookback 3", {"structure.swing_lookback": 3}),
        ("max_tests 2", {"zone.max_tests": 2}),
    ]
    survived = 0
    for label, over in perturbations:
        cfg = copy.deepcopy(base)
        for path, value in over.items():
            section, key = path.split(".")
            setattr(getattr(cfg, section), key, value)
        cfg.validate()
        r = run(cfg, data, cache)
        st = compute(r)
        pf = "  inf" if st.profit_factor == math.inf else f"{st.profit_factor:6.2f}"
        flag = "" if st.expectancy_r > 0 else "   <-- negative"
        if st.expectancy_r > 0:
            survived += 1
        print(f"  {label:<28}{st.trades:>8}{st.expectancy_r:>+9.3f}{pf}"
              f"{st.return_pct:>+9.1f}{flag}")
    print(f"\n  {survived}/{len(perturbations)} variants profitable")
    print(f"  {'PASS -- edge is not balanced on one setting' if survived >= len(perturbations) * 0.8 else 'FAIL -- edge depends on exact parameters'}")

    # -- 4. costs ----------------------------------------------------------
    hr("4. COSTS -- how much friction kills it?")
    print(f"  {'extra cost':<28}{'trades':>8}{'expR':>9}{'pf':>8}{'ret%':>9}")
    print("  " + "-" * 60)
    breaking = None
    for extra in (0, 2, 5, 10, 20, 40):
        cfg = copy.deepcopy(base)
        cfg.backtest.slippage_points = extra
        cfg.validate()
        r = run(cfg, data, cache)
        st = compute(r)
        pf = "  inf" if st.profit_factor == math.inf else f"{st.profit_factor:6.2f}"
        print(f"  {f'+{extra} pts slippage':<28}{st.trades:>8}"
              f"{st.expectancy_r:>+9.3f}{pf}{st.return_pct:>+9.1f}")
        if breaking is None and st.expectancy_r <= 0:
            breaking = extra
    print(f"\n  edge dies at roughly +{breaking} points of extra cost"
          if breaking else "\n  edge survives every cost level tested")

    hr("SUMMARY")
    print(f"  null hypothesis   p={pvalue:.4f}  "
          f"{'PASS' if pvalue < 0.05 else 'FAIL'}")
    print(f"  bootstrap 5th     {lo:+.3f}R  {'PASS' if lo > 0 else 'FAIL'}")
    print(f"  sensitivity       {survived}/{len(perturbations)}  "
          f"{'PASS' if survived >= len(perturbations) * 0.8 else 'FAIL'}")
    print(f"  cost tolerance    {'+' + str(breaking) + ' pts' if breaking else 'robust'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
