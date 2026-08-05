"""Engine tests. Run with `python tests/test_engine.py` or `pytest tests/`.

These check the parts that would fail silently and poison every result: zone
geometry, lookahead leakage, position sizing and the risk gates.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sd_bot.config import Config
from sd_bot.data import Bars, align, resample
from sd_bot.indicators import atr
from sd_bot.risk import RiskManager, currency_exposure, split_currencies
from sd_bot.scoring import curve_value, departure_value, freshness_value
from sd_bot.signals import build_plan
from sd_bot.structure import Structure, find_swings
from sd_bot.trades import LONG, SHORT, Position, SymbolSpec, TradePlan
from sd_bot.zones import DEMAND, SUPPLY, find_zones, settle

FAILURES: list[str] = []

# Fixtures need plausible epoch seconds: Bars validates timestamps on purpose.
EPOCH = 1_700_000_000  # 2023-11-14 UTC


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def make_bars(rows: list[tuple[float, float, float, float]], tf: str = "H1") -> Bars:
    o, h, l, c = (np.array([r[i] for r in rows], dtype=float) for i in range(4))
    n = len(rows)
    return Bars(
        symbol="TEST",
        timeframe=tf,
        time=np.arange(EPOCH, EPOCH + n * 3600, 3600, dtype=np.int64)[:n],
        open=o, high=h, low=l, close=c,
        volume=np.full(n, 100.0),
        spread=np.full(n, 5.0),
    )


# -- zone geometry ----------------------------------------------------------


def test_drop_base_rally():
    """A textbook DBR must be found, with body-proximal and wick-distal."""
    rows = []
    # 30 quiet bars to establish ATR around 0.0010.
    price = 1.1000
    for _ in range(30):
        rows.append((price, price + 0.0006, price - 0.0006, price - 0.0001))
        price -= 0.0001

    # Leg in: one hard drop (large body, small wicks).
    top = price
    rows.append((top, top + 0.0001, top - 0.0035, top - 0.0033))
    price = top - 0.0033

    # Base: two tiny indecision candles. Bodies define the proximal line.
    base_hi_body = price + 0.0002
    rows.append((price, price + 0.0005, price - 0.0004, price + 0.0001))
    rows.append((price + 0.0001, price + 0.0004, price - 0.0006, base_hi_body))
    base_low = price - 0.0006

    # Departure: three strong bullish bars, well over 3x the zone height.
    p = base_hi_body
    for _ in range(4):
        rows.append((p, p + 0.0036, p - 0.0001, p + 0.0034))
        p += 0.0034
    for _ in range(12):
        rows.append((p, p + 0.0004, p - 0.0004, p + 0.0002))
        p += 0.0002

    bars = make_bars(rows)
    cfg = Config()
    zones = find_zones(bars, cfg.zone, Structure(bars, 2))

    demand = [z for z in zones if z.kind == DEMAND]
    check("DBR detected", len(demand) >= 1, f"found {len(zones)} zones total")
    if not demand:
        return

    z = demand[0]
    check("pattern is DBR", z.pattern == "DBR", z.pattern)
    check(
        "proximal uses body top",
        abs(z.proximal - base_hi_body) < 1e-9,
        f"proximal={z.proximal:.5f} expected={base_hi_body:.5f}",
    )
    check(
        "distal uses base wick low",
        abs(z.distal - base_low) < 1e-9,
        f"distal={z.distal:.5f} expected={base_low:.5f}",
    )
    check("proximal above distal for demand", z.proximal > z.distal)
    check(
        "departure ratio clears the filter",
        z.departure_ratio >= cfg.zone.min_departure_ratio,
        f"{z.departure_ratio:.2f}",
    )
    check("created after the base", z.created > z.base_end)


def test_rally_base_drop():
    rows = []
    price = 1.1000
    for _ in range(30):
        rows.append((price, price + 0.0006, price - 0.0006, price + 0.0001))
        price += 0.0001

    bottom = price
    rows.append((bottom, bottom + 0.0035, bottom - 0.0001, bottom + 0.0033))
    price = bottom + 0.0033

    base_lo_body = price - 0.0002
    rows.append((price, price + 0.0004, price - 0.0005, price - 0.0001))
    rows.append((price - 0.0001, price + 0.0006, price - 0.0004, base_lo_body))
    base_high = price + 0.0006

    p = base_lo_body
    for _ in range(4):
        rows.append((p, p + 0.0001, p - 0.0036, p - 0.0034))
        p -= 0.0034
    for _ in range(12):
        rows.append((p, p + 0.0004, p - 0.0004, p - 0.0002))
        p -= 0.0002

    bars = make_bars(rows)
    cfg = Config()
    zones = find_zones(bars, cfg.zone, Structure(bars, 2))
    supply = [z for z in zones if z.kind == SUPPLY]

    check("RBD detected", len(supply) >= 1, f"found {len(zones)} zones")
    if not supply:
        return
    z = supply[0]
    check("pattern is RBD", z.pattern == "RBD", z.pattern)
    check(
        "proximal uses body bottom",
        abs(z.proximal - base_lo_body) < 1e-9,
        f"{z.proximal:.5f} vs {base_lo_body:.5f}",
    )
    check(
        "distal uses base wick high",
        abs(z.distal - base_high) < 1e-9,
        f"{z.distal:.5f} vs {base_high:.5f}",
    )
    check("proximal below distal for supply", z.proximal < z.distal)


def test_zone_lifecycle():
    """A zone counts a test on a touch and dies on a close through the distal."""
    from sd_bot.zones import Zone

    z = Zone(
        symbol="TEST", timeframe="H1", kind=DEMAND, pattern="DBR",
        base_start=1, base_end=2, created=5, created_time=0,
        proximal=1.1000, distal=1.0980, height=0.0020,
        atr_at_base=0.0010, departure_ratio=4.0, caused_bos=True,
        has_imbalance=True, curve=0.2,
    )
    z.update(1.1050, 1.1010, 1.1040)
    check("no test when price stays above the zone", z.tests == 0, f"tests={z.tests}")

    z.update(1.1020, 1.0995, 1.1015)
    check("touch counts as a test", z.tests == 1, f"tests={z.tests}")

    cfg = Config()
    check("tested zone is no longer live", not z.is_live(cfg.zone, 10))

    z.update(1.1000, 1.0970, 1.0975)
    check("close through distal invalidates", z.invalidated)


# -- no lookahead -----------------------------------------------------------


def test_align_never_leaks():
    """The HTF bar mapped to an LTF bar must already have closed."""
    m15 = make_bars(
        [(1.0 + i * 1e-5,) * 4 for i in range(400)], tf="M15"
    )
    m15.time[:] = np.arange(EPOCH, EPOCH + 400 * 900, 900, dtype=np.int64)[:400]
    h4 = resample(m15, "H4")
    idx = align(h4, m15)

    ok = True
    detail = ""
    for i, j in enumerate(idx):
        if j < 0:
            continue
        htf_close = int(h4.time[j]) + 4 * 3600
        ltf_close = int(m15.time[i]) + 900
        if htf_close > ltf_close:
            ok = False
            detail = f"bar {i} sees H4 bar {j} that closes {htf_close - ltf_close}s later"
            break
    check("align exposes only closed HTF bars", ok, detail)
    check("first bars have no HTF context", idx[0] == -1, f"idx[0]={idx[0]}")


def test_bad_data_is_rejected():
    """Corrupt bars must raise, not quietly produce fictional backtests.

    Regression guard: a Dukascopy feed returning datetime64[ms] was read as
    nanoseconds, collapsing a whole year of timestamps onto one instant. OHLC
    still looked valid and the backtest still ran, which is the dangerous case.
    """
    def bars_with(**overrides):
        n = 50
        base = dict(
            symbol="TEST", timeframe="H1",
            time=np.arange(EPOCH, EPOCH + n * 3600, 3600, dtype=np.int64),
            open=np.full(n, 1.10), high=np.full(n, 1.11),
            low=np.full(n, 1.09), close=np.full(n, 1.105),
            volume=np.full(n, 1.0), spread=np.full(n, 5.0),
        )
        base.update(overrides)
        return Bars(**base)

    def raises(label, **overrides):
        try:
            bars_with(**overrides)
            check(label, False, "no error raised")
        except ValueError:
            check(label, True)

    check("clean bars are accepted", bars_with() is not None)
    raises("millisecond timestamps rejected",
           time=np.arange(EPOCH, EPOCH + 50 * 3600, 3600, dtype=np.int64) * 1000)
    raises("truncated timestamps rejected",
           time=np.full(50, 1641, dtype=np.int64))
    raises("non-increasing timestamps rejected",
           time=np.arange(EPOCH, EPOCH - 50 * 3600, -3600, dtype=np.int64))
    raises("high below low rejected", high=np.full(50, 1.05))
    raises("close outside the bar rejected", close=np.full(50, 1.20))
    raises("non-positive prices rejected",
           open=np.zeros(50), low=np.zeros(50), close=np.zeros(50))


def test_swings_are_confirmed_late():
    bars = make_bars([
        (1.0, 1.0 + (0.001 if i == 10 else 0.0), 1.0 - 0.0005, 1.0)
        for i in range(30)
    ])
    swings = find_swings(bars, lookback=2)
    highs = [s for s in swings if s.is_high and s.index == 10]
    check("swing high found", len(highs) == 1, f"{len(highs)} found")
    if highs:
        check(
            "confirmation lags by the lookback",
            highs[0].confirmed == 12,
            f"confirmed={highs[0].confirmed}",
        )


# -- sizing and risk --------------------------------------------------------


def test_position_sizing():
    spec = SymbolSpec("EURUSD", digits=5, point=1e-5, tick_size=1e-5,
                      tick_value=1.0, volume_min=0.01, volume_step=0.01)
    rm = RiskManager(Config().risk, starting_equity=10_000)

    # 0.5% of 10,000 = $50. A 50 pip (500 point) stop costs $5 per 0.1 lot.
    volume, risk_amount = rm.volume_for(0.0050, spec)
    check("lot size matches the risk budget", abs(volume - 0.10) < 1e-9, f"{volume}")
    check("risk amount is at or under budget", risk_amount <= 50.0 + 1e-9,
          f"{risk_amount}")

    # A stop so wide the correct size rounds below the minimum lot: refuse.
    tiny = RiskManager(Config().risk, starting_equity=100)
    volume, _ = tiny.volume_for(0.0500, spec)
    check("refuses to trade rather than over-risk", volume == 0.0, f"{volume}")


def test_losing_streak_cuts_size():
    from sd_bot.trades import ClosedTrade

    rm = RiskManager(Config().risk, 10_000)
    base = rm.risk_pct()
    for _ in range(3):
        rm.on_close(ClosedTrade("EURUSD", LONG, 0, 0, 1.0, 0.99, 0.1,
                                -50, -1.0, "stop", 70, ""))
    check("size halves after three losses", abs(rm.risk_pct() - base * 0.5) < 1e-9,
          f"{rm.risk_pct()} vs {base}")

    rm.on_close(ClosedTrade("EURUSD", LONG, 0, 0, 1.0, 1.02, 0.1,
                            100, 2.0, "target", 70, ""))
    check("size restored after a win", abs(rm.risk_pct() - base) < 1e-9)


def test_currency_exposure():
    check("EURUSD splits", split_currencies("EURUSD") == ("EUR", "USD"))
    check("XAUUSD splits", split_currencies("XAUUSD") == ("XAU", "USD"))
    check("US30 does not split", split_currencies("US30") is None)

    def pos(symbol, direction):
        plan = TradePlan(symbol, direction, 1.0, 0.99, 1.02, 1.05, 0.01,
                         2, 5, 70, 0, "", 0, 0)
        return Position(symbol, direction, 1.0, 0, 0, 0.1, 0.99, 1.02, 1.05,
                        0.99, 50.0, plan)

    longs = [pos("EURUSD", LONG), pos("GBPUSD", LONG)]
    exposure = currency_exposure(longs)
    check("two long USD-quoted pairs stack short USD",
          exposure["USD"] == -2.0, f"{exposure}")

    rm = RiskManager(Config().risk, 10_000)
    veto = rm.veto("AUDUSD", LONG, longs)
    check("third correlated trade is vetoed", veto is not None, f"veto={veto}")

    veto = rm.veto("EURGBP", LONG, longs)
    check("uncorrelated trade is allowed", veto is None, f"veto={veto}")


def test_risk_gates():
    cfg = Config()
    rm = RiskManager(cfg.risk, 10_000)
    rm.on_time(int(datetime(2024, 1, 8, 9, tzinfo=timezone.utc).timestamp()), 10_000)
    check("clean slate allows a trade", rm.veto("EURUSD", LONG, []) is None)

    # Lose 3.5% intraday.
    rm.on_time(int(datetime(2024, 1, 8, 15, tzinfo=timezone.utc).timestamp()), 9_650)
    check("daily loss limit halts trading",
          rm.veto("EURUSD", LONG, []) is not None, rm.status())

    # New day resets the budget.
    rm.on_time(int(datetime(2024, 1, 9, 9, tzinfo=timezone.utc).timestamp()), 9_650)
    check("new day resets the daily budget",
          rm.veto("EURUSD", LONG, []) is None, rm.status())


# -- plan construction ------------------------------------------------------


def test_profit_margin_rule():
    """A demand zone sitting right under supply must be refused."""
    from sd_bot.zones import Zone

    cfg = Config()
    demand = Zone(
        symbol="TEST", timeframe="H4", kind=DEMAND, pattern="DBR",
        base_start=1, base_end=2, created=5, created_time=0,
        proximal=1.1000, distal=1.0980, height=0.0020, atr_at_base=0.0010,
        departure_ratio=6.0, caused_bos=True, has_imbalance=True, curve=0.1,
        htf_trend=1,
    )
    demand.score = 100.0

    # Stop sits ~0.0023 below entry, so 3R needs ~0.0069 of headroom.
    blocker = Zone(
        symbol="TEST", timeframe="H4", kind=SUPPLY, pattern="RBD",
        base_start=1, base_end=2, created=5, created_time=0,
        proximal=1.1030, distal=1.1050, height=0.0020, atr_at_base=0.0010,
        departure_ratio=6.0, caused_bos=True, has_imbalance=True, curve=0.9,
    )
    plan, why = build_plan(demand, cfg, 10, 0.0010, [demand, blocker])
    check("refuses a zone capped by nearby supply", plan is None, f"why={why}")
    check("explains the refusal", "opposing zone" in why, why)

    far = Zone(
        symbol="TEST", timeframe="H4", kind=SUPPLY, pattern="RBD",
        base_start=1, base_end=2, created=5, created_time=0,
        proximal=1.1200, distal=1.1220, height=0.0020, atr_at_base=0.0010,
        departure_ratio=6.0, caused_bos=True, has_imbalance=True, curve=0.9,
    )
    plan, why = build_plan(demand, cfg, 10, 0.0010, [demand, far])
    check("accepts the same zone with room to run", plan is not None, why)
    if plan:
        check("stop sits below the distal line", plan.stop < demand.distal,
              f"stop={plan.stop:.5f} distal={demand.distal:.5f}")
        check("entry is the proximal line", abs(plan.entry - demand.proximal) < 1e-9)
        check("reward clears the minimum", plan.rr2 >= cfg.signal.min_risk_reward,
              f"{plan.rr2:.2f}R")
        check("tp1 is nearer than tp2", plan.tp1 < plan.tp2)


def test_low_score_refused():
    from sd_bot.zones import Zone

    cfg = Config()
    z = Zone(
        symbol="TEST", timeframe="H4", kind=DEMAND, pattern="DBR",
        base_start=1, base_end=6, created=8, created_time=0,
        proximal=1.1000, distal=1.0980, height=0.0020, atr_at_base=0.0010,
        departure_ratio=3.0, caused_bos=False, has_imbalance=False, curve=0.9,
        htf_trend=-1,
    )
    z.score = 20.0
    plan, why = build_plan(z, cfg, 10, 0.0010, [z])
    check("low-scoring zone refused", plan is None, why)
    check("refusal names the score", "score" in why, why)


# -- scoring ----------------------------------------------------------------


def test_scoring_shape():
    check("untested zone scores full freshness", freshness_value(0) == 1.0)
    check("tested zone loses most freshness", freshness_value(1) < 0.5)
    check("twice-tested zone scores zero", freshness_value(2) == 0.0)
    check("weak departure scores low", departure_value(2.5) < 0.2)
    check("strong departure maxes out", departure_value(7.0) == 1.0)
    check("demand in deep discount scores full",
          curve_value(DEMAND, 0.05, 0.1) > 0.9)
    check("demand in premium scores zero",
          curve_value(DEMAND, 0.95, 0.1) == 0.0)
    check("supply in premium scores full",
          curve_value(SUPPLY, 0.95, 0.1) > 0.9)


def test_weights_sum_to_100():
    z = Config().zone
    total = (z.w_freshness + z.w_departure + z.w_base_tightness + z.w_bos
             + z.w_htf_trend + z.w_curve + z.w_htf_confluence + z.w_imbalance)
    check("scoring weights sum to 100", abs(total - 100.0) < 1e-9, f"{total}")


def test_engine_has_no_directional_bias():
    """On a driftless random walk the engine must be symmetric and unbiased.

    Two things are being checked. First, that longs and shorts come out roughly
    balanced -- a lopsided count means the zone logic favours one side. Second,
    that maximum favourable excursion reaches 1R on about half of trades, which
    is what a martingale guarantees. A number far from 50% means entries are
    systematically mistimed or fills are being taken at impossible prices.
    """
    from sd_bot.backtest import Backtester
    from sd_bot.data import synthetic

    trades = []
    for seed in range(1, 9):
        cfg = Config()
        cfg.execution.symbols = ["EURUSD"]
        cfg.backtest.slippage_points = 0.0
        cfg.backtest.commission_per_lot = 0.0
        cfg.validate()

        m15 = synthetic("EURUSD", "M15", n=35_000, seed=seed, trendiness=0.0)
        m15.spread[:] = 0.0
        trades.extend(
            Backtester(cfg, {}).run(
                {"EURUSD": {"M15": m15, "H4": resample(m15, "H4"),
                            "D1": resample(m15, "D1")}}
            ).trades
        )

    n = len(trades)
    check("driftless walk produces trades to measure", n >= 10, f"n={n}")
    if n < 10:
        return

    longs = sum(1 for t in trades if t.direction == LONG)
    share = longs / n
    check("longs and shorts stay balanced", 0.2 <= share <= 0.8,
          f"{longs}/{n} long")

    reached_1r = sum(1 for t in trades if t.mfe_r >= 1.0) / n
    check("MFE reaches 1R on roughly half of trades",
          0.30 <= reached_1r <= 0.70, f"{reached_1r:.1%} (expected ~50%)")

    check("no trade reports a positive MAE",
          all(t.mae_r <= 0.0 for t in trades))
    check("no trade reports a negative MFE",
          all(t.mfe_r >= 0.0 for t in trades))


def test_analysis_cache_is_not_poisoned():
    """Reusing cached analysis must not change results.

    Zones carry mutable per-run state. If the cache hands out the same objects
    the previous run consumed, every run after the first sees dead zones and
    silently takes no trades -- which looks like a strategy finding, not a bug.
    """
    from sd_bot.backtest import Backtester
    from sd_bot.data import synthetic

    cfg = Config()
    cfg.execution.symbols = ["EURUSD"]
    cfg.validate()
    m15 = synthetic("EURUSD", "M15", n=25_000, seed=11, trendiness=0.6)
    data = {"EURUSD": {"M15": m15, "H4": resample(m15, "H4"),
                       "D1": resample(m15, "D1")}}

    uncached = Backtester(cfg, {}).run(data)
    cache: dict = {}
    first = Backtester(cfg, {}, analysis_cache=cache).run(data)
    second = Backtester(cfg, {}, analysis_cache=cache).run(data)
    third = Backtester(cfg, {}, analysis_cache=cache).run(data)

    counts = [len(r.trades) for r in (uncached, first, second, third)]
    check("identical configs give identical trade counts",
          len(set(counts)) == 1, f"uncached/1st/2nd/3rd = {counts}")
    check("cached runs still trade", counts[-1] > 0, f"{counts}")
    check("cached run matches uncached P/L",
          abs(third.ending_balance - uncached.ending_balance) < 1e-6,
          f"{third.ending_balance:.2f} vs {uncached.ending_balance:.2f}")


def test_costs_are_actually_charged():
    """A stopped-out trade must lose slightly more than 1R, never exactly 1R."""
    from sd_bot.backtest import Backtester
    from sd_bot.data import synthetic

    cfg = Config()
    cfg.execution.symbols = ["EURUSD"]
    cfg.validate()
    m15 = synthetic("EURUSD", "M15", n=20_000, seed=3, trendiness=0.5)
    result = Backtester(cfg, {}).run(
        {"EURUSD": {"M15": m15, "H4": resample(m15, "H4"), "D1": resample(m15, "D1")}}
    )
    stopped = [t for t in result.trades if t.reason == "stop"]
    check("some trades were stopped out", len(stopped) > 0, f"{len(stopped)}")
    if stopped:
        check("stop-outs cost more than 1R once costs are charged",
              all(t.r_multiple < -1.0 for t in stopped),
              f"best stop-out was {max(t.r_multiple for t in stopped):.3f}R")


def test_config_rejects_bad_timeframes():
    cfg = Config()
    cfg.execution.zone_timeframe = "M5"   # lower than the entry timeframe
    try:
        cfg.validate()
        check("config rejects inverted timeframes", False, "no error raised")
    except ValueError:
        check("config rejects inverted timeframes", True)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"all {len(tests)} test groups passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
