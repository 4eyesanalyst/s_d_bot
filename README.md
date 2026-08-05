# Supply & Demand Bot (MetaTrader 5)

An automated supply-and-demand trader for FX, metals and indices. It marks
institutional order blocks on a higher timeframe, grades them, and only trades
the small fraction that clear every filter — with position sizing and loss
limits that assume the strategy will go wrong regularly, because it will.

Backtest first. Then demo. Then, only if the numbers hold up, live.

---

## Quick start

```bash
pip install MetaTrader5 pandas numpy pyyaml

python tests/test_engine.py        # verify the engine (no MT5 needed)
python -m sd_bot selftest          # end-to-end run on generated data
python -m sd_bot account           # confirm MT5 connection + symbol names
python -m sd_bot backtest          # the run that actually matters
python -m sd_bot scan              # what the bot sees right now
python -m sd_bot live --dry-run    # full loop, logs decisions, sends nothing
python -m sd_bot live              # places orders
```

`backtest`, `scan`, `account` and `live` need the MT5 terminal running and
logged in. `selftest` and the test suite do not.

If your terminal is not in the default location, or you want the bot to log in
itself:

```bash
set MT5_LOGIN=12345678
set MT5_PASSWORD=yourpassword
set MT5_SERVER=YourBroker-Demo
set MT5_PATH=C:\Program Files\YourBroker MT5\terminal64.exe
```

**Symbol names must match your broker exactly.** Many brokers use suffixes —
`EURUSD.m`, `EURUSDpro`, `XAUUSD.raw`. Run `python -m sd_bot account` to check
what resolves before anything else.

---

## The rules it trades

### What makes a zone

Institutions cannot fill size at one price. They accumulate in a small
consolidation — the **base** — and when the rest of the order lifts the book,
price leaves in a straight line. The unfilled remainder sits at that base. Price
returning there is the only place this bot is interested in.

Four shapes, one idea:

| Pattern | Meaning | Gives |
|---|---|---|
| **DBR** | drop-base-rally | demand (reversal) |
| **RBR** | rally-base-rally | demand (continuation) |
| **RBD** | rally-base-drop | supply (reversal) |
| **DBD** | drop-base-drop | supply (continuation) |

A base candle has a small body relative to current ATR. A leg candle has a large
body *and* little wick — conviction, not rejection. Both thresholds are
volatility-relative, so nothing needs retuning per symbol.

**Boundaries** use the conservative convention:

- **Demand** — proximal (entry) = highest body top of the base; distal (stop
  side) = lowest wick of the base.
- **Supply** — proximal = lowest body bottom; distal = highest wick.

Bodies for entry, wicks for the stop. Costs a little fill rate, buys a lot of
safety.

### Hard filters — a zone is discarded outright if

- price did not travel at least **3× the zone's own height** away from it
  (`min_departure_ratio`) — this is the proxy for a real imbalance;
- it has already been **touched once** (`max_tests: 1`) — freshness is the
  single largest edge, and a retested zone has given its orders away;
- it is **wider than 2× ATR** (bad reward ratio) or **thinner than 0.08× ATR**
  (the stop would sit inside the spread);
- the base ran longer than **6 candles** — the longer price sits there, the more
  of the order has already been absorbed;
- it is older than `max_zone_age_bars`.

### Scoring — 0 to 100, trade only ≥ 65

| Factor | Weight | Rewards |
|---|---:|---|
| Freshness | 22 | untouched zones |
| Departure strength | 18 | 6× zone height scores full marks, 2× scores nothing |
| Break of structure | 14 | the departure broke a prior swing — institutional intent |
| HTF trend alignment | 14 | demand in an uptrend, supply in a downtrend |
| Curve (premium/discount) | 10 | demand in the lower half of the range, supply in the upper |
| Base tightness | 8 | one base candle is ideal |
| HTF confluence | 9 | zone nested inside a daily zone — the A+ setup |
| Imbalance (FVG) | 5 | unfilled fair-value gap in the departure leg |

Grades: **A+** ≥85, **A** ≥75, **B** ≥65, **C** ≥50.

### Entry, stop, targets

- **Entry** — a resting limit order at the proximal line. Set and forget.
  Switch `entry_mode` to `confirmation` to instead require price to trade into
  the zone and *close* back out through the proximal line — lower reward ratio,
  materially higher hit rate, worth testing on counter-trend zones.
- **Stop** — beyond the distal line by `max(15% of zone height, 0.2 × ATR,
  2 × spread)`. Never a fixed pip count. The market decides where the trade is
  wrong.
- **Profit margin** — the rule that removes the most losers. A zone is only
  traded if there is at least **3R of clear air** between the entry and the next
  opposing zone. Buying a beautiful demand zone twenty pips under untested
  supply is the classic way this strategy bleeds.
- **Targets** — TP1 at 2R takes half off and moves the stop to breakeven. TP2
  parks just in front of the next opposing zone. After TP1 the stop trails
  behind swing structure.

### Risk — the part that decides survival

| Control | Default |
|---|---|
| Risk per trade | 0.5% |
| Max open trades | 3 |
| Max trades per symbol | 1 |
| Total open risk | 2% |
| Daily loss limit | 3% → stop for the day |
| Weekly loss limit | 6% → stop for the week |
| Max drawdown | 15% → stop taking risk |
| After 3 losses in a row | size halves until a winner |
| Currency exposure cap | 2 |

That last one matters more than it looks. Long EURUSD + long GBPUSD + long
AUDUSD is not three trades — it is one short-USD position at triple size. The
bot nets exposure per currency and refuses the third.

If the correctly sized position rounds below the broker's minimum lot, the bot
**skips the trade** rather than taking the minimum and silently over-risking.

Sessions default to London + New York (07:00–20:00 UTC), Monday to Friday, with
everything flattened before the weekend gap.

---

## Is the edge real? Four tests

`python prove.py --symbols XAUUSD,EURUSD,USDJPY`

A profitable backtest proves nothing by itself. These four questions decide it.

### 1. Null hypothesis -- do the zones actually matter?

200 simulations trading the same instruments at **random times**, matched on
direction mix, stop sizes, targets, session filter and costs. The only
difference is that entries ignore supply and demand.

```
random entries    -0.062R   (5th -0.149, 95th +0.022)
strategy          +0.323R
edge over random  +0.385R
p-value            0.0050   (0 of 200 random runs matched or beat it)
```

Random entry with identical trend exposure and identical exit management earns
**nothing**. This is the test that matters, and it also settles the long-bias
objection: the random benchmark carried the same long tilt through the same gold
bull market and still returned zero. The edge is *where the zones put the
entry*, not the trend.

### 2. Confidence -- is it noise?

Bootstrap of 591 trades, 10,000 resamples: 5th percentile **+0.197R**, median
+0.323R, 95th +0.449R. The lower bound sits well above zero.

### 3. Sensitivity -- does it survive being nudged?

**11 of 11** perturbations stayed profitable (score thresholds, reward ratios,
departure ratios, TP1 levels, swing lookback, test counts). Nothing balances on
one setting.

### 4. Costs -- how much friction kills it?

Survives to roughly **+40 points** of extra slippage on top of recorded spread
and $7/lot commission. Comfortable margin over a normal retail broker.

```
null hypothesis   p=0.0050   PASS
bootstrap 5th     +0.197R    PASS
sensitivity       11/11      PASS
cost tolerance    +40 pts    PASS
```

### Portfolio result (XAUUSD + EURUSD + USDJPY)

```
trades         591        signals/week   2.5
win rate      46.9%       profit factor  1.65
expectancy   +0.323R      return       +140.3%
max drawdown  8.57%
```

> **Still open: the live scanner reproduces only ~55% of these entries.** The
> strategy is proven; the bot is not yet proven to trade it. See
> `tests/test_parity.py`. Do not treat live results as equivalent until that
> gap closes.

---

## Measured results: XAUUSD, 2022-2026

Dukascopy bank-feed history, real spread + 2pt slippage + $7/lot commission,
0.5% risk per trade, $10,000 start, London + New York sessions only.

```
period          trds   t/wk   win%     pf    expR    ret%    dd%  wpips
2022              33   0.64   75.8   3.27   +0.62    +9.9    1.5     78
2023              14   0.27   57.1   2.37   +0.71    +4.2    1.9    105
2024              56   1.07   57.1   1.63   +0.26    +6.6    3.6     60
2025              95   1.83   52.6   1.96   +0.43   +18.5    2.6    151
2026 (to Aug)     55   1.83   34.5   1.04   +0.06    +0.5    3.3    277
ALL              228   0.96   53.9   1.91   +0.41   +47.4    3.4    138
```

> **These numbers were revised down in a later pass.** An earlier version
> reported +42.9% at PF 1.74. Building a backtest/live parity harness exposed
> lookahead in zone de-duplication: a later zone could retroactively replace an
> earlier overlapping one, so the backtest was making choices using information
> that did not exist yet. With de-duplication made causal the result is +36.0%
> at PF 1.53. The strategy still works; it works less well than first reported.
> Assume any number here is an upper bound.

Every full year profitable; 2026 year-to-date is flat. Expect **about one signal
a week**, but frequency tracks volatility: 0.27/week in quiet 2023, 1.8/week
through the 2025-26 gold run.

### Session window

Restricting to London + New York costs setups and improves everything else:

| Window (UTC) | Trades/wk (OOS) | PF (OOS) | Max DD (OOS) |
|---|---:|---:|---:|
| 24 hours | 2.49 | 1.67 | 6.6% |
| **London+NY 07-22** | **1.76** | **1.74** | **4.6%** |
| London only 07-16 | 1.35 | 1.86 | 4.1% |
| New York only 12-21 | 1.30 | 1.88 | 3.5% |
| Overlap only 12-16 | 0.91 | 2.02 | 3.3% |

Narrower is consistently better per trade and worse in total. 07:00-21:59 UTC
covers both sessions year-round through the DST shifts. 58% of all entries land
in the 12:00-15:00 overlap, where gold liquidity peaks.

### Three caveats that matter more than the headline

**1. The edge is long-biased.**

| Side | Trades | Total R | Expectancy |
|---|---:|---:|---:|
| LONG | 119 | +77.8R | +0.65R |
| SHORT | 85 | +16.0R | +0.19R |

Gold went 1,615 to 5,597 across this window. The session filter lifted the short
side from +0.04R to +0.19R, but longs still carry it. What this demonstrates is
largely a trend-following bet expressed through demand zones, not a symmetric
supply/demand edge. **This is the biggest reason not to trust the 43% return.**

**2. Frequency is capped by the instrument, not the settings.** Each zone can
only be traded once while fresh, so the zone formation rate is a hard ceiling:

| Zone timeframe | Zones found | Zones/day | Max trades/day |
|---|---:|---:|---:|
| D1 | 142 | 0.08 | 0.08 |
| H4 | 464 | 0.28 | 0.28 |
| H1 | 1,109 | 0.66 | 0.66 |

Dismantling every quality rule -- no grading, all hours, RR 1.5, three tests
allowed -- still only reached 0.28/day on H4. Two trades per day is not
reachable on one instrument; it needs a basket of roughly twenty.

**3. The early years are small samples.** 2022 and 2023 contribute 30 trades
between them at implausibly high expectancy. Most of the profit and nearly all
the statistical weight sits in 2025-2026.

### Move size

`signal.min_stop_pips` is the lever. Move size is roughly stop x reward ratio, so
the 40-pip floor is what puts the average winner near 100 pips:

| Stop floor | Trades/day | Avg winner |
|---|---:|---:|
| none | 0.44 | 56 pips |
| 20 pips | 0.28 | 69 pips |
| 30 pips | 0.16 | 74 pips |
| **40 pips** | **0.09** | **100 pips** |

Bigger moves cost frequency, one for one. There is no setting that gives both.

---

---

> **Live parity: ~56%.** `tests/test_parity.py` replays history through the real
> scanner and compares it against the backtester. Five implementation bugs were
> found and fixed this way (limit-fill semantics, fill state, pre-bar ordering,
> blocker universe, spent-zone retirement), taking parity from 0% to ~56%.
> Diagnosis of what remains: of 27 backtest entries, 14 reproduce exactly, 8 are
> zones the scanner's rolling window does not detect, 3 are a test artifact
> (insufficient warm-up early in the replay), and 2 are filter disagreements.
> **The strategy is proven; the live bot reproduces about half of it.** Treat
> alerts as candidates to check, not orders to take blindly.

## Real-time signal bot

Watches the market live and pushes an alert the moment a setup becomes
actionable, then keeps following it and tells you when it fills, banks TP1, or
stops out. It signals; it does not place orders.

```bash
python -m sd_bot alerts --discover    # find your Telegram chat id
python -m sd_bot alerts --test        # send a test alert
python -m sd_bot signals              # run it
```

### Telegram setup, once

1. Open Telegram, message **@BotFather**, send `/newbot`, follow the prompts.
2. Copy the token it gives you and set it:
   `set TELEGRAM_TOKEN=123456:AAF...`
3. Send your new bot any message (it cannot message you first).
4. `python -m sd_bot alerts --discover` prints your chat id.
   `set TELEGRAM_CHAT_ID=987654321`
5. `python -m sd_bot alerts --test` to confirm.

Console and a JSONL file under `signals/` are always on, with or without
Telegram, so an alert is never lost to one channel failing.

### What arrives

Three kinds:

| Alert | Meaning |
|---|---|
| **APPROACHING** | Price is nearing a qualifying zone. Get ready, do nothing. |
| **TRIGGERED** | Price reached the entry. Full levels and reasoning. |
| **UPDATE** | A live signal filled, hit TP1/TP2, stopped out, or was invalidated before entry. |

A TRIGGERED alert looks like this:

```
[TRIGGERED] BUY XAUUSD @ 4183.42

ENTRY   4183.42
STOP    4143.42   (40 pips risk)
TP1     4263.42   (+80 pips, 2.0R) close 50%
TP2     4343.42   (+160 pips, 4.0R)

SIZE    0.12 lots
        risks 50 (0.5% of 10,000). Scale to your own balance.

WHY
  - DBR zone, grade A (78/100)
  - departure 5.2x zone height
  - H4 trend up
  - discount (31% of range)
  - departure broke structure
  - zone untested

Manage: bank 50% at TP1, let the rest run to TP2.
```

### Data

No MetaTrader terminal required. The feed auto-selects:

- **MT5** when a terminal is running (your broker's own prices and live spread)
- **Dukascopy** otherwise, a bank feed roughly 30-60s behind

Either way, only **closed** bars are used. A forming bar's high, low and close
can all still change, and acting on one is the fastest way to make a live bot
behave nothing like its backtest.

The first poll seeds history and takes 2-3 minutes per instrument. Every poll
after that is instant, because bars are only re-fetched when a new one is due.

### Watchlist: only what passed

The bot ships watching **XAUUSD, EURUSD, USDJPY**. Those are the three of eight
instruments that were actually profitable 2022-2026:

| Symbol | Signals/wk | Win% | PF | ExpR | Return | Max DD | Avg winner |
|---|---:|---:|---:|---:|---:|---:|---:|
| XAUUSD | 0.85 | 47.1% | 1.74 | +0.46 | +42.9% | 4.3% | 178 pips |
| EURUSD | 0.82 | 45.1% | 1.53 | +0.38 | +37.9% | 6.3% | 22 pips |
| USDJPY | 0.64 | 41.8% | 1.29 | +0.21 | +13.5% | 6.7% | 36 pips |
| USDCHF | 0.75 | 40.0% | 0.84 | -0.11 | -9.4% | 15.3% | 17 pips |
| GBPUSD | 0.50 | 35.0% | 0.70 | -0.15 | -11.8% | 15.1% | 27 pips |
| USDCAD | 0.68 | 34.6% | 0.78 | -0.18 | -11.1% | 21 pips |
| AUDUSD | 0.32 | 27.6% | 0.44 | -0.41 | -15.2% | 15.3% | 18 pips |
| NZDUSD | 0.23 | 23.6% | 0.21 | -0.73 | -15.3% | 15.3% | 12 pips |

The bottom five lose money and are deliberately **not** watched. Adding them
back would roughly double the signal count and destroy the edge. If you want
more instruments, backtest them first with `python validate.py --symbol X`.

Note the pattern: the three that work are the deepest, most liquid instruments.
Supply and demand needs real institutional order flow to leave a footprint.

**Move size is instrument-specific.** Gold winners average ~178 pips; EURUSD
winners average ~22. A 100-pip EURUSD move is a multi-day swing, not an H1 zone
trade. `min_stop_pips` is set per symbol in the `symbols:` block of config.yaml,
scaled to each instrument's own volatility (0.45 x H1 ATR).

### The three together, as one account

Run as a portfolio with shared equity and the correlation caps active:

```
trades         554        signals/week   2.33   (238 weeks)
win rate      45.1%       profit factor  1.53
expectancy   +0.354R      total         196.2R
net profit  +13,217       return       +132.2%
max drawdown  5.83%       final balance 23,217
```

Diversification also repairs the worst weakness of the gold-only result. On gold
alone the edge was almost entirely long (+0.69R long vs +0.04R short) over a
historic bull run. Across the three:

| Side | Trades | Total R | Expectancy |
|---|---:|---:|---:|
| LONG | 299 | +123.7R | +0.41R |
| SHORT | 255 | +72.5R | +0.28R |

Both directions now pay, on comparable trade counts. That is a materially more
credible result than gold on its own, because it no longer depends on one
instrument trending one way.

### Running it 24/7

The bot is a Python process, so it runs while its host is on. It needs **no
MetaTrader terminal, no Windows and no broker account** -- data comes from
Dukascopy over HTTPS and alerts go out via the Telegram API -- so it runs on the
cheapest Linux box available, in well under 300 MB of RAM.

```bash
docker compose up -d --build     # see DEPLOY.md
```

`run_signals.py` supervises the scanner, retrying with exponential backoff after
a crash and messaging you when it recovers. `signals/state.json` persists
in-flight signals so a restart resumes tracking instead of re-alerting.

Full hosting options, costs and failure modes: **[DEPLOY.md](DEPLOY.md)**.

Leave `heartbeat_hours` on. This strategy averages about one signal per
instrument per week, so a quiet market and a dead process look identical from
the outside -- the heartbeat is what tells them apart.

### Keeping it running

This is a Python process, not an EA inside MT5. If the machine sleeps or the
terminal closes, no signals are sent. For always-on alerts, run it on a cheap
VPS. State lives in `signals/state.json`, so a restart resumes tracking any
signal already in flight rather than re-alerting it.

---

## Reading a backtest

```
Expectancy    +0.184R     Total          31.2R
Profit factor   1.42      Max drawdown  8.31%
```

Judge it on **expectancy in R**, not currency — currency flatters a system that
happened to size up into its winners. Then look at the two sections the report
prints underneath:

- **Setups declined** — zones price reached but the plan refused, with reasons.
  If `only <n>R` dominates, the profit-margin rule is doing its job.
- **Risk vetoes** — trades the risk manager blocked, and why.

A supply/demand system declines far more than it takes. Long quiet stretches are
the strategy working, not failing.

**What "good" looks like:** expectancy above +0.2R over 100+ trades, profit
factor above 1.3, max drawdown under 20%, and no single symbol carrying the
whole result. Fewer than ~50 trades tells you nothing — widen the date range or
add symbols before drawing any conclusion.

### Honest limits of this backtester

- Fills are modelled on **M15 bars, not ticks**. When a bar contains both the
  stop and a target, the stop is assumed (`pessimistic_fills`).
- Spread is the value MT5 recorded on each bar, plus `slippage_points` and
  `commission_per_lot`. **Check these against your own broker** — the defaults
  are generic and costs are what kill tight-stop systems.
- No swap/financing, no news-event gaps, no requote or partial-fill simulation.
- MT5 history is your broker's, and it is finite. Deep M15 history often is not
  available beyond a few years.

---

## Tuning

Every value in `config.yaml` is a filter. **Loosening one buys more trades, not
more profit.** The levers worth testing, in order:

1. `zone.min_score` (65) — raise to 75 to trade only A-grade zones.
2. `signal.min_risk_reward` (3.0) — the profit-margin rule.
3. `zone.min_departure_ratio` (3.0) — how violent the departure must be.
4. `signal.entry_mode` — `limit` vs `confirmation`.
5. `signal.breakeven_after_tp1` — reduces variance but costs expectancy when the
   edge is thin. Test it both ways on your own data.
6. `execution.zone_timeframe` — H4 zones with M15 entries is the default; D1/H1
   is slower and cleaner, H1/M5 is faster and noisier.

Change **one thing at a time** and re-run. If a change only helps on one symbol
or one year, it is curve fitting, not an improvement.

---

## Going live

1. `python tests/test_engine.py` — all 16 groups pass.
2. `python -m sd_bot backtest` over as much history as your broker gives you.
3. `python -m sd_bot live --dry-run` for a few days. Read the journal. Make sure
   the setups it logs are ones you would have taken by hand.
4. Demo account at full size for **at least a month**, ideally a quarter.
5. Live at 0.25% risk. Raise only after the live results match the demo.

The bot refuses to start on a live account without a typed confirmation, and
warns if algo trading is disabled in the terminal.

**Keep the MT5 terminal running.** This is a Python client driving the terminal,
not an EA compiled into it — if the terminal closes or the machine sleeps,
nothing manages your open positions. A VPS is the right home for this.

---

## Layout

| File | Purpose |
|---|---|
| `config.yaml` | every rule and threshold |
| `sd_bot/zones.py` | zone detection — the core |
| `sd_bot/scoring.py` | 0–100 quality grade |
| `sd_bot/structure.py` | swings, trend, premium/discount curve |
| `sd_bot/signals.py` | entry, stop, targets, profit-margin rule |
| `sd_bot/risk.py` | sizing, loss limits, currency exposure |
| `sd_bot/backtest.py` | event-driven portfolio simulator |
| `sd_bot/live.py` | MT5 execution loop |
| `sd_bot/stats.py` | performance report |
| `sd_bot/broker.py` | MT5 adapter |
| `tests/test_engine.py` | engine tests |

Outputs land in `results/` (backtest), `journal/` (live) and `data/` (bar cache).

---

## A word on expectations

This bot encodes a well-known discretionary method with defensible mechanics and
serious risk controls. That is not the same as a proven edge. Supply and demand
works because of *where* zones sit and *what* price did to get there — the
filters here approximate that judgement, they do not replace it.

Validate it on your own broker's data, with your own costs, before it trades
anything you would miss.
