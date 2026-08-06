# Project log — Supply & Demand trading bot

A complete record of what was built, what was proven, what was found broken, and
what is still open. Written so that in six months you can reconstruct every
decision without re-deriving it.

**Repository:** https://github.com/4eyesanalyst/s_d_bot
**Status:** live on GitHub Actions, alerting to Telegram
**Built:** August 2026

---

## 1. What this is

An automated supply-and-demand trader. It marks institutional order blocks on a
higher timeframe, grades them 0–100, and alerts you when price returns to the
small fraction that clear every filter.

It **signals**; it does not place orders. You get entry, stop and both targets on
your phone and execute yourself.

**Instruments:** XAUUSD, EURUSD, USDJPY
**Timeframes:** H4 bias → H1 zones → M15 entry
**Sessions:** 07:00–22:59 UTC, Mon–Fri (London open → New York close)
**Frequency:** ~2.5 signals/week
**Risk:** 0.5% per trade

---

## 2. How it got here

| Stage | What happened |
|---|---|
| Engine | Zone detection, scoring, risk, backtester, 18 unit tests |
| Data | No MT5 terminal available → switched to Dukascopy bank feed |
| First real backtest | Only 41 zones in 7,341 bars — detection was far too strict |
| Rewrite | Detection rebuilt around maximal consolidation runs → 465 zones |
| Instrument search | Tested 8 instruments; **5 lost money** and were dropped |
| Tuning | Session window, stop floors, TP1 level — each validated in/out of sample |
| Proof | Null-hypothesis, bootstrap, sensitivity and cost tests |
| Signal bot | Live feed, Telegram delivery, signal tracking |
| Parity | Replay harness comparing live scanner to backtester — six bugs found |
| Deployment | Oracle Cloud failed → GitHub Actions instead |

---

## 3. The strategy

### Zones

Institutions cannot fill size at one price. They accumulate in a small
consolidation (the **base**), and when the rest of the order lifts the book,
price leaves in a straight line (the **departure**). The unfilled remainder sits
at that base. Price returning there is the only place this bot trades.

Four shapes: **DBR** and **RBR** give demand; **RBD** and **DBD** give supply.

**Boundaries** — bodies for entry, wicks for the stop:
- Demand: proximal = highest body top of the base; distal = lowest wick
- Supply: proximal = lowest body bottom; distal = highest wick

### Hard filters

A zone is discarded outright if price did not travel **3× the zone's height**
away from it, if it has already been **touched once**, if it is wider than 2×
ATR or thinner than 0.08× ATR, or if the base ran longer than 6 candles.

### Scoring (trade only ≥ 55)

| Factor | Weight |
|---|---:|
| Freshness (untouched) | 22 |
| Departure strength | 18 |
| Break of structure | 14 |
| HTF trend alignment | 14 |
| Curve (premium/discount) | 10 |
| Base tightness | 8 |
| HTF confluence | 9 |
| Imbalance (FVG) | 5 |

### Entry, stop, targets

- **Entry** — resting limit order at the proximal line
- **Stop** — beyond the distal by `max(15% of zone height, 0.2×ATR, 2×spread)`,
  never a fixed pip count
- **Profit margin** — requires 3R of clear air to the next opposing zone. This
  rule declines more setups than any other, and it is the one that stops you
  buying good demand twenty pips under untested supply
- **TP1 at 1.25R** closes half; **TP2** parks in front of the opposing zone
- Breakeven **off** — measured as costing expectancy on this strategy

### Risk

0.5%/trade · max 3 open · 2% total · 3% daily stop · 6% weekly stop · 15% max
drawdown · size halves after 3 consecutive losses · currency exposure capped at 2

---

## 4. Results (2022-01 → 2026-08, real costs)

```
starting          10,000.00
ending            24,385.16
net profit        14,385.16   (+143.9%)
CAGR                   21.5%
max drawdown       1,577.44   (8.60%)
return/drawdown       16.73

trades                  601   (2.52/week)
win rate              46.9%
profit factor          1.65
expectancy           +0.320R
avg win/loss     +1.70R / -0.90R
Sharpe                 1.83
worst streak             12 losses
```

**Monthly (fixed 10k base):**
```
year     Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec   total
2022     183   242  1213    94   584  1711   389  -415   435   349  2761   -46    7499
2023     557  -150  -115   130  -420     1   -81  -254   204  -251  -342     .    -722
2024     111     .   967   521  -163   330  -310  -108   -60    26   418   -96    1637
2025     287   136   323   447   462   168  -107  -139  1417   402   796    41    4232
2026     169   777    65   548  -128   274    33     .     .     .     .     .    1738
```
53 months · 68% profitable · worst month −420 · worst run 3 losing months

| Instrument | Trades | Win% | PF | ExpR | | Direction | Trades | ExpR |
|---|---:|---:|---:|---:|---|---|---:|---:|
| XAUUSD | 232 | 53.9 | 1.86 | +0.387 | | LONG | 338 | +0.330R |
| EURUSD | 213 | 44.6 | 1.65 | +0.365 | | SHORT | 263 | +0.307R |
| USDJPY | 156 | 39.7 | 1.39 | +0.159 | | | | |

Full ledger: `results/ledger.csv` (601 trades).

### Instruments tested and rejected

Five of eight lost money and are deliberately **not** watched:

| Rejected | PF | Return |
|---|---:|---:|
| USDCHF | 0.84 | −9.4% |
| GBPUSD | 0.70 | −11.8% |
| USDCAD | 0.78 | −11.1% |
| AUDUSD | 0.44 | −15.2% |
| NZDUSD | 0.21 | −15.3% |

The three that work are the deepest, most liquid instruments. Supply and demand
needs real institutional flow to leave a footprint.

---

## 5. Proof that the edge is real

`python prove.py` — four tests.

**1. Null hypothesis (the one that matters).** 200 simulations entering at random
times, matched on direction mix, stop sizes, targets, sessions and costs. Only
the zone logic removed.

```
random entries    -0.062R   (5th -0.149, 95th +0.022)
strategy          +0.323R
edge over random  +0.385R
p-value            0.0050   (0 of 200 random runs matched it)
```

Random entry with identical trend exposure and identical exit management earns
nothing. This also settles the long-bias question: the random benchmark carried
the same long tilt through the same gold bull market and still returned zero.
**The edge is where the zones put the entry, not the trend.**

**2. Confidence.** Bootstrap of 591 trades, 10,000 resamples: 5th percentile
**+0.197R**, median +0.323R. Lower bound well above zero.

**3. Sensitivity.** 11 of 11 parameter perturbations stayed profitable.

**4. Costs.** Survives to **+40 points** of extra slippage.

---

## 6. Bugs found — the most valuable part of this log

Every one of these was found by a test, and every one would have quietly
corrupted results or cost signals forever.

### In the research pipeline

**1. Corrupt timestamps (silent, catastrophic).** Dukascopy returns
`datetime64[ms]`; the code divided by 1e9 assuming nanoseconds. Every timestamp
in 72 files collapsed to `1641`. OHLC still looked valid and backtests still
ran — producing pure fiction. *Fix: explicit second-resolution conversion, plus
`Bars.validate()` which now rejects implausible or non-monotonic timestamps on
construction.*

**2. Poisoned analysis cache.** The sweep cache handed out zone objects a
previous run had already consumed, so every variant after the first took zero
trades. This produced a convincing fake finding ("score ≥60 kills all trades")
that was nearly reported as real. *Fix: cache stores pristine deep copies.*

**3. Zone detection 10× too strict.** Requiring a single explosive candle on
both sides of the base killed 98% of candidates (4,240 → 74). Methodologically
wrong too: real departures build over two or three candles. *Fix: rebuilt around
maximal consolidation runs with a windowed departure test. 41 → 465 zones.*

**4. Lookahead in the backtester.** `_dedupe` let a later zone retroactively
replace an earlier overlapping one — choosing zones using information that did
not exist yet. *Fix: made causal. This reduced reported returns from +42.9% to
+36.0%, confirming the earlier numbers were inflated.*

**5. Per-symbol overrides ignored.** The backtester applied gold's 40-pip stop
floor to EURUSD, producing 0–1 trades and the false impression that "FX doesn't
work". *Fix: `Config.for_symbol()` resolved per instrument.*

### In the live scanner (found by the parity harness)

**6. Triggered on bar close, not bar range.** The backtest fills a resting limit
order on the bar's low; the scanner only checked the close — missing every
wick-fill.

**7. Signals stuck `pending` forever** when price never closed back through the
level, blocking all later setups on that symbol.

**8. Zones aged through the current bar**, so the first touch marked a zone
"tested" and retired it one moment before it could be signalled.

**9. Blocker universe mismatch** — the two engines disagreed about which zones
still obstructed price.

**10. Spent zones resurrected** on every zone-map rebuild.

**11. `find_zones` reserved `leg_lookahead` bars at the end of the series.**
Harmless for a backtest reading full history, fatal for a live scanner working
at the data edge: a zone whose departure had just completed stayed invisible for
another 12 hours — exactly the zones price was about to return to. **This single
fix took parity from 56% to 88% and changed backtest results by nothing at all**
(601 trades, +143.9% before and after).

### In monitoring

**12. Heartbeat documented but never implemented.** The in-process heartbeat was
disabled with a comment claiming the workflow drove it instead; nothing did. The
bot had no liveness signal — and for a strategy that fires 2–3 times a week,
silence and death are indistinguishable. *Fix: daily heartbeat on its own cron.*

---

## 7. Backtest ↔ live parity

`python tests/test_parity.py` replays history through the **real** scanner, bar
by bar, and compares it against the backtester on identical data.

| Instrument | Detection | End-to-end | Invented signals |
|---|---:|---:|---:|
| XAUUSD | 23/26 = 88% | 21/26 = 81% | 0 |
| EURUSD | 18/22 = 82% | 16/22 = 73% | 1 |
| USDJPY | 27/29 = 93% | 21/29 = 72% | 2 |
| **All** | **68/77 = 88%** | **58/77 = 75%** | 3 |

Consistent across three instruments with different volatilities and pip
conventions — a fix that only worked on gold would have been luck.

The remaining gap is trade **management**, not detection: both engines hold one
position per symbol but release it at slightly different moments, shifting which
later setups are blocked. Detection — the strategy itself — reproduces at 88%,
and the scanner never invents a signal.

---

## 8. Deployment

**Oracle Cloud was the original plan and failed** (account/capacity). Replaced
with GitHub Actions: free, no card, no server to maintain.

```
GitHub Actions (cron)  ->  github_poll.py  ->  Dukascopy feed
                                           ->  zone engine
                                           ->  Telegram -> your phone
                            state committed back to the repo
```

- **Schedule:** `*/20 7-22 * * 1-5` — every 20 min, London open → NY close
- **Heartbeat:** `10 7 * * 1-5` — daily liveness message
- **Budget:** ~1,056 runs/month ≈ 1,270 min against a 2,000 free allowance
- **State:** `signals/state.json` committed each run, so the bot remembers live
  signals and the repo stays active (GitHub pauses schedules on dormant repos)
- **Secrets:** `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` as encrypted repo secrets

Nothing needs activating or deactivating. Outside the window no run is scheduled
at all. Friday scanning stops at 20:00 UTC to avoid opening into the weekend gap.

Docker, systemd and the Oracle bootstrap script remain in the repo for a future
VPS move but are not used.

---

## 9. Operating it

```bash
python -m sd_bot alerts --test         # verify delivery
python -m sd_bot alerts --discover     # find Telegram chat id
python -m sd_bot signals               # run locally
python prove.py                        # re-run the four proofs
python records.py                      # regenerate the performance record
python validate.py --symbol XAUUSD     # year-by-year for one instrument
python sweep.py --grid quality         # parameter sweep, in/out of sample
python tests/test_engine.py            # 18 unit tests
python tests/test_parity.py            # backtest/live parity
```

**Alert types:** `APPROACHING` (price nearing a zone), `TRIGGERED` (entry
reached, full levels), `UPDATE` (filled / TP1 / TP2 / stopped / invalidated).

---

## 10. Known limitations

1. **Parity is 88%, not 100%.** The bot reproduces most of the strategy, not all.
   Alerts are candidates to check against the chart, not orders to fire blindly.
2. **2023 lost money** (−722). One losing year in five.
3. **Half the profit came from 2022** (+7,499 of +14,385), and November 2022
   alone was +2,761 — 19% of everything, in one month. That was a violent
   post-rate-hike regime. If it does not repeat, expect materially less than
   21.5% CAGR.
4. **A 12-loss streak occurred.** At 0.5% risk that is ~6% drawdown. Around loss
   eight you will be sure the bot is broken. It will not be.
5. **USDJPY is marginal** — PF 1.39, less than half the edge of the other two.
   Watch it; drop it if live results disappoint.
6. **No forward validation yet.** Every number here is historical.
7. **`sd_bot/live.py` (MT5 auto-trading) is not parity-checked** and carries a
   warning. Do not use it without fixing and testing it first.
8. **GitHub scheduler drifts** 5–15 min under load. Tolerable because signals are
   limit orders at a level, not market orders needing an exact second.

---

## 11. File map

| File | Purpose |
|---|---|
| `config.yaml` | Every rule and threshold, incl. per-symbol overrides |
| `sd_bot/zones.py` | Zone detection — the core |
| `sd_bot/scoring.py` | 0–100 quality grade |
| `sd_bot/structure.py` | Swings, trend, premium/discount curve |
| `sd_bot/signals.py` | Entry, stop, targets, profit-margin rule |
| `sd_bot/risk.py` | Sizing, loss limits, currency exposure |
| `sd_bot/backtest.py` | Event-driven portfolio simulator |
| `sd_bot/scanner.py` | Live signal service |
| `sd_bot/feed.py` | Real-time data (Dukascopy or MT5) |
| `sd_bot/notify.py` | Telegram / Discord / console / file delivery |
| `sd_bot/sources.py` | Dukascopy history + instrument catalogue |
| `github_poll.py` | One scan then exit — the CI entry point |
| `prove.py` | The four statistical proofs |
| `records.py` | Performance record + trade ledger |
| `validate.py` | Year-by-year validation |
| `sweep.py` | Parameter sweeps with in/out-of-sample split |
| `tests/test_engine.py` | 18 unit tests |
| `tests/test_parity.py` | Backtest ↔ live parity |

---

## 12. If you change anything

The discipline that made this work:

1. **Change one thing, then re-run `prove.py`.** If the null-hypothesis p-value
   rises above 0.05, the change destroyed the edge regardless of what the return
   says.
2. **Check in-sample AND out-of-sample** (`sweep.py`). A variant that only works
   on one half is curve fitting.
3. **Re-run `tests/test_parity.py`.** Changing the strategy without changing the
   scanner silently widens the gap between what you tested and what you run.
4. **Be suspicious of improvements.** Every large gain in this project turned out
   to be a bug — lookahead, a poisoned cache, or a filter that was not doing what
   it claimed.
