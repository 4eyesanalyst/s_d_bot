"""Command line entry point.

    python -m sd_bot selftest              engine check, no terminal needed
    python -m sd_bot backtest              historical run on MT5 data
    python -m sd_bot scan                  what the bot sees right now
    python -m sd_bot live --dry-run        full loop, logs orders without sending
    python -m sd_bot live                  places orders on the connected account
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from . import broker
from . import timeframes as tfmod
from .backtest import Backtester, load_series
from .config import Config
from .data import Bars, resample, synthetic
from .journal import Journal, write_equity, write_trades
from .stats import compute, report


def _config(args) -> Config:
    cfg = Config.load(args.config)
    if getattr(args, "symbols", None):
        cfg.execution.symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if getattr(args, "start", None):
        cfg.backtest.start = args.start
    if getattr(args, "end", None):
        cfg.backtest.end = args.end
    if getattr(args, "risk", None):
        cfg.risk.risk_per_trade_pct = args.risk
    cfg.validate()
    return cfg


# -- commands ---------------------------------------------------------------


def cmd_selftest(args) -> int:
    """Exercise the whole pipeline on generated data, with no MT5 involved."""
    cfg = _config(args)
    # Three correlated USD-quoted pairs, so the portfolio risk caps and the
    # currency-exposure guard are exercised too.
    cfg.execution.symbols = ["EURUSD", "GBPUSD", "AUDUSD"]
    cfg.execution.bias_timeframe = "D1"
    cfg.execution.zone_timeframe = "H4"
    cfg.execution.entry_timeframe = "M15"
    cfg.validate()

    print("generating synthetic M15 series...")
    data = {}
    for offset, symbol in enumerate(cfg.execution.symbols):
        m15 = synthetic(symbol, "M15", n=args.bars, seed=args.seed + offset,
                        price=1.1000 + offset * 0.15)
        data[symbol] = {
            "M15": m15,
            "H4": resample(m15, "H4"),
            "D1": resample(m15, "D1"),
        }
        spans = "  ".join(f"{tf}={len(b)}" for tf, b in data[symbol].items())
        print(f"  {symbol}  {spans}  "
              f"{m15.datetimes()[0]:%Y-%m-%d} -> {m15.datetimes()[-1]:%Y-%m-%d}")

    bt = Backtester(cfg)
    result = bt.run(data)
    stats = compute(result)
    print(report(result, stats))
    print(
        "\nNOTE: this is randomly generated price, not a market. Random data has\n"
        "no supply/demand structure to exploit, so a loss here is the expected\n"
        "result and says nothing about the strategy. What it does prove is that\n"
        "the pipeline runs end to end: zones are found, scored, filtered, sized,\n"
        "executed and accounted for.\n\n"
        "Run `python -m sd_bot backtest` against real MT5 history to learn\n"
        "anything about the edge."
    )
    return 0


def _pick_source(requested: str) -> str:
    """Resolve the data source, preferring a real terminal when one is running."""
    if requested != "auto":
        return requested
    try:
        broker.ensure_initialized()
        return "mt5"
    except Exception:
        print("no MetaTrader 5 terminal reachable -- using Dukascopy history\n")
        return "dukascopy"


def cmd_backtest(args) -> int:
    from . import sources

    cfg = _config(args)
    source = _pick_source(args.source)
    if args.symbols is None and source == "dukascopy":
        cfg.execution.symbols = [
            s for s in cfg.execution.symbols if s.upper() in sources.CATALOGUE
        ] or sources.DEFAULT_BASKET

    data: dict[str, dict[str, Bars]] = {}
    specs = {}

    for symbol in cfg.execution.symbols:
        try:
            if source == "dukascopy":
                data[symbol] = sources.load_series(cfg, symbol, cache_dir=args.cache,
                                                   refresh=args.refresh)
                specs[symbol] = sources.spec_for(symbol)
            else:
                data[symbol] = load_series(cfg, symbol, cache_dir=args.cache,
                                           refresh=args.refresh)
                specs[symbol] = broker.spec_for(symbol)
        except Exception as exc:
            print(f"  skipped {symbol}: {exc}")
            continue
        spans = "  ".join(
            f"{tf}={len(b)}"
            for tf, b in sorted(data[symbol].items(),
                                key=lambda kv: tfmod.minutes(kv[0]))
        )
        print(f"  {symbol:<8} {spans}", flush=True)

    if not data:
        print("no data loaded -- nothing to test")
        return 1
    print(f"\n{len(data)} symbols loaded from {source}")

    result = Backtester(cfg, specs).run(data)
    stats = compute(result)
    print(report(result, stats))

    trades_path = write_trades(result.trades, f"{args.out}/trades.csv")
    equity_path = write_equity(result.equity_curve, f"{args.out}/equity.csv")
    print(f"\ntrades -> {trades_path}\nequity -> {equity_path}")
    return 0


def cmd_scan(args) -> int:
    """Print the live map: trend, zones, scores, and the plan for each."""
    from .indicators import atr as atr_fn
    from .data import recent
    from .live import BARS_BIAS, BARS_ENTRY, BARS_ZONE
    from .scoring import apply_score, explain
    from .signals import build_plan
    from .structure import Structure, trend_name
    from .zones import find_zones, settle

    cfg = _config(args)
    broker.ensure_initialized()
    print(f"account: {broker.account_summary()}\n")

    for symbol in cfg.execution.symbols:
        ex = cfg.execution
        entry = recent(symbol, ex.entry_timeframe, BARS_ENTRY)
        zone_bars = recent(symbol, ex.zone_timeframe, BARS_ZONE)
        bias_bars = recent(symbol, ex.bias_timeframe, BARS_BIAS)

        zs = Structure(zone_bars, cfg.structure.swing_lookback)
        bs = Structure(bias_bars, cfg.structure.swing_lookback)
        zone_pool = settle(find_zones(zone_bars, cfg.zone, zs), zone_bars)
        bias_pool = settle(find_zones(bias_bars, cfg.zone, bs), bias_bars)

        live = [z for z in zone_pool if z.is_live(cfg.zone, len(zone_bars) - 1)]
        bias_live = [z for z in bias_pool if not z.invalidated]
        trend = int(bs.trend[-1])
        for z in live:
            apply_score(z, cfg, bias_live, trend)
        live.sort(key=lambda z: -z.score)

        price = float(entry.close[-1])
        spec = broker.spec_for(symbol)
        atr_value = float(atr_fn(entry.high, entry.low, entry.close)[-1])
        curve = zs.curve_position(len(zone_bars) - 1, price)

        print("=" * 78)
        print(
            f"{symbol}  price={price:.{spec.digits}f}  "
            f"{ex.bias_timeframe} trend={trend_name(trend)}  "
            f"curve={curve:.0%} ({'discount' if curve < 0.5 else 'premium'})  "
            f"spread={broker.current_spread_points(symbol):.0f}pts"
        )
        if not live:
            print("  no live zones")
            continue

        for z in live[: args.top]:
            distance = abs(price - z.proximal) / max(atr_value, 1e-9)
            print(f"  {explain(z)}  ({distance:.1f} ATR away)")
            plan, why = build_plan(
                z, cfg, len(entry) - 1, atr_value, live + bias_live,
                broker.current_spread_points(symbol) * spec.point,
            )
            if plan is None:
                print(f"      -> no trade: {why}")
            else:
                print(
                    f"      -> {'BUY' if plan.is_long else 'SELL'} limit "
                    f"{plan.entry:.{spec.digits}f}  sl {plan.stop:.{spec.digits}f}  "
                    f"tp1 {plan.tp1:.{spec.digits}f}  tp2 {plan.tp2:.{spec.digits}f}  "
                    f"({plan.rr2:.1f}R)"
                )
    return 0


def cmd_live(args) -> int:
    from .live import LiveTrader

    cfg = _config(args)
    journal = Journal(args.journal)

    trader = LiveTrader(cfg, journal, dry_run=args.dry_run)
    if not args.dry_run and not trader.account["is_demo"]:
        print("\n" + "!" * 70)
        print("  This account is LIVE. Real money will be risked.")
        print(f"  Equity {trader.account['equity']:,.2f} {trader.account['currency']}")
        print(f"  Risk per trade: {cfg.risk.risk_per_trade_pct}%")
        print("!" * 70)
        if input("\nType 'I ACCEPT' to continue: ").strip() != "I ACCEPT":
            print("aborted")
            return 1

    trader.run()
    return 0


def cmd_signals(args) -> int:
    """Run the real-time signal scanner."""
    from .feed import Feed
    from .notify import build
    from .scanner import SignalScanner

    cfg = _config(args)
    if args.channels:
        cfg.alerts.channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    if args.poll:
        cfg.alerts.poll_seconds = args.poll

    out = build(cfg)
    problems = [ln for ln in out.preflight() if ln.strip().startswith("DEAD")]
    if problems and not args.force:
        print("channel problems:")
        for line in problems:
            print(line)
        print("\nFix these, or pass --force to run on the working channels only.")
        print("Telegram setup:  python -m sd_bot alerts --discover")
        return 1

    feed = Feed(args.feed, history_dir=args.cache)
    SignalScanner(cfg, out, feed, state_dir=cfg.alerts.directory).run()
    return 0


def cmd_alerts(args) -> int:
    """Test and configure alert delivery."""
    from .notify import DiscordNotifier, TelegramNotifier, build

    cfg = _config(args)

    if args.discover:
        tg = TelegramNotifier(cfg.alerts.telegram_token or None,
                              cfg.alerts.telegram_chat_id or None)
        if not tg.token:
            print("No bot token found.\n")
            print("1. Open Telegram, message @BotFather, send /newbot")
            print("2. Follow the prompts and copy the token it gives you")
            print("3. set TELEGRAM_TOKEN=<token>   (or put it in config.yaml)")
            print("4. Send your new bot any message, then re-run this")
            return 1
        chats = tg.discover_chat_id()
        if not chats and args.wait:
            ok, detail = tg.check()
            print(detail)
            print(f"waiting up to {args.wait}s for you to message the bot...")
            chats = tg.wait_for_chat(timeout=args.wait)
        if not chats:
            print("Token works, but no chat has messaged the bot yet.")
            print("Open Telegram, send your bot any message, then re-run with:")
            print("    python -m sd_bot alerts --discover --wait 180")
            return 1
        print("Chats that have messaged your bot:\n")
        for chat_id, label in chats:
            print(f"  chat id {chat_id}   ({label})")
        chat_id = chats[0][0]
        if getattr(args, "save", False):
            _save_chat_id(chat_id)
            print(f"\nsaved TELEGRAM_CHAT_ID={chat_id} to .env")
            print("verify with:  python -m sd_bot alerts --test")
        else:
            print("\nSet the one you want:")
            print(f"  set TELEGRAM_CHAT_ID={chat_id}")
            print(f"or save it:  python -m sd_bot alerts --discover --save")
        return 0

    out = build(cfg)
    print("channels:")
    for line in out.preflight():
        print(line)

    if args.test:
        print("\nsending test alert...")
        results = out.send(
            "[TEST] Signal bot delivery check",
            "BUY XAUUSD @ 4183.42\n"
            "STOP    4143.42   (40 pips risk)\n"
            "TP1     4263.42   (+80 pips, 2.0R)\n"
            "TP2     4343.42   (+160 pips, 4.0R)\n\n"
            "This is a test. No trade was signalled.",
            {"side": "BUY", "kind": "test"},
        )
        for name, ok in results.items():
            print(f"  {'sent' if ok else 'FAILED'}  {name}")
        if not results.get("telegram", True):
            print("\nTelegram failed. Run: python -m sd_bot alerts --discover")
    return 0


def _save_chat_id(chat_id: str) -> None:
    """Write the chat id into .env, preserving whatever else is there."""
    from pathlib import Path

    env = Path(".env")
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    out, seen = [], False
    for line in lines:
        if line.startswith("TELEGRAM_CHAT_ID="):
            out.append(f"TELEGRAM_CHAT_ID={chat_id}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"TELEGRAM_CHAT_ID={chat_id}")
    env.write_text("\n".join(out) + "\n", encoding="utf-8")


def cmd_account(args) -> int:
    broker.ensure_initialized()
    for key, value in broker.account_summary().items():
        print(f"  {key:<14} {value}")
    cfg = _config(args)
    print("\n  symbol specs:")
    for symbol in cfg.execution.symbols:
        try:
            spec = broker.spec_for(symbol)
            print(
                f"    {spec.name:<10} digits={spec.digits} point={spec.point} "
                f"tick_value={spec.tick_value} "
                f"vol={spec.volume_min}-{spec.volume_max}/{spec.volume_step}"
            )
        except Exception as exc:
            print(f"    {symbol:<10} {exc}")
    return 0


# -- wiring -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sd_bot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    st = sub.add_parser("selftest", help="run the engine on generated data")
    st.add_argument("--bars", type=int, default=40_000)
    st.add_argument("--seed", type=int, default=7)
    st.set_defaults(func=cmd_selftest)

    bt = sub.add_parser("backtest", help="historical run on MT5 data")
    bt.add_argument("--symbols", help="comma separated, overrides config")
    bt.add_argument("--start", help="YYYY-MM-DD")
    bt.add_argument("--end", help="YYYY-MM-DD")
    bt.add_argument("--risk", type=float, help="risk %% per trade")
    bt.add_argument("--cache", default="data", help="bar cache directory")
    bt.add_argument("--refresh", action="store_true", help="re-download bars")
    bt.add_argument("--out", default="results", help="output directory")
    bt.add_argument("--source", default="auto", choices=["auto", "mt5", "dukascopy"],
                    help="where bars come from (auto prefers a running terminal)")
    bt.set_defaults(func=cmd_backtest)

    sc = sub.add_parser("scan", help="show current zones and plans")
    sc.add_argument("--symbols")
    sc.add_argument("--top", type=int, default=6)
    sc.set_defaults(func=cmd_scan)

    lv = sub.add_parser("live", help="run the trading loop")
    lv.add_argument("--symbols")
    lv.add_argument("--risk", type=float)
    lv.add_argument("--dry-run", action="store_true",
                    help="log every decision but send no orders")
    lv.add_argument("--journal", default="journal")
    lv.set_defaults(func=cmd_live)

    sg = sub.add_parser("signals", help="run the real-time signal scanner")
    sg.add_argument("--symbols")
    sg.add_argument("--channels", help="override alert channels, comma separated")
    sg.add_argument("--poll", type=int, help="seconds between scans")
    sg.add_argument("--feed", default="auto", choices=["auto", "mt5", "dukascopy"])
    sg.add_argument("--cache", default="data")
    sg.add_argument("--force", action="store_true",
                    help="start even if some channels are misconfigured")
    sg.set_defaults(func=cmd_signals)

    al = sub.add_parser("alerts", help="test or configure alert delivery")
    al.add_argument("--test", action="store_true", help="send a test alert")
    al.add_argument("--discover", action="store_true",
                    help="find your Telegram chat id")
    al.add_argument("--wait", type=int, default=0, metavar="SECONDS",
                    help="with --discover, wait for you to message the bot")
    al.add_argument("--save", action="store_true",
                    help="write the discovered chat id into .env")
    al.set_defaults(func=cmd_alerts)

    ac = sub.add_parser("account", help="show account and symbol specs")
    ac.add_argument("--symbols")
    ac.set_defaults(func=cmd_account)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
