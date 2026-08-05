"""MetaTrader 5 adapter: connection, symbol specs and order plumbing.

Every MT5 import is local to a function so the rest of the package (zones,
scoring, backtester) runs on a machine with no terminal installed.
"""

from __future__ import annotations

import os
from datetime import datetime

from .trades import LONG, SymbolSpec, side_name

_initialized = False

# Fallbacks used when no terminal is available. Good enough to backtest with,
# never used when MT5 can answer for itself.
_ESTIMATES: dict[str, tuple[int, float, float, float]] = {
    # prefix: (digits, tick_size, tick_value_per_lot, contract_size)
    "XAU": (2, 0.01, 1.0, 100.0),
    "XAG": (3, 0.001, 5.0, 5000.0),
    "JPY": (3, 0.001, 0.67, 100_000.0),
}


def estimate_spec(symbol: str) -> SymbolSpec:
    """Best-effort contract spec for offline work."""
    s = symbol.upper()
    if s.startswith("XAU") or s.startswith("XAG"):
        digits, tick, value, contract = _ESTIMATES["XAU" if s.startswith("XAU") else "XAG"]
    elif s.endswith("JPY"):
        digits, tick, value, contract = _ESTIMATES["JPY"]
    elif len(s) == 6 and s.isalpha():
        digits, tick, value, contract = 5, 0.00001, 1.0, 100_000.0
    else:
        digits, tick, value, contract = 2, 0.01, 1.0, 1.0
    return SymbolSpec(
        name=symbol,
        digits=digits,
        point=tick,
        tick_size=tick,
        tick_value=value,
        contract_size=contract,
    )


def ensure_initialized() -> None:
    """Connect to the terminal, using MT5_* environment variables if present."""
    global _initialized
    if _initialized:
        return
    import MetaTrader5 as mt5

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH")

    kwargs: dict = {}
    if path:
        kwargs["path"] = path
    if login and password and server:
        kwargs.update(login=int(login), password=password, server=server)

    if not mt5.initialize(**kwargs):
        raise RuntimeError(
            "could not connect to MetaTrader 5 "
            f"(last_error={mt5.last_error()}). Start the terminal and log in, or "
            "set MT5_LOGIN / MT5_PASSWORD / MT5_SERVER (and MT5_PATH if the "
            "terminal is not in its default location)."
        )
    _initialized = True


def shutdown() -> None:
    global _initialized
    if not _initialized:
        return
    import MetaTrader5 as mt5

    mt5.shutdown()
    _initialized = False


def ensure_symbol(symbol: str) -> None:
    """Make sure the symbol exists and is visible in Market Watch."""
    import MetaTrader5 as mt5

    ensure_initialized()
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(
            f"symbol {symbol!r} not found on this broker. Check the exact name "
            "in Market Watch -- brokers add suffixes such as EURUSD.m or EURUSDpro."
        )
    if not info.visible and not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"could not select {symbol!r} in Market Watch")


def spec_for(symbol: str) -> SymbolSpec:
    """Real contract spec from the terminal."""
    import MetaTrader5 as mt5

    ensure_symbol(symbol)
    info = mt5.symbol_info(symbol)
    return SymbolSpec(
        name=symbol,
        digits=info.digits,
        point=info.point,
        tick_size=info.trade_tick_size or info.point,
        tick_value=info.trade_tick_value or 1.0,
        volume_min=info.volume_min,
        volume_max=info.volume_max,
        volume_step=info.volume_step,
        contract_size=info.trade_contract_size,
    )


def account_equity() -> float:
    import MetaTrader5 as mt5

    ensure_initialized()
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"account_info failed: {mt5.last_error()}")
    return float(info.equity)


def account_summary() -> dict:
    import MetaTrader5 as mt5

    ensure_initialized()
    info = mt5.account_info()
    return {
        "login": info.login,
        "server": info.server,
        "currency": info.currency,
        "balance": float(info.balance),
        "equity": float(info.equity),
        "margin_free": float(info.margin_free),
        "leverage": info.leverage,
        "trade_allowed": bool(info.trade_allowed),
        "is_demo": info.trade_mode != 0,
    }


def current_spread_points(symbol: str) -> float:
    import MetaTrader5 as mt5

    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None or info.point <= 0:
        return float("inf")
    return (tick.ask - tick.bid) / info.point


def open_positions(magic: int | None = None) -> list:
    import MetaTrader5 as mt5

    ensure_initialized()
    positions = mt5.positions_get() or []
    if magic is None:
        return list(positions)
    return [p for p in positions if p.magic == magic]


def send_market_order(
    symbol: str,
    direction: int,
    volume: float,
    stop: float,
    take_profit: float,
    magic: int,
    deviation: int,
    comment: str,
) -> tuple[bool, str, int]:
    """Place a market order. Returns ``(ok, message, ticket)``."""
    import MetaTrader5 as mt5

    ensure_symbol(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, f"no tick data for {symbol}", 0

    price = tick.ask if direction == LONG else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": round(volume, 2),
        "type": mt5.ORDER_TYPE_BUY if direction == LONG else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": stop,
        "tp": take_profit,
        "deviation": deviation,
        "magic": magic,
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_mode(symbol),
    }
    result = mt5.order_send(request)
    if result is None:
        return False, f"order_send returned None: {mt5.last_error()}", 0
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"retcode {result.retcode}: {result.comment}", 0
    return True, (
        f"{side_name(direction)} {volume:.2f} {symbol} @ {result.price:.5f}"
    ), int(result.order)


def send_pending_limit(
    symbol: str,
    direction: int,
    volume: float,
    price: float,
    stop: float,
    take_profit: float,
    magic: int,
    comment: str,
    expiry: datetime | None = None,
) -> tuple[bool, str, int]:
    """Rest a limit order at the zone's proximal line -- the set-and-forget entry."""
    import MetaTrader5 as mt5

    ensure_symbol(symbol)
    info = mt5.symbol_info(symbol)
    digits = info.digits

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": round(volume, 2),
        "type": mt5.ORDER_TYPE_BUY_LIMIT if direction == LONG
        else mt5.ORDER_TYPE_SELL_LIMIT,
        "price": round(price, digits),
        "sl": round(stop, digits),
        "tp": round(take_profit, digits),
        "magic": magic,
        "comment": comment[:31],
        "type_filling": _filling_mode(symbol),
    }
    if expiry is not None:
        request["type_time"] = mt5.ORDER_TIME_SPECIFIED
        request["expiration"] = expiry
    else:
        request["type_time"] = mt5.ORDER_TIME_GTC

    result = mt5.order_send(request)
    if result is None:
        return False, f"order_send returned None: {mt5.last_error()}", 0
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"retcode {result.retcode}: {result.comment}", 0
    return True, (
        f"{side_name(direction)} LIMIT {volume:.2f} {symbol} @ {price:.{digits}f} "
        f"sl={stop:.{digits}f} tp={take_profit:.{digits}f}"
    ), int(result.order)


def pending_orders(magic: int | None = None) -> list:
    import MetaTrader5 as mt5

    ensure_initialized()
    orders = mt5.orders_get() or []
    if magic is None:
        return list(orders)
    return [o for o in orders if o.magic == magic]


def cancel_order(ticket: int) -> tuple[bool, str]:
    import MetaTrader5 as mt5

    result = mt5.order_send(
        {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
    )
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        detail = mt5.last_error() if result is None else result.comment
        return False, f"cancel failed: {detail}"
    return True, f"cancelled order {ticket}"


def modify_stop(ticket: int, stop: float, take_profit: float) -> tuple[bool, str]:
    import MetaTrader5 as mt5

    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return False, f"position {ticket} not found"
    p = positions[0]
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": p.symbol,
        "sl": stop,
        "tp": take_profit,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        detail = mt5.last_error() if result is None else result.comment
        return False, f"modify failed: {detail}"
    return True, f"stop -> {stop:.5f}"


def close_partial(
    ticket: int, volume: float, magic: int, deviation: int, comment: str
) -> tuple[bool, str]:
    import MetaTrader5 as mt5

    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return False, f"position {ticket} not found"
    p = positions[0]
    tick = mt5.symbol_info_tick(p.symbol)
    closing_long = p.type == mt5.POSITION_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": p.symbol,
        "volume": round(min(volume, p.volume), 2),
        "type": mt5.ORDER_TYPE_SELL if closing_long else mt5.ORDER_TYPE_BUY,
        "position": ticket,
        "price": tick.bid if closing_long else tick.ask,
        "deviation": deviation,
        "magic": magic,
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_mode(p.symbol),
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        detail = mt5.last_error() if result is None else result.comment
        return False, f"close failed: {detail}"
    return True, f"closed {volume:.2f} of {ticket}"


def _filling_mode(symbol: str):
    """Pick a fill mode the broker actually supports for this symbol."""
    import MetaTrader5 as mt5

    info = mt5.symbol_info(symbol)
    modes = getattr(info, "filling_mode", 0)
    if modes & 1:  # SYMBOL_FILLING_FOK
        return mt5.ORDER_FILLING_FOK
    if modes & 2:  # SYMBOL_FILLING_IOC
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN
