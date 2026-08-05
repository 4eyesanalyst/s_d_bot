"""Trade journal and run log.

A supply/demand system declines far more setups than it takes, so the log
records refusals as well as fills. When the bot has a quiet week you want to be
able to answer "was that correct?" from the record rather than from memory.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .trades import ClosedTrade

_FIELDS = [
    "symbol", "side", "entry_time", "exit_time", "entry", "exit", "volume",
    "profit", "R", "mae_R", "mfe_R", "bars", "reason", "score", "zone",
]


def write_trades(trades: list[ClosedTrade], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        for t in trades:
            writer.writerow(t.as_row())
    return path


def write_equity(curve: list[tuple[int, float]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", "equity"])
        for ts, equity in curve:
            writer.writerow(
                [datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                 round(equity, 2)]
            )
    return path


class Journal:
    """Append-only log for a live or paper session."""

    def __init__(self, directory: str | Path = "journal"):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.log_path = self.dir / f"session_{stamp}.log"
        self.event_path = self.dir / f"events_{stamp}.jsonl"

    def log(self, message: str, echo: bool = True) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if echo:
            print(line, flush=True)

    def event(self, kind: str, **payload) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **payload,
        }
        with self.event_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
