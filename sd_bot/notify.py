"""Signal delivery.

Notifiers are independent and failure-isolated: if Telegram is unreachable the
alert still reaches the console and the log file. A trading alert you never see
is worse than no alert, so delivery never depends on a single channel.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


class Notifier:
    name = "notifier"

    def send(self, subject: str, body: str, meta: dict | None = None) -> bool:
        raise NotImplementedError

    def check(self) -> tuple[bool, str]:
        """Verify configuration before the session starts."""
        return True, "ok"


class ConsoleNotifier(Notifier):
    name = "console"

    _COLOURS = {"BUY": "\033[92m", "SELL": "\033[91m"}
    _RESET = "\033[0m"

    def send(self, subject: str, body: str, meta: dict | None = None) -> bool:
        side = (meta or {}).get("side", "")
        colour = self._COLOURS.get(side, "")
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"\n{colour}{'=' * 60}")
        print(f"  {stamp}Z  {subject}")
        print(f"{'=' * 60}{self._RESET}")
        print(body)
        return True


class FileNotifier(Notifier):
    """Append every signal to a JSONL file -- the auditable record."""

    name = "file"

    def __init__(self, directory: str = "signals"):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, subject: str, body: str, meta: dict | None = None) -> bool:
        stamp = datetime.now(timezone.utc)
        path = self.dir / f"signals_{stamp:%Y%m}.jsonl"
        record = {"time": stamp.isoformat(), "subject": subject,
                  "body": body, **(meta or {})}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return True


class TelegramNotifier(Notifier):
    """Push to a Telegram chat via the Bot API.

    Setup, once:
      1. Message @BotFather, send /newbot, follow the prompts, copy the token.
      2. Send your new bot any message.
      3. Run `python -m sd_bot alerts --discover` to find your chat id.
    """

    name = "telegram"
    API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self.token = token or os.getenv("TELEGRAM_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

    def _call(self, method: str, payload: dict, timeout: int = 15) -> dict:
        url = self.API.format(token=self.token, method=method)
        data = urllib.parse.urlencode(payload).encode()
        with urllib.request.urlopen(url, data=data, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def check(self) -> tuple[bool, str]:
        if not self.token:
            return False, "TELEGRAM_TOKEN not set"

        # Validate the token *before* worrying about the chat id, so a missing
        # chat never masks a bad token. Diagnosing one problem at a time is the
        # whole point of a preflight check.
        try:
            me = self._call("getMe", {})
            if not me.get("ok"):
                return False, f"getMe failed: {me}"
            username = me["result"].get("username")
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                return False, "token rejected (401) -- check TELEGRAM_TOKEN"
            return False, f"HTTP {exc.code}"
        except Exception as exc:
            return False, f"{exc!r}"

        if not self.chat_id:
            return False, (
                f"token OK (@{username}) but TELEGRAM_CHAT_ID not set -- "
                f"message https://t.me/{username} then run: alerts --discover"
            )
        return True, f"connected as @{username}"

    def wait_for_chat(self, timeout: int = 180, interval: int = 3
                      ) -> list[tuple[str, str]]:
        """Poll until someone messages the bot, or ``timeout`` expires."""
        import time as _time

        deadline = _time.time() + timeout
        while _time.time() < deadline:
            chats = self.discover_chat_id()
            if chats:
                return chats
            _time.sleep(interval)
        return []

    def send(self, subject: str, body: str, meta: dict | None = None) -> bool:
        text = f"*{_escape(subject)}*\n```\n{body}\n```"
        try:
            result = self._call("sendMessage", {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "MarkdownV2",
                "disable_web_page_preview": "true",
            })
            return bool(result.get("ok"))
        except Exception:
            # Retry once as plain text: a formatting error must not lose the alert.
            try:
                result = self._call("sendMessage", {
                    "chat_id": self.chat_id,
                    "text": f"{subject}\n\n{body}",
                })
                return bool(result.get("ok"))
            except Exception:
                return False

    def discover_chat_id(self) -> list[tuple[str, str]]:
        """Chats that have messaged this bot recently."""
        found = []
        try:
            updates = self._call("getUpdates", {})
        except Exception:
            return found
        for item in updates.get("result", []):
            chat = (item.get("message") or item.get("channel_post") or {}).get("chat")
            if not chat:
                continue
            label = chat.get("username") or chat.get("title") or chat.get("first_name")
            pair = (str(chat["id"]), str(label))
            if pair not in found:
                found.append(pair)
        return found


class DiscordNotifier(Notifier):
    name = "discord"

    def __init__(self, webhook: str | None = None):
        self.webhook = webhook or os.getenv("DISCORD_WEBHOOK", "")

    def check(self) -> tuple[bool, str]:
        if not self.webhook:
            return False, "DISCORD_WEBHOOK not set"
        return True, "webhook configured"

    def send(self, subject: str, body: str, meta: dict | None = None) -> bool:
        payload = json.dumps({"content": f"**{subject}**\n```\n{body}\n```"}).encode()
        request = urllib.request.Request(
            self.webhook, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as resp:
                return resp.status in (200, 204)
        except Exception:
            return False


class DesktopNotifier(Notifier):
    """Windows toast via PowerShell. Silently inert elsewhere."""

    name = "desktop"

    def check(self) -> tuple[bool, str]:
        if os.name != "nt":
            return False, "Windows only"
        return True, "ok"

    def send(self, subject: str, body: str, meta: dict | None = None) -> bool:
        if os.name != "nt":
            return False
        import subprocess

        first = body.strip().splitlines()[0] if body.strip() else ""
        script = (
            '[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications,'
            ' ContentType = WindowsRuntime] | Out-Null;'
            '$t = [Windows.UI.Notifications.ToastNotificationManager]::'
            'GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);'
            f'$t.GetElementsByTagName("text")[0].AppendChild($t.CreateTextNode("{_ps(subject)}")) | Out-Null;'
            f'$t.GetElementsByTagName("text")[1].AppendChild($t.CreateTextNode("{_ps(first)}")) | Out-Null;'
            '[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier'
            '("SD Bot").Show([Windows.UI.Notifications.ToastNotification]::new($t));'
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, timeout=20, check=False,
            )
            return True
        except Exception:
            return False


class Broadcaster:
    """Fan one alert out to every configured channel, isolating failures."""

    def __init__(self, notifiers: list[Notifier]):
        self.notifiers = notifiers
        self.failures: dict[str, int] = {}

    def send(self, subject: str, body: str, meta: dict | None = None) -> dict[str, bool]:
        results = {}
        for n in self.notifiers:
            try:
                ok = n.send(subject, body, meta)
            except Exception:
                ok = False
            results[n.name] = ok
            if not ok:
                self.failures[n.name] = self.failures.get(n.name, 0) + 1
        return results

    def preflight(self) -> list[str]:
        lines = []
        for n in self.notifiers:
            ok, detail = n.check()
            lines.append(f"  {'OK  ' if ok else 'DEAD'} {n.name:<10} {detail}")
        return lines


def build(cfg) -> Broadcaster:
    """Assemble notifiers from config. Console and file are always present."""
    notifiers: list[Notifier] = [ConsoleNotifier(), FileNotifier(cfg.alerts.directory)]
    for channel in cfg.alerts.channels:
        channel = channel.lower().strip()
        if channel == "telegram":
            notifiers.append(TelegramNotifier(cfg.alerts.telegram_token or None,
                                              cfg.alerts.telegram_chat_id or None))
        elif channel == "discord":
            notifiers.append(DiscordNotifier(cfg.alerts.discord_webhook or None))
        elif channel == "desktop":
            notifiers.append(DesktopNotifier())
    return Broadcaster(notifiers)


_MD_SPECIALS = r"_*[]()~`>#+-=|{}.!"


def _escape(text: str) -> str:
    for ch in _MD_SPECIALS:
        text = text.replace(ch, f"\\{ch}")
    return text


def _ps(text: str) -> str:
    return text.replace('"', "'").replace("`", "'").replace("$", "")
