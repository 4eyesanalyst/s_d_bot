"""Package exactly what the server needs into one upload.

    python make_bundle.py

Produces sd-bot.tar.gz containing the bot, its config and the bootstrap script.
Deliberately excludes .env (secrets), data/ (hundreds of MB the server will
re-download itself) and every backtesting artefact.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

INCLUDE = [
    "sd_bot",
    "config.yaml",
    "requirements.txt",
    "run_signals.py",
    "Dockerfile",
    "docker-compose.yml",
    "deploy",
    "DEPLOY.md",
]

# Never ship: secrets, caches, results, or the venv.
SKIP_NAMES = {"__pycache__", ".env", ".venv", "data", "results", "signals"}
SKIP_SUFFIX = {".pyc", ".log", ".tar.gz"}


def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if any(p in SKIP_NAMES for p in parts):
        return None
    if Path(info.name).suffix in SKIP_SUFFIX:
        return None
    return info


def main() -> int:
    out = Path("sd-bot.tar.gz")
    missing = [p for p in INCLUDE if not Path(p).exists()]
    if missing:
        print(f"missing, will be skipped: {missing}")

    with tarfile.open(out, "w:gz") as tar:
        for item in INCLUDE:
            path = Path(item)
            if path.exists():
                tar.add(path, arcname=f"sd-bot/{item}", filter=keep)

    size = out.stat().st_size / 1024
    print(f"{out}  ({size:.0f} KB)")
    print("\nUpload it:")
    print(f"  scp -i <your-key.key> {out} ubuntu@<SERVER_IP>:~/")
    print("\nThen on the server:")
    print("  tar xzf sd-bot.tar.gz && cd sd-bot")
    print("  printf 'TELEGRAM_TOKEN=...\\nTELEGRAM_CHAT_ID=...\\n' > .env")
    print("  chmod 600 .env")
    print("  bash deploy/bootstrap.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
