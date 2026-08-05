#!/usr/bin/env bash
# One-shot setup for a fresh Ubuntu server (Oracle Cloud Always Free, or any VPS).
#
#   bash deploy/bootstrap.sh
#
# Installs Docker, builds the signal bot, and starts it under a restart policy.
# Safe to re-run: every step checks before acting.

set -euo pipefail

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m !  %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX  %s\033[0m\n' "$*"; exit 1; }

cd "$(dirname "$0")/.."

# --- sanity ---------------------------------------------------------------
[ -f config.yaml ]    || die "config.yaml missing -- run this from the project root"
[ -d sd_bot ]         || die "sd_bot/ missing -- the upload is incomplete"
[ -f run_signals.py ] || die "run_signals.py missing -- the upload is incomplete"

if [ ! -f .env ]; then
    die ".env missing. Create it first:
    printf 'TELEGRAM_TOKEN=your_token\\nTELEGRAM_CHAT_ID=your_chat_id\\n' > .env
    chmod 600 .env"
fi
grep -q '^TELEGRAM_TOKEN=.\+' .env    || die "TELEGRAM_TOKEN is empty in .env"
grep -q '^TELEGRAM_CHAT_ID=.\+' .env  || die "TELEGRAM_CHAT_ID is empty in .env"
chmod 600 .env

# --- clock ----------------------------------------------------------------
# Session filtering is UTC-based, so a drifting clock silently changes which
# setups are eligible.
say "Setting clock to UTC"
sudo timedatectl set-timezone UTC 2>/dev/null || warn "could not set timezone"
sudo timedatectl set-ntp true 2>/dev/null     || warn "could not enable NTP"

# --- docker ---------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    say "Installing Docker"
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER" || true
    warn "You were added to the docker group. If the next step fails with a"
    warn "permissions error, log out and back in, then re-run this script."
else
    say "Docker already installed ($(docker --version))"
fi

DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

# --- build ----------------------------------------------------------------
say "Building the signal bot image"
$DOCKER compose build

# --- verify delivery BEFORE going live ------------------------------------
say "Testing Telegram delivery from this server"
if $DOCKER compose run --rm signals python -m sd_bot alerts --test; then
    say "Alert delivered -- check your phone"
else
    die "Telegram test failed. Fix .env before starting the bot."
fi

# --- run ------------------------------------------------------------------
say "Starting the bot"
$DOCKER compose up -d

sleep 5
$DOCKER compose ps

cat <<'EOF'

================================================================
  Running. Your machine can now be switched off.
================================================================

  Watch it:      docker compose logs -f
  Stop it:       docker compose down
  Restart it:    docker compose restart
  Update it:     git pull && docker compose up -d --build

  First scan seeds price history and takes a few minutes per
  instrument. After that each poll is instant.

  Roughly 2-3 signals a week is normal. Long quiet spells are the
  strategy working, not a fault -- the heartbeat message every 8
  hours is how you tell a quiet market from a dead process.

EOF
