# Running the signal bot 24/7

**Your machine being off is not the problem to solve — the bot just has to run
somewhere that stays on.** It is a Python process, not a cloud service, so it
needs a host.

The useful part: the signal bot needs **no MetaTrader terminal, no Windows and
no broker account**. Market data comes from Dukascopy over plain HTTPS and
alerts go out over the Telegram API. That means it runs on the cheapest Linux
box you can rent, and it is verified to work with the `MetaTrader5` package
completely absent.

It is also tiny — under 300 MB of RAM, near-zero CPU between polls. Any entry
level VPS is overkill.

---

## Pick a host

| Option | Cost | Notes |
|---|---|---|
| **Oracle Cloud Always Free** | **$0 forever** | 4 ARM cores, 24 GB RAM. Genuinely free, not a trial. Sign-up needs a card for identity checks and ARM capacity can be scarce in popular regions — try a different region if it refuses. Best value by a distance. |
| **Hetzner CX22** | ~€4/mo | Fast, reliable, EU/US regions. What I would pick if the free tier annoys you. |
| **DigitalOcean / Vultr / Linode** | $5-6/mo | Simplest signup, good docs, one-click Docker images. |
| **Your broker's free VPS** | $0 with a funded account | Many FX brokers offer one. Usually Windows and intended for MT5 EAs, but it will run Python fine. |
| **Raspberry Pi at home** | ~$50 once | No monthly cost, but now your home power and internet are the single point of failure. |
| **GitHub Actions / free "serverless" tiers** | $0 | **Not recommended.** Cron granularity is 5 minutes at best, jobs are killed after a few hours, and the bot would re-seed history on every run. It fights the design. |

Location does not matter. The bot works on 15-minute bars, so tens of
milliseconds of latency are irrelevant.

---

## Deploy with Docker (recommended)

On a fresh Ubuntu box:

```bash
# 1. Docker
curl -fsSL https://get.docker.com | sh

# 2. Your project
git clone <your repo> sd-bot && cd sd-bot     # or scp the folder up

# 3. Secrets - never commit this file
cat > .env <<'EOF'
TELEGRAM_TOKEN=123456:AAF...
TELEGRAM_CHAT_ID=987654321
EOF
chmod 600 .env

# 4. Go
docker compose up -d --build
docker compose logs -f
```

That is it. `restart: unless-stopped` brings it back after a crash or a server
reboot, and the healthcheck restarts it if the poll loop ever wedges.

Copying your existing `data/*.csv` up first makes the first poll instant instead
of spending a few minutes seeding history. It is optional — the bot downloads
what it needs on its own.

---

## Deploy without Docker (systemd)

```bash
sudo adduser --system --group --home /home/trader trader
sudo -u trader git clone <your repo> /home/trader/sd-bot
cd /home/trader/sd-bot

sudo -u trader python3 -m venv .venv
sudo -u trader .venv/bin/pip install -r requirements.txt

sudo -u trader tee .env >/dev/null <<'EOF'
TELEGRAM_TOKEN=123456:AAF...
TELEGRAM_CHAT_ID=987654321
EOF
sudo chmod 600 .env

sudo cp deploy/sd-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sd-bot
journalctl -u sd-bot -f
```

Edit the paths and `User=` in the unit file if you did not use `/home/trader`.

---

## Confirm it is actually working

Do this before you trust it with anything.

```bash
# 1. Alerts reach your phone from the server, not just from your laptop
docker compose run --rm signals python -m sd_bot alerts --test

# 2. It is alive and polling
docker compose logs --tail 50

# 3. State is being written (this is what the healthcheck watches)
ls -l signals/state.json
```

Then leave `heartbeat_hours: 8` on in `config.yaml`. You will get a short "still
watching" message three times a day. **Silence from a signal bot is ambiguous —
a quiet market and a dead process look identical.** The heartbeat removes that
ambiguity, and this strategy averages about one signal per instrument per week,
so genuine silence is normal and expected.

---

## What still breaks it

Being honest about the failure modes:

- **Dukascopy outage.** Data stops; the supervisor retries with backoff. No
  signals until it returns. Nothing alerts you to this except the missing
  heartbeat.
- **Telegram rate limits or an outage.** Alerts still land in the console log
  and `signals/*.jsonl`, so nothing is lost, but you will not be pushed.
- **You revoke or rotate the bot token.** Delivery fails silently to your phone.
  Re-run `alerts --test` after any credential change.
- **Server clock drift.** Everything is UTC and session filtering depends on it.
  Keep `systemd-timesyncd` or `chrony` running; both are on by default.
- **Free-tier reclamation.** Oracle has been known to reclaim idle Always Free
  instances. Consider it best-effort rather than guaranteed.

---

## A note on cost versus benefit

This strategy produces roughly **2.3 signals per week** across the three
validated instruments. A $5/month VPS costs $60 a year to catch about 120
signals. That is fine if you act on them — but if you are at your desk during
London and New York anyway, running it locally during those hours costs nothing
and misses very little. The session window is 07:00-21:59 UTC, so a machine that
is on during your normal trading day already covers it.

Deploy to a server when you want the alerts to find you, not because the bot
needs to be somewhere clever.
