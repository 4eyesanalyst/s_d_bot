# Signal bot only. No MetaTrader, no GUI, no broker account.
FROM python:3.12-slim

# Every timestamp in this system is UTC; make the container agree.
ENV TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sd_bot/ ./sd_bot/
COPY run_signals.py config.yaml ./

# Bar cache and signal state survive restarts when these are mounted.
VOLUME ["/app/data", "/app/signals"]

# Fails if no signal state has been written in the last 30 minutes, which is
# the symptom of a wedged poll loop rather than a quiet market.
HEALTHCHECK --interval=5m --timeout=20s --start-period=15m --retries=3 \
    CMD python -c "import pathlib,time,sys; \
p=pathlib.Path('/app/signals/state.json'); \
sys.exit(0 if p.exists() and time.time()-p.stat().st_mtime < 1800 else 1)"

CMD ["python", "run_signals.py"]
