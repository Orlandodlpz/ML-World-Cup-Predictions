#!/bin/bash
# Daily refresh: fetch latest WC26 results (simulator does this on startup),
# re-run the Monte Carlo simulation, and regenerate the HTML dashboard.
# Scheduled via launchd: ~/Library/LaunchAgents/com.elpato.wc2026-daily-refresh.plist

PROJECT="/Users/elpato/Claude/Projects/ML World Cup Predictions"
PY="$PROJECT/.venv/bin/python"
LOG="$PROJECT/logs/refresh.log"

mkdir -p "$PROJECT/logs"

{
  echo ""
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') daily refresh ====="
  cd "$PROJECT" || exit 1
  "$PY" models/simulator.py
  "$PY" outputs/report_generator.py
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') done ====="
} >> "$LOG" 2>&1
