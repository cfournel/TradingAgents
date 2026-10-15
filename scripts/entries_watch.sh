#!/usr/bin/env bash
# Intraday watcher (paper): send proposals whose pullback level is reached, close positions that broke their stop.
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/alpaca_orders.py fill
.venv/bin/python scripts/alpaca_orders.py stops --close
