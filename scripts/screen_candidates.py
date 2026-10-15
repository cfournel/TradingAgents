#!/usr/bin/env python3
"""Non-interactive candidate screener — the "which tickers are worth a deep
dive" pre-filter, as opposed to ``batch_analyze.py`` which runs the full
multi-agent pipeline on tickers you already picked.

Prints a single JSON object to stdout:
    {"candidates": [{"ticker": ..., "asset_type": ..., "source": ...,
                      "metrics": {...}}, ...]}

Used by the hermes-tradingagents-plugin as the fallback screener path (its
own in-process screener is preferred when yfinance is importable directly
in Hermes's Python environment; this script is the same logic run inside
whatever environment TradingAgents itself runs in, so it's always
available there — see tradingagents/dataflows/screener.py).
"""

from __future__ import annotations

import argparse
import json
import sys

from tradingagents.dataflows.screener import discover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-classes",
        required=True,
        help="Comma-separated asset classes: stock,crypto,commodity",
    )
    parser.add_argument("--risk", required=True, choices=["low", "medium", "high", "any"])
    parser.add_argument("--horizon", required=True, choices=["swing", "position"])
    parser.add_argument("--limit", type=int, default=20, help="Max candidates per asset class")
    parser.add_argument(
        "--price-range",
        default="all",
        choices=["all", "pennies", "5_50", "51_100", "101_300", "301_plus"],
        help="Price filter: all, pennies (<$5), 5_50, 51_100, 101_300, or 301_plus. Default: all.",
    )
    parser.add_argument("--min-price", type=float, help="Stocks via Alpaca: exact lower price bound (overrides --price-range)")
    parser.add_argument("--max-price", type=float, help="Stocks via Alpaca: exact upper price bound (overrides --price-range)")
    parser.add_argument("--min-dollar-volume", type=float, default=5_000_000, help="Stocks via Alpaca: min 20d average dollar volume")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asset_classes = [a.strip() for a in args.asset_classes.split(",") if a.strip()]

    try:
        if args.min_price is not None or args.max_price is not None or args.risk == "any":
            from tradingagents.dataflows.alpaca_screener import screen_alpaca

            candidates = screen_alpaca(
                risk=args.risk, horizon=args.horizon, limit=args.limit,
                min_price=args.min_price if args.min_price is not None else 2.0,
                max_price=args.max_price if args.max_price is not None else 1000.0,
                min_dollar_volume=args.min_dollar_volume,
            )
            print(json.dumps({"candidates": candidates}, default=str))
            return 0
        candidates = discover(asset_classes, args.risk, args.horizon, args.limit, args.price_range)
    except Exception as exc:  # noqa: BLE001 — report as JSON, not a traceback
        print(json.dumps({"candidates": [], "error": f"{type(exc).__name__}: {exc}"}))
        return 1

    print(json.dumps({"candidates": candidates}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
