#!/usr/bin/env python3
"""Non-interactive, multi-ticker entry point for TradingAgents.

Runs ``TradingAgentsGraph.propagate()`` for each ticker and prints a single
JSON object to stdout. Intended to be invoked from outside the container
(e.g. ``docker compose run --rm -T --entrypoint python tradingagents
scripts/batch_analyze.py --tickers AAPL,NVDA,BTC-USD`` — note
``--entrypoint python``, required because the image's ENTRYPOINT is the
``tradingagents`` CLI itself) by an external scheduler or agent that
cannot drive the interactive ``tradingagents`` CLI.

A failure on one ticker is recorded in its result entry and does not stop
the rest of the batch.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

from cli.models import AnalystType
from cli.prompts import detect_asset_type, filter_analysts_for_asset_type
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.reporting import extract_screen_summary, write_report_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated ticker list, e.g. AAPL,NVDA,BTC-USD",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Analysis date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--horizon",
        default="position",
        choices=["swing", "position"],
        help=(
            "Trade horizon passed to every ticker's run: 'swing' (a quick trade, "
            "a few days) or 'position' (a hold, multi-month trend). Default: position."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use the quick-think model for every agent, including the research "
            "manager and portfolio manager (normally the deep-think model). "
            "Useful when deep_think_llm is a much larger/slower model than "
            "quick_think_llm and this run is a cheap pre-filter rather than a "
            "final decision, e.g. the screener's shortlist deep-dive."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def analyze_one(
    ticker: str, trade_date: str, debug: bool, horizon: str = "position", quick: bool = False,
) -> dict:
    asset_type = detect_asset_type(ticker)
    analysts = filter_analysts_for_asset_type(list(AnalystType), asset_type)
    config = DEFAULT_CONFIG.copy()
    if quick:
        config["deep_think_llm"] = config["quick_think_llm"]
    graph = TradingAgentsGraph(
        selected_analysts=tuple(a.value for a in analysts),
        debug=debug,
        config=config,
    )
    final_state, decision = graph.propagate(
        ticker, trade_date, asset_type=asset_type.value, horizon=horizon,
    )
    summary = extract_screen_summary(final_state)

    # Render the same consolidated markdown report the interactive CLI
    # writes to disk, so callers get more than the one-line decision
    # without having to reassemble final_state themselves. Written to a
    # throwaway directory since the caller only wants the string back
    # (the container is typically ephemeral, `docker compose run --rm`).
    report_markdown = None
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_file = write_report_tree(final_state, ticker, Path(tmp_dir))
            report_markdown = report_file.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - report rendering must not fail the run
        report_markdown = f"(failed to render full report: {type(exc).__name__}: {exc})"

    return {
        "ticker": ticker,
        "date": trade_date,
        "asset_type": asset_type.value,
        "horizon": horizon,
        "decision": decision,
        "sentiment_band": summary["sentiment_band"],
        "sentiment_score": summary["sentiment_score"],
        "price_target": summary["price_target"],
        "time_horizon": summary["time_horizon"],
        "report": report_markdown,
    }


def main() -> int:
    args = parse_args()
    trade_date = args.date or date.today().isoformat()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print(json.dumps({"date": trade_date, "results": [], "error": "no tickers provided"}))
        return 1

    results = []
    for ticker in tickers:
        try:
            results.append(analyze_one(ticker, trade_date, args.debug, args.horizon, args.quick))
        except Exception as exc:  # noqa: BLE001 - batch must not die on one bad ticker
            results.append(
                {"ticker": ticker, "date": trade_date, "error": f"{type(exc).__name__}: {exc}"}
            )

    print(json.dumps({"date": trade_date, "results": results}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
