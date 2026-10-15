"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section markdown (analysts, research, trading, risk,
portfolio) plus a consolidated ``complete_report.md`` under ``save_path``. The
CLI and ``TradingAgentsGraph.save_reports`` both call this, so a headless / API
run produces the same on-disk report tree a CLI run does.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from tradingagents.agents.rating import parse_rating

# Matches the deterministic header render_sentiment_report() prepends to
# sentiment_report, e.g. "**Overall Sentiment:** **Bullish** (Score: 7.2/10)".
_SENTIMENT_HEADER_RE = re.compile(
    r"\*\*Overall Sentiment:\*\*\s*\*\*([^*]+)\*\*\s*\(Score:\s*([\d.]+)/10\)"
)
# Matches render_pm_decision()'s optional lines, e.g.
# "**Price Target**: 215.5" / "**Time Horizon**: 3-6 months".
_PRICE_TARGET_RE = re.compile(r"\*\*Price Target\*\*:\s*([^\n]+)")
_TIME_HORIZON_RE = re.compile(r"\*\*Time Horizon\*\*:\s*([^\n]+)")


def extract_screen_summary(final_state: dict) -> dict[str, Any]:
    """Pull a compact, structured summary out of a completed run's final
    state — the fields a screener result table wants (direction, sentiment,
    price target / time horizon) without re-parsing prose by hand.

    Reads fields TradingAgents' agents already produce as structured output
    (``PortfolioDecision.price_target``/``time_horizon``, the Sentiment
    Analyst's ``SentimentReport`` header) rather than asking any agent for a
    new "6-month potential" field — those already exist, just rendered to
    markdown for the saved report. Every field is best-effort: missing or
    unparseable input yields ``None``, never an exception, so a screener run
    on many tickers isn't derailed by one odd report.
    """
    empty = {
        "direction": None,
        "sentiment_band": None,
        "sentiment_score": None,
        "price_target": None,
        "time_horizon": None,
    }
    if not isinstance(final_state, dict):
        return empty

    final_trade_decision = final_state.get("final_trade_decision") or ""
    sentiment_report = final_state.get("sentiment_report") or ""

    direction = parse_rating(final_trade_decision) if final_trade_decision else None

    sentiment_match = _SENTIMENT_HEADER_RE.search(sentiment_report)
    sentiment_band = sentiment_match.group(1).strip() if sentiment_match else None
    sentiment_score = None
    if sentiment_match:
        try:
            sentiment_score = float(sentiment_match.group(2))
        except ValueError:
            pass

    price_target_match = _PRICE_TARGET_RE.search(final_trade_decision)
    time_horizon_match = _TIME_HORIZON_RE.search(final_trade_decision)

    return {
        "direction": direction,
        "sentiment_band": sentiment_band,
        "sentiment_score": sentiment_score,
        "price_target": price_target_match.group(1).strip() if price_target_match else None,
        "time_horizon": time_horizon_match.group(1).strip() if time_horizon_match else None,
    }

from tradingagents.agents.rating import run_rating


def analyst_names(keys) -> list[str]:
    """The analysts as users select them: the sentiment analyst's key is "social"."""
    return ["sentiment" if key == "social" else key for key in keys or []]


def _header(ticker: str, final_state: dict, settings: dict | None) -> str:
    """The report's title and what produced it: analysis date, version, models, analysts, vendors."""
    lines = [f"# Trading Analysis Report: {ticker}", ""]
    if final_state.get("trade_date"):
        lines.append(f"- Analysis date: {final_state['trade_date']}")
    if final_state.get("final_trade_decision"):
        lines.append(f"- Rating: {run_rating(final_state)}")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if settings:
        s = settings.get
        deep_provider = s("deep_think_provider") or s("llm_provider", "?")
        quick_provider = s("quick_think_provider") or s("llm_provider", "?")
        if deep_provider == quick_provider:
            models = f"{deep_provider}, deep {s('deep_think_llm', '?')}, quick {s('quick_think_llm', '?')}"
        else:   # each tier on its own provider (#1440)
            models = (f"deep {deep_provider} {s('deep_think_llm', '?')}, "
                      f"quick {quick_provider} {s('quick_think_llm', '?')}")
        lines.append(f"- TradingAgents {s('version', '?')}: {models}")
        lines.append(f"- Analysts: {', '.join(analyst_names(s('analysts')))}; "
                     f"research debate rounds {s('max_debate_rounds', '?')}, "
                     f"risk debate rounds {s('max_risk_discuss_rounds', '?')}")
        vendors = {**(s("data_vendors") or {}), **(s("tool_vendors") or {})}
        if vendors:
            lines.append("- Data vendors: " + ", ".join(f"{k} {v}" for k, v in vendors.items()))
    if final_state.get("memory_note"):
        lines.append(f"- Memory log: {final_state['memory_note']}")
    return "\n".join(lines) + "\n\n"


# The report's sections in order: (heading, folder, [(agent, file, state path)]).
_SECTIONS = (
    ("I. Analyst Team Reports", "1_analysts", (
        ("Market Analyst", "market.md", ("market_report",)),
        ("Sentiment Analyst", "sentiment.md", ("sentiment_report",)),
        ("News Analyst", "news.md", ("news_report",)),
        ("Fundamentals Analyst", "fundamentals.md", ("fundamentals_report",)),
    )),
    ("II. Research Team Decision", "2_research", (
        ("Bull Researcher", "bull.md", ("investment_debate_state", "bull_history")),
        ("Bear Researcher", "bear.md", ("investment_debate_state", "bear_history")),
        ("Research Manager", "manager.md", ("investment_plan",)),
    )),
    ("III. Trading Team Plan", "3_trading", (
        ("Trader", "trader.md", ("trader_investment_plan",)),
    )),
    ("IV. Risk Management Team Decision", "4_risk", (
        ("Aggressive Analyst", "aggressive.md", ("risk_debate_state", "aggressive_history")),
        ("Conservative Analyst", "conservative.md", ("risk_debate_state", "conservative_history")),
        ("Neutral Analyst", "neutral.md", ("risk_debate_state", "neutral_history")),
    )),
    ("V. Portfolio Manager Decision", "5_portfolio", (
        ("Portfolio Manager", "decision.md", ("final_trade_decision",)),
    )),
)


def _report_parts(final_state: dict) -> list[tuple[str, str, list[tuple[str, str, str]]]]:
    """The sections with something to say: (heading, folder, [(agent, file, text)])."""
    parts = []
    for heading, folder, agents in _SECTIONS:
        written = []
        for agent, filename, path in agents:
            value = final_state
            for key in path:
                value = (value or {}).get(key)
            if value:
                written.append((agent, filename, value))
        if written:
            parts.append((heading, folder, written))
    return parts


def write_report_tree(final_state: dict, ticker: str, save_path, settings: dict | None = None,
                      html: bool = True) -> Path:
    """Save a completed run's reports to ``save_path``; return the complete-report path.

    ``settings`` (``TradingAgentsGraph.run_settings()``) adds what produced the run
    to the report's header. ``html`` also writes ``complete_report.html``, one
    self-contained page of the same report (#1419).
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    parts = _report_parts(final_state)
    sections = []
    for heading, folder, written in parts:
        (save_path / folder).mkdir(exist_ok=True)
        for _agent, filename, text in written:
            (save_path / folder / filename).write_text(text, encoding="utf-8")
        content = "\n\n".join(f"### {agent}\n{text}" for agent, _filename, text in written)
        sections.append(f"## {heading}\n\n{content}")

    (save_path / "complete_report.md").write_text(
        _header(ticker, final_state, settings) + "\n\n".join(sections), encoding="utf-8"
    )
    if html:
        from tradingagents.report_html import render_report

        (save_path / "complete_report.html").write_text(
            render_report(ticker, final_state, settings, parts), encoding="utf-8"
        )
    return save_path / "complete_report.md"
