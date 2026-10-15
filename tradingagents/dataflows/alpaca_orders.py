"""Order proposals and manual-confirmation execution on Alpaca (paper by default).

Flow: ``propose`` turns analysis results into sized order proposals (stored on
disk, nothing is sent); ``execute`` sends one stored proposal, and only when
called explicitly with its id; ``check_stops`` closes positions that broke
their stop level (needed because fractional orders cannot carry a stop).

Sizing (fixed-fractional, deliberately conservative):
  * only Buy / Overweight decisions become proposals;
  * risk per trade = ``risk_per_trade`` of capital (default 2%);
  * stop = entry - k * ATR (k = 2 swing, 3 position), never closer than 5%;
  * position = risk amount / stop distance, capped at ``max_position_pct`` of
    capital (default 25%) and at the available buying power;
  * at most ``max_positions`` open positions.

Paper vs live follows ``ALPACA_PAPER_TRADE`` (default true). Credentials come
from the same place as the screener.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from tradingagents.dataflows.alpaca_screener import DATA_URL, TRADING_URL, _credentials, _get

PROPOSALS_PATH = Path.home() / ".hermes" / "tradingagents" / "proposals.json"
BUYABLE = {"buy", "overweight"}


def _paper() -> bool:
    import os

    return os.environ.get("ALPACA_PAPER_TRADE", "true").strip().lower() not in ("0", "false", "no")


def _base() -> str:
    return TRADING_URL if _paper() else "https://api.alpaca.markets"


def account() -> dict:
    return _get(f"{_base()}/v2/account", _credentials())


def positions() -> list[dict]:
    return _get(f"{_base()}/v2/positions", _credentials())


def _load() -> dict[str, dict]:
    return json.loads(PROPOSALS_PATH.read_text()) if PROPOSALS_PATH.exists() else {}


def _save(data: dict[str, dict]) -> None:
    PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROPOSALS_PATH.write_text(json.dumps(data, indent=2))


def size_position(price: float, atr_pct: float, horizon: str, capital: float, buying_power: float,
                  risk_per_trade: float = 0.02, max_position_pct: float = 0.25) -> dict[str, float] | None:
    """Notional and stop for one trade, or None when it cannot be sized."""
    if price <= 0 or capital <= 0:
        return None
    k = 2.0 if horizon == "swing" else 3.0
    stop_pct = max(k * atr_pct, 0.05)
    notional = min(capital * risk_per_trade / stop_pct, capital * max_position_pct, buying_power)
    if notional < 1.0:
        # Alpaca's minimum order is $1. On a tiny account the exact size can fall just under it;
        # round up to the minimum only while the risk stays within twice the per-trade budget.
        if min(capital, buying_power) >= 1.0 and 1.0 * stop_pct <= 2 * capital * risk_per_trade:
            notional = 1.0
        else:
            return None
    return {
        "notional": round(notional, 2),
        "stop_price": round(price * (1 - stop_pct), 2),
        "stop_pct": round(stop_pct, 4),
        "risk_amount": round(notional * stop_pct, 2),
    }


def _latest(symbol: str) -> float | None:
    snap = _get(f"{DATA_URL}/v2/stocks/snapshots", _credentials(), {"symbols": symbol, "feed": "iex"})
    s = snap.get(symbol) or {}
    return (s.get("latestTrade") or {}).get("p") or (s.get("dailyBar") or {}).get("c")


def propose(results: list[dict], candidates: dict[str, dict] | None = None, capital: float | None = None,
            max_positions: int = 5, risk_per_trade: float = 0.02, max_position_pct: float = 0.25) -> list[dict]:
    """Proposals from ``batch_analyze`` results. ``candidates`` maps ticker -> screener metrics (for ATR)."""
    acct = account()
    capital = capital if capital is not None else float(acct["equity"])
    bp = float(acct["buying_power"])
    held = {p["symbol"] for p in positions()}
    stored = _load()
    open_slots = max_positions - len(held)
    out = []
    for r in results:
        ticker, decision = r.get("ticker"), str(r.get("decision") or "").lower()
        if decision not in BUYABLE or not ticker or ticker in held or open_slots <= 0:
            continue
        price = _latest(ticker)
        metrics = (candidates or {}).get(ticker, {}).get("metrics", {})
        if not price:
            continue
        sizing = size_position(price, metrics.get("atr_pct", 0.04), r.get("horizon") or "swing",
                               capital, min(bp, capital), risk_per_trade, max_position_pct)
        if not sizing:
            continue
        pid = uuid.uuid4().hex[:8]
        stored[pid] = {
            "id": pid, "ticker": ticker, "side": "buy", "status": "proposed", "created_at": int(time.time()),
            "decision": r.get("decision"), "horizon": r.get("horizon"), "entry_price": price,
            "price_target": r.get("price_target"), "time_horizon": r.get("time_horizon"),
            "paper": _paper(), **sizing,
        }
        out.append(stored[pid])
        open_slots -= 1
    _save(stored)
    return out


def execute(proposal_id: str) -> dict:
    """Send one stored proposal as a fractional market buy (DAY). Never called implicitly."""
    stored = _load()
    p = stored.get(proposal_id)
    if not p:
        raise KeyError(f"no proposal {proposal_id!r}")
    if p["status"] != "proposed":
        raise ValueError(f"proposal {proposal_id} is {p['status']}, not 'proposed'")
    if p["paper"] != _paper():
        raise ValueError("proposal was made in a different paper/live mode; re-propose")
    r = requests.post(f"{_base()}/v2/orders", headers=_credentials(), timeout=30, json={
        "symbol": p["ticker"], "notional": str(p["notional"]), "side": "buy",
        "type": "market", "time_in_force": "day",
    })
    if r.status_code >= 300:
        raise RuntimeError(f"Alpaca refused the order: {r.status_code} {r.text}")
    p.update(status="submitted", order_id=r.json()["id"], submitted_at=int(time.time()))
    _save(stored)
    return p


def check_stops(close: bool = False) -> list[dict]:
    """Submitted proposals whose position broke the stop; closes them only when ``close`` is true."""
    stored = _load()
    pos = {p["symbol"]: p for p in positions()}
    hits = []
    for p in stored.values():
        cur = pos.get(p["ticker"])
        if p["status"] != "submitted" or not cur:
            continue
        price = float(cur["current_price"])
        if price <= p["stop_price"]:
            hit = {**p, "current_price": price}
            if close:
                r = requests.delete(f"{_base()}/v2/positions/{p['ticker']}", headers=_credentials(), timeout=30)
                hit["closed"] = r.status_code < 300
                if hit["closed"]:
                    p["status"] = "stopped"
            hits.append(hit)
    if close:
        _save(stored)
    return hits


def list_proposals(status: str | None = None) -> list[dict[str, Any]]:
    return [p for p in _load().values() if status in (None, p["status"])]


EQUITY_LOG = PROPOSALS_PATH.with_name("equity_log.jsonl")


def track() -> dict[str, Any]:
    """Log today's equity and report P&L: account vs its first logged value, and each open position.

    Append one line per call to ``equity_log.jsonl`` so performance can be judged over weeks.
    """
    acct = account()
    row = {"ts": int(time.time()), "equity": float(acct["equity"]), "cash": float(acct["cash"])}
    log = [json.loads(line) for line in EQUITY_LOG.read_text().splitlines()] if EQUITY_LOG.exists() else []
    EQUITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EQUITY_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    start = log[0]["equity"] if log else row["equity"]
    return {
        "equity": row["equity"], "since_first_log": round(row["equity"] / start - 1, 4) if start else 0.0,
        "logged_days": len(log) + 1,
        "positions": [{"symbol": p["symbol"], "unrealized_plpc": p["unrealized_plpc"], "market_value": p["market_value"]}
                      for p in positions()],
    }
