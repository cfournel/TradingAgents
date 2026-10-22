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

Above ``RISK_BAND_MIN_BALANCE`` ($1000 of real equity) the sizing uses the real account instead of the
simulated capital, and the risk per trade is a band (``RISK_BAND``) picked by conviction: Overweight
risks the low end, Buy the high end (scaled by the analysis confidence when it gives one). A cash
reserve is kept untouched.

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

from tradingagents.notify import notify
from tradingagents.dataflows.alpaca_screener import DATA_URL, TRADING_URL, _credentials, _get

PROPOSALS_PATH = Path.home() / ".hermes" / "tradingagents" / "proposals.json"
BUYABLE = {"buy", "overweight"}
RISK_BAND_MIN_BALANCE = 1000.0
RISK_BAND = (0.005, 0.02)  # fraction of equity risked per trade: Overweight / low confidence -> Buy / high
CASH_RESERVE = 0.10


def risk_for(decision: str, confidence: float | None = None) -> float:
    """Risk per trade inside ``RISK_BAND``: Overweight at the low end, Buy at the high end, scaled by confidence."""
    lo, hi = RISK_BAND
    top = hi if decision.lower() == "buy" else (lo + hi) / 2
    if confidence is None:
        return top
    return lo + (top - lo) * min(max(confidence, 0.0), 1.0)


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
                  risk_per_trade: float = 0.02, max_position_pct: float = 0.25,
                  analysis_stop: float | None = None) -> dict[str, float] | None:
    """Notional and stop for one trade, or None when it cannot be sized.

    ``analysis_stop`` (the trader agent's stop level) replaces the ATR stop when it is a sane distance
    below ``price`` (3% to 20%).
    """
    if price <= 0 or capital <= 0:
        return None
    k = 2.0 if horizon == "swing" else 3.0
    stop_pct = max(k * atr_pct, 0.05)
    if analysis_stop and 0.03 <= 1 - analysis_stop / price <= 0.20:
        stop_pct = 1 - analysis_stop / price
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
    equity = float(acct["equity"])
    banded = equity > RISK_BAND_MIN_BALANCE
    if banded:  # real balance large enough: size on it, not on the simulated capital
        capital = equity
        bp = min(float(acct["buying_power"]), float(acct["cash"]) - CASH_RESERVE * equity)
    else:
        capital = capital if capital is not None else equity
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
        # The trader's entry level is a pullback when it sits clearly below the market; wait for it.
        entry = r.get("entry_price")
        pullback = entry if isinstance(entry, (int, float)) and 0.8 * price <= entry <= 0.995 * price else None
        risk = risk_for(decision, r.get("confidence")) if banded else risk_per_trade
        sizing = size_position(pullback or price, metrics.get("atr_pct", 0.04), r.get("horizon") or "swing",
                               capital, min(bp, capital), risk, max_position_pct, r.get("stop_loss"))
        if not sizing:
            continue
        pid = uuid.uuid4().hex[:8]
        stored[pid] = {
            "id": pid, "ticker": ticker, "side": "buy", "status": "proposed", "created_at": int(time.time()),
            "decision": r.get("decision"), "horizon": r.get("horizon"), "entry_price": price,
            "price_target": r.get("price_target"), "time_horizon": r.get("time_horizon"),
            "paper": _paper(), "risk_per_trade": risk, **sizing,
        }
        if pullback:
            stored[pid]["entry_limit"] = pullback
        out.append(stored[pid])
        bp -= sizing["notional"]
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


def _target(p: dict) -> float | None:
    """The proposal's price target as a number, only when it is a plausible gain over the entry."""
    try:
        t = float(str(p.get("price_target")).replace("$", "").replace(",", ""))
    except ValueError:
        return None
    return t if t > p["entry_price"] * 1.02 else None


def check_stops(close: bool = False) -> list[dict]:
    """Submitted proposals whose position broke the stop or reached the target.

    Each hit carries ``reason`` ("stop" or "target"); positions are closed only when ``close`` is true.
    """
    if close:
        _require_paper()
    stored = _load()
    pos = {p["symbol"]: p for p in positions()}
    hits = []
    for p in stored.values():
        cur = pos.get(p["ticker"])
        if p["status"] != "submitted" or not cur:
            continue
        price = float(cur["current_price"])
        target = _target(p)
        if price <= p["stop_price"]:
            reason = "stop"
        elif target and price >= target:
            reason = "target"
        else:
            continue
        hit = {**p, "current_price": price, "reason": reason}
        if close:
            r = requests.delete(f"{_base()}/v2/positions/{p['ticker']}", headers=_credentials(), timeout=30)
            hit["closed"] = r.status_code < 300
            if hit["closed"]:
                p["status"] = "stopped" if reason == "stop" else "target_hit"
                level = p["stop_price"] if reason == "stop" else target
                notify(f"[trading] {'STOP' if reason == 'stop' else 'OBJECTIF'} {p['ticker']}",
                       f"Position {p['ticker']} fermée (paper) : prix {price}, "
                       f"{'stop' if reason == 'stop' else 'objectif'} {level}.")
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


def _require_paper() -> None:
    if not _paper():
        raise RuntimeError("automatic execution is allowed in paper mode only (ALPACA_PAPER_TRADE=true)")


def market_open() -> bool:
    return bool(_get(f"{_base()}/v2/clock", _credentials()).get("is_open"))


def set_entry_limit(proposal_id: str, price: float) -> dict:
    """Wait for a pullback: the proposal is only sent once the price is at or below ``price``."""
    stored = _load()
    p = stored[proposal_id]
    if p["status"] != "proposed":
        raise ValueError(f"proposal {proposal_id} is {p['status']}, not 'proposed'")
    p["entry_limit"] = float(price)
    _save(stored)
    return p


def fill_entries(max_age_days: float = 10) -> list[dict]:
    """Paper only: send proposals whose entry level is reached (or that have no entry limit).

    Does nothing while the market is closed. Proposals older than ``max_age_days`` expire.
    """
    _require_paper()
    if not market_open():
        return []
    now, sent = time.time(), []
    for p in list(_load().values()):
        if p["status"] != "proposed":
            continue
        if now - p["created_at"] > max_age_days * 86400:
            stored = _load()
            stored[p["id"]]["status"] = "expired"
            _save(stored)
            continue
        limit = p.get("entry_limit")
        if limit is not None:
            price = _latest(p["ticker"])
            if not price or price > limit:
                continue
        done = execute(p["id"])
        sent.append(done)
        notify(f"[trading] ACHAT {done['ticker']}",
               f"Ordre paper envoyé : {done['notional']} $ de {done['ticker']} (décision {done.get('decision')}, "
               f"stop {done['stop_price']}, objectif {done.get('price_target')}).")
    return sent
