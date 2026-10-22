#!/usr/bin/env python3
"""Daily paper-trading cycle: screen -> analyse -> propose -> (paper only) execute -> track.

Runs unattended from a systemd timer. Automatic execution is refused unless
``ALPACA_PAPER_TRADE`` is true; in live mode the cycle stops at proposals.

    daily_cycle.py [--capital 5] [--analyse 5] [--risk medium] [--min-price 2] [--max-price 50]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

from tradingagents.dataflows import alpaca_orders as ao
from tradingagents.notify import notify

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
OUT = Path.home() / ".hermes" / "tradingagents" / "cycles"
HISTORY = Path.home() / ".hermes" / "tradingagents" / "cycle_history.json"
RECENT_DAYS = 5


def run(*args: str, timeout: int) -> dict:
    proc = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(proc.stdout[proc.stdout.index("{"):])
    except (ValueError, json.JSONDecodeError):
        raise RuntimeError(f"{args[0]} failed ({proc.returncode}): {(proc.stderr or proc.stdout)[-400:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capital", type=float, default=5.0, help="capital to size against (mimics the real account)")
    ap.add_argument("--hand", type=int, default=5, help="buy-rated names to keep in hand (positions + waiting proposals)")
    ap.add_argument("--max-analyses", type=int, default=15, help="cap on analyses per cycle")
    ap.add_argument("--budget-hours", type=float, default=4.0, help="stop starting new analyses after this long")
    ap.add_argument("--full", action="store_true", help="full multi-agent analysis instead of the decision-only run")
    ap.add_argument("--risk", default="medium", choices=["low", "medium", "high", "any"])
    ap.add_argument("--min-price", default="2")
    ap.add_argument("--max-price", default="50")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    log = []

    def say(msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        log.append(line)

    paper = ao._paper()
    say(f"cycle start, paper={paper}, capital={a.capital}")
    # 1. Stops first: close what broke its level.
    if paper:
        for hit in ao.check_stops(close=True):
            say(f"{hit['reason']} hit {hit['ticker']} @ {hit['current_price']} closed={hit.get('closed')}")
    # 2. Keep ``--hand`` buy-rated names in hand (open positions + proposals waiting for their entry):
    #    analyse one fresh candidate at a time, set the rejected ones aside, until the hand is full.
    history = json.loads(HISTORY.read_text()) if HISTORY.exists() else {}
    held = {p["symbol"] for p in ao.positions()}

    def in_hand() -> int:
        waiting = {p["ticker"] for p in ao.list_proposals("proposed")}
        return len(held | waiting)

    say(f"in hand: {in_hand()}/{a.hand}")
    deadline = time.time() + a.budget_hours * 3600
    analysed, tried = 0, set()
    for risk in (a.risk, "any"):
        if in_hand() >= a.hand or analysed >= a.max_analyses or time.time() > deadline:
            break
        screen = run("scripts/screen_candidates.py", "--asset-classes", "stock", "--risk", risk, "--horizon", "swing",
                     "--limit", "40", "--min-price", a.min_price, "--max-price", a.max_price, timeout=900)
        by_ticker = {c["ticker"]: c for c in screen["candidates"]}
        recent = {t for t, d in history.items() if time.time() - d < RECENT_DAYS * 86400}
        busy = {p["ticker"] for p in ao.list_proposals() if p["status"] in ("proposed", "submitted")}
        picks = [t for t in by_ticker if t not in held | busy | recent | tried]
        say(f"screen risk={risk}: {len(picks)} fresh candidates")
        for ticker in picks:
            if in_hand() >= a.hand or analysed >= a.max_analyses or time.time() > deadline:
                break
            tried.add(ticker)
            analysed += 1
            # 3. Decide: decision-only run by default; --full runs the whole agent graph (~10 min per ticker).
            try:
                analysis = run("scripts/batch_analyze.py", "--tickers", ticker, "--horizon", "swing",
                               "--quick" if a.full else "--decision-only", "--date", date.today().isoformat(), timeout=2 * 3600)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                history[ticker] = time.time()
                say(f"analysis {ticker} failed: {exc}")
                continue
            r = analysis["results"][0]
            history[ticker] = time.time()
            HISTORY.write_text(json.dumps(history))
            (OUT / f"{stamp}-{ticker}.json").write_text(json.dumps(analysis))
            say(f"analysis {ticker}: {r.get('decision')}" + (f" ({r['error']})" if r.get("error") else ""))
            # 4. Proposal if buy-rated; otherwise the name is set aside and the loop moves on.
            for p in ao.propose([r], by_ticker, a.capital):
                waits = f" waits for {p['entry_limit']}" if p.get("entry_limit") else ""
                say(f"proposal {p['id']} {p['ticker']} notional={p['notional']} stop={p['stop_price']}{waits}")
            if paper:
                for p in ao.fill_entries():
                    say(f"EXECUTED (paper) {p['id']} {p['ticker']} notional={p['notional']} order={p.get('order_id')}")
                held = {p["symbol"] for p in ao.positions()}
    say(f"hand after cycle: {in_hand()}/{a.hand}, analysed {analysed}")
    if not paper:
        say("live mode: proposals left for manual confirmation")
    # 5. Performance log.
    say("track: " + json.dumps(ao.track()))
    (OUT / f"{stamp}.log").write_text("\n".join(log) + "\n")
    notify(f"[trading] cycle du {stamp}", "\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
