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
    ap.add_argument("--analyse", type=int, default=5, help="how many fresh candidates to analyse")
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
            say(f"stop hit {hit['ticker']} @ {hit['current_price']} (stop {hit['stop_price']}) closed={hit.get('closed')}")
    # 2. Fresh candidates.
    history = json.loads(HISTORY.read_text()) if HISTORY.exists() else {}
    held = {p["symbol"] for p in ao.positions()}
    busy = {p["ticker"] for p in ao.list_proposals() if p["status"] in ("proposed", "submitted")}
    recent = {t for t, d in history.items() if time.time() - d < RECENT_DAYS * 86400}
    screen = run("scripts/screen_candidates.py", "--asset-classes", "stock", "--risk", a.risk, "--horizon", "swing",
                 "--limit", "20", "--min-price", a.min_price, "--max-price", a.max_price, timeout=900)
    picks = [c for c in screen["candidates"] if c["ticker"] not in held | busy | recent][: a.analyse]
    say("candidates: " + ", ".join(c["ticker"] for c in picks))
    if not picks:
        say("nothing new to analyse")
    else:
        # 3. Analyse (slow: ~10 min per ticker on the local model).
        tickers = ",".join(c["ticker"] for c in picks)
        analysis = run("scripts/batch_analyze.py", "--tickers", tickers, "--horizon", "swing", "--quick",
                       "--date", date.today().isoformat(), timeout=6 * 3600)
        for r in analysis["results"]:
            history[r["ticker"]] = time.time()
            say(f"analysis {r['ticker']}: {r['decision']}" + (f" ({r['error']})" if r.get("error") else ""))
        HISTORY.write_text(json.dumps(history))
        (OUT / f"{stamp}-analysis.json").write_text(json.dumps(analysis))
        # 4. Proposals, then paper-only execution.
        proposals = ao.propose(analysis["results"], {c["ticker"]: c for c in screen["candidates"]}, a.capital)
        for p in proposals:
            say(f"proposal {p['id']} {p['ticker']} notional={p['notional']} stop={p['stop_price']}")
    if paper:
        for p in ao.fill_entries():
            say(f"EXECUTED (paper) {p['id']} {p['ticker']} notional={p['notional']} order={p.get('order_id')}")
    else:
        say("live mode: proposals left for manual confirmation")
    # 5. Performance log.
    say("track: " + json.dumps(ao.track()))
    (OUT / f"{stamp}.log").write_text("\n".join(log) + "\n")
    notify(f"[trading] cycle du {stamp}", "\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
