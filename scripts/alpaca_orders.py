#!/usr/bin/env python3
"""Order proposals + manual execution on Alpaca (paper by default). Prints one JSON object.

    propose  --analysis FILE [--screen FILE] [--capital N]   size orders from batch_analyze output (sends nothing)
    list     [--status proposed|submitted|stopped]
    account                                                   equity, cash, open positions
    execute  ID                                               send ONE stored proposal (explicit confirmation)
    entry-limit ID PRICE                                      wait for a pullback before sending that proposal
    fill                                                      paper only: send proposals whose entry level is reached
    track                                                     log equity, print P&L since the first log
    stops    [--close]                                        report (or close) positions that broke their stop
"""

from __future__ import annotations

import argparse
import json
import sys

from tradingagents.dataflows import alpaca_orders as ao


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--analysis", required=True, help="JSON from batch_analyze.py ({'results': [...]})")
    p.add_argument("--screen", help="JSON from screen_candidates.py, for ATR-based stops")
    p.add_argument("--capital", type=float, help="Capital to size against (default: account equity)")
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--risk-per-trade", type=float, default=0.02)
    p.add_argument("--max-position-pct", type=float, default=0.25)
    ls = sub.add_parser("list")
    ls.add_argument("--status")
    sub.add_parser("account")
    ex = sub.add_parser("execute")
    ex.add_argument("id")
    el = sub.add_parser("entry-limit")
    el.add_argument("id")
    el.add_argument("price", type=float)
    sub.add_parser("fill")
    sub.add_parser("track")
    st = sub.add_parser("stops")
    st.add_argument("--close", action="store_true")
    a = ap.parse_args()

    try:
        if a.cmd == "propose":
            results = json.load(open(a.analysis))["results"]
            cands = {}
            if a.screen:
                cands = {c["ticker"]: c for c in json.load(open(a.screen)).get("candidates", [])}
            out = ao.propose(results, cands, a.capital, a.max_positions, a.risk_per_trade, a.max_position_pct)
        elif a.cmd == "list":
            out = ao.list_proposals(a.status)
        elif a.cmd == "account":
            acct = ao.account()
            out = {"paper": ao._paper(), "equity": acct["equity"], "cash": acct["cash"],
                   "buying_power": acct["buying_power"], "positions": [
                       {k: p[k] for k in ("symbol", "qty", "avg_entry_price", "current_price", "unrealized_plpc")}
                       for p in ao.positions()]}
        elif a.cmd == "execute":
            out = ao.execute(a.id)
        elif a.cmd == "entry-limit":
            out = ao.set_entry_limit(a.id, a.price)
        elif a.cmd == "fill":
            out = ao.fill_entries()
        elif a.cmd == "track":
            out = ao.track()
        else:
            out = ao.check_stops(a.close)
    except Exception as exc:  # noqa: BLE001 — report as JSON, not a traceback
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(out, default=str, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
