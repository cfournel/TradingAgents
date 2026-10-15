"""Candidate screener over the Alpaca-tradable US equity universe (no LLM).

Finds stocks matching the user's criteria (price band, risk, potential) so the
slow multi-agent analysis is only spent on candidates worth it. Everything is
computed locally from Alpaca market data:

  stage 0  Alpaca assets: active, tradable, fractionable, listed on a real
           exchange (no OTC), plain tickers (no warrants/rights/units).
  stage 1  IEX snapshots (free plan): price band + a rough activity floor.
  stage 2  consolidated (SIP) daily bars for the survivors: liquidity, risk
           (annualised volatility, ATR%), trend and momentum factors.

Risk is realised volatility, not beta: low < 30%, medium 30-60%, high > 60%
annualised (60 sessions). Potential is a percentile-rank blend of momentum,
relative strength against SPY, trend and proximity to the 52-week high, with
a penalty for overbought / over-extended names (chasing is not potential).

Credentials: ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` from the environment,
falling back to the env file at ``ALPACA_ENV_FILE`` (default
``~/.hermes/secrets/alpaca.env``). Read-only: this module never places orders.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

TRADING_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
VALID_RISK = {"low", "medium", "high", "any"}
VALID_HORIZONS = {"swing", "position"}
EXCHANGES = {"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS"}
_PLAIN_TICKER = re.compile(r"^[A-Z]{1,5}$")
_JUNK_NAME = re.compile(r"\b(WARRANT|RIGHTS?|UNITS?|NOTES?|PREFERRED|DEPOSITARY|ACQUISITION)\b", re.I)
_FUND_NAME = re.compile(r"\b(ETF|ETN|FUND|TRUST|INDEX|ISHARES|SPDR|PROSHARES|DIREXION)\b", re.I)
_RISK_VOL = {"low": (0.0, 0.30), "medium": (0.30, 0.60), "high": (0.60, 10.0), "any": (0.0, 10.0)}
_ASSET_CACHE = Path.home() / ".cache" / "tradingagents" / "alpaca_assets.json"


def _credentials() -> dict[str, str]:
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not (key and secret):
        env_file = Path(os.environ.get("ALPACA_ENV_FILE", "~/.hermes/secrets/alpaca.env")).expanduser()
        if env_file.exists():
            vals = dict(
                line.strip().split("=", 1) for line in env_file.read_text().splitlines()
                if "=" in line and not line.lstrip().startswith("#")
            )
            key, secret = vals.get("ALPACA_API_KEY"), vals.get("ALPACA_SECRET_KEY")
    if not (key and secret):
        raise RuntimeError("Alpaca credentials missing (ALPACA_API_KEY / ALPACA_SECRET_KEY).")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _get(url: str, headers: dict, params: dict | None = None) -> Any:
    for attempt in range(4):
        r = requests.get(url, headers=headers, params=params, timeout=40)
        if r.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def load_universe(headers: dict, include_funds: bool = False, max_age_h: float = 24) -> list[dict]:
    """Tradable, fractionable, exchange-listed plain tickers (cached for a day)."""
    if _ASSET_CACHE.exists() and time.time() - _ASSET_CACHE.stat().st_mtime < max_age_h * 3600:
        assets = json.loads(_ASSET_CACHE.read_text())
    else:
        assets = _get(f"{TRADING_URL}/v2/assets", headers, {"status": "active", "asset_class": "us_equity"})
        _ASSET_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _ASSET_CACHE.write_text(json.dumps(assets))
    out = []
    for a in assets:
        if not (a.get("tradable") and a.get("fractionable") and a.get("exchange") in EXCHANGES):
            continue
        if not _PLAIN_TICKER.match(a["symbol"]) or _JUNK_NAME.search(a.get("name", "")):
            continue
        if not include_funds and _FUND_NAME.search(a.get("name", "")):
            continue
        out.append(a)
    return out


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _stage1(symbols: list[str], headers: dict, min_price: float, max_price: float,
            min_iex_dollar_vol: float, keep: int) -> list[str]:
    """Price band + activity floor from IEX snapshots; keep the ``keep`` most active."""
    rows = []
    for chunk in _chunks(symbols, 400):
        snaps = _get(f"{DATA_URL}/v2/stocks/snapshots", headers,
                     {"symbols": ",".join(chunk), "feed": "iex"})
        for sym, s in snaps.items():
            bar = s.get("dailyBar") or s.get("prevDailyBar")
            price = (s.get("latestTrade") or {}).get("p") or (bar or {}).get("c")
            if not bar or not price or not (min_price <= price <= max_price):
                continue
            prev = s.get("prevDailyBar") or bar
            dollar_vol = max(bar["v"] * price, prev["v"] * price)
            if dollar_vol >= min_iex_dollar_vol:
                rows.append((dollar_vol, sym))
    rows.sort(reverse=True)
    return [s for _, s in rows[:keep]]


def _bars(symbols: list[str], headers: dict, start: str, end: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for chunk in _chunks(symbols, 80):
        token = None
        while True:
            params = {"symbols": ",".join(chunk), "timeframe": "1Day", "start": start, "end": end,
                      "feed": "sip", "adjustment": "all", "limit": 10000}
            if token:
                params["page_token"] = token
            data = _get(f"{DATA_URL}/v2/stocks/bars", headers, params)
            for sym, bars in (data.get("bars") or {}).items():
                out.setdefault(sym, []).extend(bars)
            token = data.get("next_page_token")
            if not token:
                break
    return out


def _sma(xs: list[float], n: int) -> float:
    return sum(xs[-n:]) / n


def _rsi(closes: list[float], n: int = 14) -> float:
    gains = losses = 0.0
    for a, b in zip(closes[-n - 1:-1], closes[-n:]):
        d = b - a
        gains += max(d, 0)
        losses += max(-d, 0)
    return 100.0 if losses == 0 else 100 - 100 / (1 + gains / losses)


def _metrics(bars: list[dict]) -> dict[str, float] | None:
    """Factors from daily bars (oldest first); None if the history is too short."""
    if len(bars) < 130:
        return None
    c = [b["c"] for b in bars]
    h = [b["h"] for b in bars]
    lo = [b["l"] for b in bars]
    v = [b["v"] for b in bars]
    rets = [math.log(b / a) for a, b in zip(c[-61:-1], c[-60:]) if a > 0 and b > 0]
    if len(rets) < 55:
        return None
    mean = sum(rets) / len(rets)
    vol = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)) * math.sqrt(252)
    atr = sum(max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(-14, 0)) / 14
    window = min(len(c), 252)
    sma200 = _sma(c, 200) if len(c) >= 200 else _sma(c, len(c))
    return {
        "price": c[-1],
        "ret_5d": c[-1] / c[-6] - 1,
        "ret_21d": c[-1] / c[-22] - 1,
        "ret_63d": c[-1] / c[-64] - 1,
        "ret_126d": c[-1] / c[-127] - 1,
        "vol_60d": vol,
        "atr_pct": atr / c[-1],
        "rsi14": _rsi(c),
        "sma50": _sma(c, 50),
        "sma200": sma200,
        "pct_of_52w_high": c[-1] / max(h[-window:]),
        "extension_vs_sma50": c[-1] / _sma(c, 50) - 1,
        "avg_dollar_vol_20d": sum(a * b for a, b in zip(c[-20:], v[-20:])) / 20,
        "vol_surge": (sum(v[-5:]) / 5) / max(sum(v[-50:]) / 50, 1),
        "max_day_move_20d": max(abs(c[i] / c[i - 1] - 1) for i in range(-20, 0)),
    }


def _percentile_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for pos, i in enumerate(order):
        ranks[i] = pos / max(len(values) - 1, 1)
    return ranks


def screen_alpaca(
    risk: str = "medium",
    horizon: str = "swing",
    limit: int = 15,
    min_price: float = 2.0,
    max_price: float = 100.0,
    min_dollar_volume: float = 5_000_000,
    include_funds: bool = False,
    prefilter_keep: int = 700,
) -> list[dict[str, Any]]:
    """Ranked candidates ``[{"ticker", "asset_type", "source", "score", "metrics", "reasons"}]``."""
    if risk not in VALID_RISK or horizon not in VALID_HORIZONS:
        raise ValueError(f"risk in {sorted(VALID_RISK)}, horizon in {sorted(VALID_HORIZONS)}")
    headers = _credentials()
    universe = load_universe(headers, include_funds)
    names = {a["symbol"]: a.get("name", "") for a in universe}
    survivors = _stage1(list(names), headers, min_price, max_price,
                        min_iex_dollar_vol=min_dollar_volume * 0.005, keep=prefilter_keep)
    logger.info("alpaca screen: %d tradable -> %d after price/activity", len(names), len(survivors))

    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=400)).strftime("%Y-%m-%d")
    end = (now - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    all_bars = _bars(survivors + ["SPY"], headers, start, end)
    spy = _metrics(all_bars.get("SPY", []))

    vol_lo, vol_hi = _RISK_VOL[risk]
    pool: list[tuple[str, dict]] = []
    for sym in survivors:
        m = _metrics(all_bars.get(sym, []))
        if not m or not (min_price <= m["price"] <= max_price):
            continue
        if m["avg_dollar_vol_20d"] < min_dollar_volume or not (vol_lo <= m["vol_60d"] < vol_hi):
            continue
        if m["max_day_move_20d"] > 0.5:  # a +-50% day in the last month: halt/pump territory
            continue
        pool.append((sym, m))
    if not pool:
        return []

    spy63 = spy["ret_63d"] if spy else 0.0
    if horizon == "swing":
        factors = {"ret_21d": 0.30, "ret_5d": 0.15, "rs63": 0.15, "trend": 0.15,
                   "pct_of_52w_high": 0.10, "vol_surge": 0.15}
    else:
        factors = {"ret_126d": 0.25, "ret_63d": 0.20, "rs63": 0.15, "trend": 0.20,
                   "pct_of_52w_high": 0.15, "vol_surge": 0.05}
    raw: dict[str, list[float]] = {f: [] for f in factors}
    for _, m in pool:
        m["rs63"] = m["ret_63d"] - spy63
        m["trend"] = (m["price"] > m["sma50"]) + (m["sma50"] > m["sma200"]) + (m["price"] > m["sma200"])
        for f in factors:
            raw[f].append(min(m[f], 3.0) if f == "vol_surge" else m[f])
    ranked = {f: _percentile_ranks(vals) for f, vals in raw.items()}

    out = []
    for i, (sym, m) in enumerate(pool):
        score = sum(w * ranked[f][i] for f, w in factors.items())
        reasons = []
        if m["rsi14"] > 78:
            score -= 0.15
            reasons.append(f"overbought (RSI {m['rsi14']:.0f})")
        if m["extension_vs_sma50"] > 0.40:
            score -= 0.15
            reasons.append(f"extended {m['extension_vs_sma50']:.0%} above its 50d average")
        if m["trend"] == 3:
            reasons.append("uptrend: price > 50d > 200d")
        if m["rs63"] > 0.10:
            reasons.append(f"beats SPY by {m['rs63']:.0%} over 3 months")
        if m["pct_of_52w_high"] > 0.95:
            reasons.append("within 5% of its 52-week high")
        if m["vol_surge"] > 1.5:
            reasons.append(f"volume x{m['vol_surge']:.1f} vs its 50d average")
        out.append({
            "ticker": sym, "asset_type": "stock", "source": "alpaca_screen",
            "score": round(score, 4), "name": names.get(sym, ""),
            "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
            "reasons": reasons,
        })
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:limit]
