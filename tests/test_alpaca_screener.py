import math

import pytest

from tradingagents.dataflows import alpaca_screener as a


def _bars(n, start=100.0, drift=0.001, vol=1_000_000):
    out, p = [], start
    for i in range(n):
        p *= 1 + drift + (0.01 if i % 2 else -0.01)
        out.append({"c": p, "h": p * 1.01, "l": p * 0.99, "v": vol})
    return out


@pytest.mark.unit
def test_metrics_need_enough_history():
    assert a._metrics(_bars(100)) is None


@pytest.mark.unit
def test_metrics_measure_volatility_and_trend():
    m = a._metrics(_bars(260))
    assert 0.1 < m["vol_60d"] < 0.5          # +-1% daily swings ~ 16% annualised
    assert m["avg_dollar_vol_20d"] > 1e7
    assert math.isclose(m["pct_of_52w_high"], m["price"] / max(b["h"] for b in _bars(260)[-252:]))


@pytest.mark.unit
def test_percentile_ranks_are_ordered():
    assert a._percentile_ranks([3.0, 1.0, 2.0]) == [1.0, 0.0, 0.5]


@pytest.mark.unit
def test_rejects_unknown_risk():
    with pytest.raises(ValueError):
        a.screen_alpaca(risk="reckless")
