import pytest

from tradingagents.dataflows import alpaca_orders as ao


@pytest.mark.unit
def test_sizing_risks_two_percent_of_capital():
    s = ao.size_position(price=50, atr_pct=0.05, horizon="swing", capital=1000, buying_power=1000)
    assert s["stop_pct"] == 0.10 and s["notional"] == 200.0 and s["risk_amount"] == 20.0
    assert s["stop_price"] == 45.0


@pytest.mark.unit
def test_sizing_caps_position_and_buying_power():
    assert ao.size_position(10, 0.01, "swing", 1000, 1000)["notional"] == 250.0   # 25% cap, 5% min stop
    assert ao.size_position(10, 0.01, "swing", 1000, 40)["notional"] == 40.0


@pytest.mark.unit
def test_position_horizon_uses_a_wider_stop():
    swing = ao.size_position(50, 0.04, "swing", 1000, 1000)
    position = ao.size_position(50, 0.04, "position", 1000, 1000)
    assert position["stop_pct"] > swing["stop_pct"] and position["notional"] < swing["notional"]


@pytest.mark.unit
def test_orders_below_the_one_dollar_minimum_are_dropped():
    assert ao.size_position(10, 0.05, "swing", 0.8, 0.8) is None   # less than the $1 minimum is available


@pytest.mark.unit
def test_tiny_account_rounds_up_to_the_one_dollar_minimum_within_twice_the_risk_budget():
    s = ao.size_position(21.34, 0.0529, "swing", 5, 5)    # exact size $0.94, risk at $1 = $0.106 <= $0.20
    assert s["notional"] == 1.0
    assert ao.size_position(21.34, 0.0529, "swing", 1.5, 1.5) is None   # $1 would risk 7% of capital


@pytest.mark.unit
def test_the_analysis_stop_replaces_the_atr_stop_only_when_sane():
    s = ao.size_position(100, 0.01, "swing", 1000, 1000, analysis_stop=92)
    assert s["stop_pct"] == 0.08 and s["stop_price"] == 92.0
    assert ao.size_position(100, 0.01, "swing", 1000, 1000, analysis_stop=99)["stop_pct"] == 0.05   # too tight
    assert ao.size_position(100, 0.01, "swing", 1000, 1000, analysis_stop=60)["stop_pct"] == 0.05   # too wide


@pytest.mark.unit
def test_target_must_be_a_real_gain_over_the_entry():
    assert ao._target({"price_target": "25.5", "entry_price": 20}) == 25.5
    assert ao._target({"price_target": "not provided", "entry_price": 20}) is None
    assert ao._target({"price_target": "20.1", "entry_price": 20}) is None


@pytest.mark.unit
def test_a_pullback_entry_becomes_a_limit_and_a_target_hit_closes_the_position(monkeypatch, tmp_path):
    monkeypatch.setattr(ao, "PROPOSALS_PATH", tmp_path / "proposals.json")
    monkeypatch.setattr(ao, "account", lambda: {"equity": "5", "buying_power": "5"})
    monkeypatch.setattr(ao, "positions", lambda: [])
    monkeypatch.setattr(ao, "_latest", lambda t: 21.0)
    monkeypatch.setattr(ao, "notify", lambda *a, **k: [])
    r = {"ticker": "ABC", "decision": "Overweight", "horizon": "swing", "price_target": "26",
         "entry_price": 20.0, "stop_loss": 18.5}
    (p,) = ao.propose([r], {}, 5)
    assert p["entry_limit"] == 20.0 and p["stop_price"] == 18.5

    stored = ao._load()
    stored[p["id"]]["status"] = "submitted"
    ao._save(stored)
    monkeypatch.setattr(ao, "positions", lambda: [{"symbol": "ABC", "current_price": "26.5"}])
    closed = []
    monkeypatch.setattr(ao.requests, "delete", lambda *a, **k: closed.append(a) or type("R", (), {"status_code": 200})())
    monkeypatch.setattr(ao, "_require_paper", lambda: None)
    monkeypatch.setattr(ao, "_credentials", lambda: {})
    (hit,) = ao.check_stops(close=True)
    assert hit["reason"] == "target" and closed
    assert ao._load()[p["id"]]["status"] == "target_hit"
