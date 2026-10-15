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
