from tradingagents.agents.schemas import PortfolioDecisionLite, render_pm_decision_lite
from tradingagents.graph.setup import GraphSetup
from tradingagents.reporting import extract_screen_summary


def test_lite_render_is_parsed_without_write_up():
    text = render_pm_decision_lite(
        PortfolioDecisionLite(rating="Buy", confidence=0.7, price_target=12.5, time_horizon="5-10 days"))
    assert "Thesis" not in text and "Summary" not in text
    s = extract_screen_summary({"final_trade_decision": text})
    assert (s["direction"], s["confidence"], s["price_target"], s["time_horizon"]) == ("Buy", 0.7, "12.5", "5-10 days")


def test_decision_only_graph_skips_the_research_stage():
    from tradingagents.graph.conditional_logic import ConditionalLogic

    class Stub:
        def bind_tools(self, *a, **k): return self
        def with_structured_output(self, *a, **k): return self

    g = GraphSetup(Stub(), Stub(), ConditionalLogic(), 5, decision_only=True).setup_graph(("market", "news"))
    assert not {"Bull Researcher", "Bear Researcher", "Research Manager"} & set(g.nodes)
    assert {"Trader", "Aggressive Analyst", "Portfolio Manager"} <= set(g.nodes)
