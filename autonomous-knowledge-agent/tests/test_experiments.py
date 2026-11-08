"""A/B testing framework: assignment, LLM routing strategy, reporting and offline evaluation."""
from langchain_core.messages import HumanMessage

from agentic import evaluation
from agentic.agents.supervisor import ACCOUNT, ESCALATION, KNOWLEDGE, Supervisor, compute_priority, llm_route
from agentic.experiments import VARIANTS, ab_report, assign_variant, strategy_setting
from agentic.workflow import build_graph
from tests.fake_llm import ScriptedLLM, broken_brain


def test_assignment_is_sticky_and_balanced():
    ids = [f"ticket-{i}" for i in range(400)]
    variants = [assign_variant(t, "ab") for t in ids]
    assert variants == [assign_variant(t, "ab") for t in ids]          # deterministic per ticket
    assert 0.4 < variants.count("rules") / len(ids) < 0.6              # ~50/50
    assert assign_variant("x", "llm") == "llm" and assign_variant("x", "unknown") == "rules"


def test_strategy_setting_from_config_or_env(monkeypatch):
    monkeypatch.delenv("UDAHUB_ROUTING_STRATEGY", raising=False)
    assert strategy_setting(None) == "rules"
    monkeypatch.setenv("UDAHUB_ROUTING_STRATEGY", "ab")
    assert strategy_setting({}) == "ab"
    assert strategy_setting({"configurable": {"routing_strategy": "llm"}}) == "llm"


def test_llm_strategy_still_obeys_hard_policies(scripted_llm):
    state = {"classification": {"category": "human_request"}, "messages": [HumanMessage("a human please")]}
    assert llm_route(state, lambda: scripted_llm)[0] == ESCALATION
    assert scripted_llm.calls == []  # policy decided, LLM not consulted
    state = {"classification": {"category": "complaint", "sentiment": "negative", "urgency": "high"},
             "messages": [HumanMessage("awful")]}
    assert llm_route(state, lambda: scripted_llm)[0] == KNOWLEDGE  # scripted router never escalates
    state["classification"]["category"] = "reservation"
    assert llm_route(state, lambda: scripted_llm)[0] == ACCOUNT


def test_llm_router_failure_falls_back_to_rules():
    sup = Supervisor(lambda: ScriptedLLM(handler=broken_brain, calls=[]))
    out = sup({"ticket_id": "t", "classification": {"category": "reservation"}, "messages": [HumanMessage("book")]},
              {"configurable": {"routing_strategy": "llm"}})
    assert out["route"] == ACCOUNT and out["trace"][0]["strategy_used"].startswith("rules (llm router failed")


def test_priority_from_sentiment_urgency_and_tier():
    p1 = compute_priority({"classification": {"urgency": "high", "sentiment": "negative", "category": "billing_refund"},
                           "customer_context": {"subscription": {"tier": "premium"}}})
    p4 = compute_priority({"classification": {"urgency": "low", "sentiment": "positive", "category": "general_question"}})
    assert p1[0] == "P1" and "negative sentiment" in p1[2]
    assert p4[0] == "P4"
    # platform urgency adds to the (already floored to high) classifier urgency
    assert compute_priority({"classification": {"urgency": "high"}, "ticket_metadata": {"urgency": "high"}})[0] == "P2"
    assert compute_priority({"classification": {"urgency": "medium"}})[0] == "P3"


def test_ab_report_from_logged_events(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    for i in range(12):
        graph.invoke({"messages": [HumanMessage("Which reservations do I have?")], "external_user_id": "f556c0"},
                     config={"configurable": {"thread_id": f"ab-{i}", "routing_strategy": "ab"}})
    report = ab_report()
    assert set(report) == set(VARIANTS)
    assert sum(report[v]["tickets"] for v in VARIANTS) == 12
    for v in VARIANTS:
        if report[v]["tickets"]:
            assert report[v]["resolution_rate"] == 1.0 and report[v]["route_mix"] == {"account_agent": report[v]["tickets"]}


def test_offline_evaluation_harness(dbs, scripted_llm):
    tickets = evaluation.load_dataset()
    assert len(tickets) >= 20 and all(t["acceptable_routes"] and t["acceptable_statuses"] for t in tickets)
    result = evaluation.evaluate(("rules", "llm"), tickets[:6], llm=scripted_llm)
    assert set(result["summary"]) == {"rules", "llm"}
    assert result["summary"]["rules"]["tickets"] == 6 and result["summary"]["rules"]["errors"] == 0
    report = evaluation.to_markdown(result)
    assert "| routing_accuracy |" in report and "Winner on this set" in report
    # the sandbox never touches the configured databases
    from agentic.tools import udahub_tools
    assert udahub_tools.get_ticket("eval-rules-T01")["error"] == "ticket_not_found"
