"""Unit tests for individual agents and routing decisions."""
import inspect

import pytest
from langchain_core.messages import HumanMessage

from agentic.agents.classifier import ClassifierAgent, heuristic_classify
from agentic.agents.knowledge import KnowledgeAgent
from agentic.agents.supervisor import ACCOUNT, ESCALATION, KNOWLEDGE, decide_route
from agentic.tools.agent_tools import build_account_tools
from tests.fake_llm import ScriptedLLM, _call, broken_brain


@pytest.mark.parametrize(
    "text,category",
    [
        ("I can't log in, password reset doesn't work", "login_access"),
        ("My account is blocked!", "account_status"),
        ("Please book Samba Night for me", "reservation"),
        ("How do I pause my subscription?", "subscription"),
        ("I was charged twice, I need a refund", "billing_refund"),
        ("The app keeps crashing when I open the QR code", "technical_issue"),
        ("I want to speak to a human", "human_request"),
        ("Thanks, that's all!", "closing"),
        ("Tell me something", "general_question"),
    ],
)
def test_heuristic_classifier(text, category):
    assert heuristic_classify(text).category == category


def test_heuristic_urgency_and_sentiment():
    cls = heuristic_classify("This is unacceptable, my event is tonight!")
    assert cls.urgency == "high" and cls.sentiment == "angry"
    mild = heuristic_classify("My QR code is missing in the app")
    assert mild.sentiment == "negative" and mild.urgency == "medium"


def test_classifier_uses_llm_structured_output():
    llm = ScriptedLLM(handler=lambda m, t: _call("TicketClassification", {
        "category": "experience_info", "urgency": "low", "sentiment": "positive", "confidence": 0.9,
        "summary": "Wants concerts in Rio", "tags": ["events"], "customer_preferences": ["likes concerts"]}), calls=[])
    out = ClassifierAgent(lambda: llm)({"messages": [HumanMessage("Any concerts in Rio?")], "ticket_id": "t"})
    assert out["classification"]["category"] == "experience_info"
    assert out["classification"]["customer_preferences"] == ["likes concerts"]
    assert out["trace"][0]["source"] == "llm"


def test_classifier_human_request_guardrail_overrides_llm():
    llm = ScriptedLLM(handler=lambda m, t: _call("TicketClassification", {
        "category": "general_question", "urgency": "low", "sentiment": "neutral", "confidence": 0.9,
        "summary": "x"}), calls=[])
    out = ClassifierAgent(lambda: llm)({"messages": [HumanMessage("let me talk to a real person")]})
    assert out["classification"]["category"] == "human_request"


def test_classifier_falls_back_when_llm_fails():
    out = ClassifierAgent(lambda: ScriptedLLM(handler=broken_brain, calls=[]))(
        {"messages": [HumanMessage("I need a refund")]}
    )
    assert out["classification"]["category"] == "billing_refund"
    assert out["trace"][0]["source"].startswith("heuristic")


@pytest.mark.parametrize(
    "classification,extra,expected",
    [
        ({"category": "billing_refund"}, {}, ESCALATION),
        ({"category": "human_request"}, {}, ESCALATION),
        ({"category": "complaint", "sentiment": "negative"}, {}, ESCALATION),
        ({"category": "reservation"}, {}, ACCOUNT),
        ({"category": "subscription"}, {}, ACCOUNT),
        ({"category": "experience_info"}, {}, ACCOUNT),
        ({"category": "login_access"}, {"customer_context": {"is_blocked": True}}, ESCALATION),
        ({"category": "reservation"}, {"customer_context": {"is_blocked": True}}, ACCOUNT),
        ({"category": "login_access"}, {"customer_context": {"is_blocked": False}}, KNOWLEDGE),
        ({"category": "technical_issue"}, {}, KNOWLEDGE),
        ({"category": "general_question", "confidence": 0.1}, {}, KNOWLEDGE),
        ({"category": "reservation"}, {"ticket_status": "escalated"}, ESCALATION),
        ({"category": "billing_refund"}, {"external_user_id": "88382b"}, ACCOUNT),
        ({"category": "general_question"}, {"awaiting_confirmation": {"tool": "pause_my_subscription"}}, ACCOUNT),
        ({"category": "human_request"}, {"awaiting_confirmation": {"tool": "pause_my_subscription"}}, ESCALATION),
        ({"category": "technical_issue", "sentiment": "angry", "urgency": "high"}, {}, ESCALATION),
        ({"category": "technical_issue", "sentiment": "negative", "urgency": "high"}, {}, KNOWLEDGE),  # mild: AI tries
        ({"category": "technical_issue", "sentiment": "angry", "urgency": "medium"}, {}, KNOWLEDGE),
        ({"category": "subscription"}, {"turn": 6}, ESCALATION),
        ({"category": "closing"}, {"turn": 6}, KNOWLEDGE),
    ],
)
def test_supervisor_routing(classification, extra, expected):
    route, reason = decide_route({"classification": classification, **extra})
    assert route == expected and reason


def test_account_tools_are_bound_to_the_customer():
    """The LLM can never pass a user_id: tools are scoped to the ticket's customer."""
    tools = build_account_tools("cultpass", "f556c0")
    assert {t.name for t in tools} >= {"get_my_account", "reserve_experience", "cancel_reservation", "escalate_to_human"}
    for tool in tools:
        assert "user_id" not in tool.args, tool.name


def test_account_tools_for_guest(dbs):
    tools = {t.name: t for t in build_account_tools("cultpass", None)}
    assert tools["get_my_account"].invoke({})["error"] == "no_customer"
    assert tools["search_experiences"].invoke({"query": "samba"})["ok"]


def test_knowledge_agent_escalates_when_nothing_relevant(dbs, scripted_llm):
    state = {
        "ticket_id": "t", "account_id": "cultpass",
        "messages": [HumanMessage("What is the capital of France?")],
        "classification": {"category": "general_question", "summary": "capital of France"},
    }
    out = KnowledgeAgent(lambda: scripted_llm)(state)
    assert "No relevant knowledge-base article" in out["escalation_reason"]
    assert scripted_llm.calls == []  # never asks the LLM to guess


def test_knowledge_agent_answers_with_grounding(dbs, scripted_llm):
    state = {
        "ticket_id": "t", "account_id": "cultpass",
        "messages": [HumanMessage("How do I reserve a spot for an event?")],
        "classification": {"category": "general_question", "summary": "how to reserve"},
    }
    out = KnowledgeAgent(lambda: scripted_llm)(state)
    assert out["resolution"]["status"] == "resolved"
    assert "Reserve" in out["messages"][0].content
    system_prompt = scripted_llm.calls[0]["messages"][0].content
    assert "### How to Reserve a Spot for an Event" in system_prompt


def test_classifier_anger_guardrail_overrides_llm():
    llm = ScriptedLLM(handler=lambda m, t: _call("TicketClassification", {
        "category": "technical_issue", "urgency": "high", "sentiment": "negative", "confidence": 0.9,
        "summary": "app crashed"}), calls=[])
    out = ClassifierAgent(lambda: llm)({"messages": [HumanMessage("The app crashed and I missed my event, ridiculous!!")]})
    assert out["classification"]["sentiment"] == "angry"
    assert "anger_guardrail" in out["trace"][0]["source"]


def test_account_agent_prompt_is_grounded_in_relevant_articles(dbs, scripted_llm):
    from agentic.agents.account import AccountAgent

    state = {
        "ticket_id": "t", "account_id": "cultpass", "external_user_id": "f556c0",
        "messages": [HumanMessage("Can I bring a guest to Samba Night?")],
        "classification": {"category": "reservation", "summary": "bring a guest to an event"},
    }
    out = AccountAgent(lambda: scripted_llm)(state)
    system_prompt = scripted_llm.calls[0]["messages"][0].content
    assert "### Bringing a Guest to an Experience" in system_prompt
    assert out["retrieved_articles"][0]["title"] == "Bringing a Guest to an Experience"


def test_unknown_customer_is_not_called_there(dbs, scripted_llm):
    from agentic.agents.common import addressee

    assert addressee({}) == "the customer without a name (name unknown)"
    assert addressee({"customer_context": {"full_name": "Bob Stone"}}) == "the customer by first name (Bob)"
