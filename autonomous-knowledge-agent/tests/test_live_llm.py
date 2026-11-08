"""Integration tests against the real LLM (gpt-4o-mini by default).

Skipped automatically unless VOCAREUM_API_KEY or OPENAI_API_KEY is set
(e.g. in solution/.env). Assertions are on routing / outcome, not wording.

    pytest -m live -v
"""
import pytest
from langchain_core.messages import HumanMessage

from agentic import config
from agentic.tools import cultpass_tools, udahub_tools
from agentic.tools.retrieval import HYBRID_THRESHOLD
from tests.test_retrieval import CASES
from agentic.workflow import build_graph
from tests.seed import ALICE_TICKET

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not config.has_llm_credentials(), reason="no LLM credentials"),
]


@pytest.fixture
def graph(dbs, monkeypatch):
    monkeypatch.setenv("UDAHUB_EMBEDDINGS", "auto")  # live runs use hybrid retrieval
    return build_graph(llm=config.get_llm())


PARAPHRASES = [
    ("I can't get into my account, it says wrong credentials", "How to Handle Login Issues?"),
    ("can my partner come with me to the show", "Bringing a Guest to an Experience"),
    ("I want my money returned", "Refund Requests and Billing Disputes"),
    ("I use a wheelchair, will the venue be ok?", "Accessibility and Special Assistance"),
    ("the gallery called off the exhibition", "Event Cancelled or Changed by the Venue"),
]
OFF_TOPIC = ["What is the capital of France?", "recommend a pizza recipe", "What's the weather on Mars?",
             "Can you write me a poem about cats?", "How do I fix my car engine?", "Explain quantum physics"]


def test_live_hybrid_retrieval_calibration(dbs, monkeypatch):
    """Embeddings + TF-IDF: correct article for direct and paraphrased questions, off-topic below threshold."""
    monkeypatch.setenv("UDAHUB_EMBEDDINGS", "auto")
    assert udahub_tools.search_knowledge("x")["mode"] == "hybrid"
    for query, expected in CASES:  # direct questions: right article ranked first
        top = udahub_tools.search_knowledge(query, k=1)["results"][0]
        assert top["title"] == expected, query
        assert top["score"] >= HYBRID_THRESHOLD, query
    for query, expected in PARAPHRASES:  # paraphrases: right article among the top 3 the agent receives
        hits = udahub_tools.search_knowledge(query, k=3)["results"]
        assert expected in [h["title"] for h in hits if h["score"] >= HYBRID_THRESHOLD], query
    for query in OFF_TOPIC:
        assert udahub_tools.search_knowledge(query, k=1)["results"][0]["score"] < HYBRID_THRESHOLD, query
    # vectors are persisted: a second index build re-embeds nothing
    from agentic.tools.db import session_scope, udahub_engine
    from data.models import udahub
    with session_scope(udahub_engine()) as s:
        assert s.query(udahub.KnowledgeEmbedding).count() == 16


def run(graph, text, thread_id, **extra):
    return graph.invoke(
        {"messages": [HumanMessage(text)], **extra}, config={"configurable": {"thread_id": thread_id}}
    )


def test_live_knowledge_answer(graph):
    state = run(graph, "My QR code won't show up in the app, what should I do?", "live-1", external_user_id="88382b")
    assert state["route"] == "knowledge_agent"
    assert state["resolution"]["status"] == "resolved"
    assert "app" in state["messages"][-1].content.lower()


def test_live_reservation(graph):
    state = run(graph, "Please book Samba Night at Lapa for me.", "live-2", external_user_id="f556c0")
    assert state["route"] == "account_agent"
    titles = [r["title"] for r in cultpass_tools.list_reservations("f556c0")["reservations"]]
    assert "Samba Night at Lapa" in titles


def test_live_blocked_account_escalates(graph):
    state = run(graph, "I can't log in, please unblock my account.", ALICE_TICKET)
    assert state["resolution"]["status"] == "escalated"
    assert udahub_tools.get_ticket(ALICE_TICKET)["status"] == "escalated"


def test_live_refund_escalates(graph):
    state = run(graph, "I was charged twice, I want my money back.", "live-4", external_user_id="88382b")
    assert state["route"] == "account_agent" and state["resolution"]["status"] == "escalated"
    assert cultpass_tools.list_refund_requests("88382b")["refunds"][0]["status"] == "pending_approval"


def test_live_off_topic_is_not_answered(graph):
    state = run(graph, "Can you give me a lasagna recipe?", "live-5", external_user_id="88382b")
    assert state["resolution"]["status"] == "escalated"


def test_live_multi_turn_memory(graph):
    run(graph, "Which reservations do I have?", "live-6", external_user_id="f556c0")
    state = run(graph, "Thanks, that's all!", "live-6")
    assert state["turn"] == 2 and state["resolution"]["status"] == "resolved"


def test_live_subscription_pause_needs_confirmation(graph):
    state = run(graph, "Please pause my subscription.", "live-7", external_user_id="88382b")
    assert state["pending_action"] and state["pending_action"]["tool"] == "pause_my_subscription"
    assert cultpass_tools.get_customer_profile("88382b")["subscription"]["status"] == "active"
    run(graph, "Yes, please go ahead.", "live-7")
    assert cultpass_tools.get_customer_profile("88382b")["subscription"]["status"] == "paused"


def test_live_email_channel(graph):
    from agentic.channels import from_email

    payload = from_email("bob.stone@granite.com", "QR code", "Hello, my QR code does not appear in the app. What can I do?")
    state = graph.invoke(payload, config={"configurable": {"thread_id": "live-8"}})
    assert state["external_user_id"] == "f556c0" and state["channel"] == "email"
    assert state["messages"][-1].content.rstrip().endswith("CultPass Support (UDA-Hub)")


def test_live_social_channel_has_no_personal_data(graph):
    from agentic.channels import from_social

    state = graph.invoke(from_social("@fan", "@cultpass how can I reset my password?"),
                         config={"configurable": {"thread_id": "live-9"}})
    reply = state["messages"][-1].content
    assert len(reply) <= 280 and "@" not in reply.replace("@cultpass", "")


def test_live_llm_routing_strategy(graph):
    state = graph.invoke(
        {"messages": [HumanMessage("Which reservations do I have?")], "external_user_id": "f556c0"},
        config={"configurable": {"thread_id": "live-10", "routing_strategy": "llm"}},
    )
    assert state["experiment"]["variant"] == "llm" and state["route"] == "account_agent"
