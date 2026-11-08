"""End-to-end tests of the compiled LangGraph workflow (offline, scripted LLM).

Each test runs the real graph against real SQLite databases through the MCP
tool layer; only the LLM is replaced by a deterministic script.
"""
from langchain_core.messages import AIMessage, HumanMessage

from agentic.logging_utils import read_events
from agentic.tools import cultpass_tools, udahub_tools
from agentic.workflow import build_graph
from tests.fake_llm import ScriptedLLM, broken_brain
from tests.seed import ALICE_TICKET


def run(graph, text, thread_id, **extra):
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"messages": [HumanMessage(text)], **extra}, config=config)


def events(ticket_id, event):
    return read_events(ticket_id=ticket_id, event=event)


def test_graph_structure():
    graph = build_graph(llm=ScriptedLLM(handler=broken_brain, calls=[])).get_graph()
    nodes = set(graph.nodes)
    assert {"ticket_intake", "classifier", "supervisor", "knowledge_agent", "account_agent",
            "account_tools", "escalation_agent", "finalize"} <= nodes


def test_knowledge_question_is_answered_and_ticket_resolved(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    state = run(graph, "The QR code is not showing in the app", "t-kb", external_user_id="88382b")

    assert state["route"] == "knowledge_agent"
    assert state["resolution"]["status"] == "resolved"
    assert state["retrieved_articles"][0]["title"].startswith("App Troubleshooting")
    assert isinstance(state["messages"][-1], AIMessage) and "update the app" in state["messages"][-1].content

    ticket = udahub_tools.get_ticket("t-kb")
    assert ticket["status"] == "resolved" and ticket["main_issue_type"] == "technical_issue"
    assert [m["role"] for m in ticket["messages"]] == ["user", "ai"]
    assert events("t-kb", "routing")[0]["route"] == "knowledge_agent"


def test_reservation_action_through_account_agent(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    state = run(graph, "Please book Samba Night at Lapa for me", "t-book", external_user_id="f556c0")

    assert state["route"] == "account_agent"
    assert state["resolution"]["actions"] == [{"tool": "reserve_experience", "args": {"experience_id": "exp-3"}}]
    assert "Reserved 'Samba Night at Lapa'" in state["messages"][-1].content
    titles = [r["title"] for r in cultpass_tools.list_reservations("f556c0")["reservations"]]
    assert "Samba Night at Lapa" in titles
    tool_calls = [e["tool"] for e in events("t-book", "tool_call")]
    assert tool_calls == ["search_experiences", "reserve_experience"]
    # tool-loop messages stay out of the customer-facing history
    assert all(not getattr(m, "tool_calls", None) for m in state["messages"])


def test_business_rule_failure_is_explained(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    state = run(graph, "Book the Carnival History Tour in Olinda", "t-prem", external_user_id="88382b")
    assert "premium experience" in state["messages"][-1].content
    assert state["resolution"]["actions"] == []


def test_blocked_account_login_is_escalated(dbs, scripted_llm):
    """Existing ticket from the setup notebook: Alice (blocked) cannot log in."""
    graph = build_graph(llm=scripted_llm)
    state = run(graph, "I still can't log in, please help", ALICE_TICKET)

    assert state["external_user_id"] == "a4ab87"
    assert state["prior_ticket_messages"][0]["content"] == "I can't log in to my Cultpass account."
    assert state["route"] == "escalation_agent"  # hard policy, same for every routing strategy
    assert state["resolution"]["status"] == "escalated"
    assert "blocked" in state["escalation_reason"] and state["resolution"]["queue"] == "trust_safety"
    ticket = udahub_tools.get_ticket(ALICE_TICKET)
    assert ticket["status"] == "escalated"
    assert any(m["role"] == "system" and m["content"].startswith("[ESCALATION]") for m in ticket["messages"])


def test_late_cancellation_escalates_after_tool_refusal(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    state = run(graph, "Cancel my reservation for the jazz brunch", "t-late", external_user_id="f556c0")
    assert state["resolution"]["status"] == "escalated"
    assert [e["error"] for e in events("t-late", "tool_call") if e["tool"] == "cancel_reservation"] == ["late_cancellation"]
    assert cultpass_tools.list_reservations("f556c0")["reservations"][-1]["status"] == "reserved"


def test_refund_request_is_filed_then_escalated_to_billing(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    state = run(graph, "I was charged twice this month, I want a refund today!", "t-refund", external_user_id="88382b")
    assert state["route"] == "account_agent"
    assert state["resolution"]["status"] == "escalated"
    assert state["resolution"]["priority"] == "high"  # urgent -> never downgraded
    assert state["resolution"]["queue"] == "billing"
    refunds = cultpass_tools.list_refund_requests("88382b")["refunds"]
    assert len(refunds) == 1 and refunds[0]["status"] == "pending_approval"
    assert refunds[0]["refund_id"] in state["escalation_reason"]
    ticket = udahub_tools.get_ticket("t-refund")
    assert ticket["assigned_queue"] == "billing" and ticket["priority"] in {"P1", "P2"}


def test_unanswerable_question_is_escalated(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    state = run(graph, "What is the capital of France?", "t-off", external_user_id="88382b")
    assert state["route"] == "knowledge_agent"
    assert state["resolution"]["status"] == "escalated"
    assert udahub_tools.get_ticket("t-off")["status"] == "escalated"


def test_short_term_memory_across_turns_in_a_thread(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    run(graph, "What's included in my subscription?", "t-multi", external_user_id="88382b")
    state = run(graph, "Thanks, that's all!", "t-multi")  # no user id: remembered by the thread

    assert state["turn"] == 2 and state["external_user_id"] == "88382b"
    assert [type(m).__name__ for m in state["messages"]] == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]
    assert state["classification"]["category"] == "closing"
    assert state["messages"][-1].content.startswith("You're welcome, Cathy")
    snapshot = graph.get_state({"configurable": {"thread_id": "t-multi"}})
    assert len(snapshot.values["trace"]) > 8  # whole decision history is inspectable
    assert len(udahub_tools.get_ticket("t-multi")["messages"]) == 4


def test_long_term_memory_is_recalled_in_a_new_thread(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    # seeded memory from a past ticket is recalled at intake
    state = run(graph, "How do I reserve a spot for an event?", "t-new-1", external_user_id="f556c0")
    contents = [m["content"] for m in state["long_term_memory"]]
    assert "Enjoys samba / music events in Rio de Janeiro" in contents
    assert "samba" in scripted_llm.calls[-1]["messages"][0].content  # memory injected in the prompt

    # a preference stated now is saved and recalled in a different conversation
    run(graph, "Any upcoming events in Rio de Janeiro?", "t-new-2", external_user_id="88382b")
    state = run(graph, "How do I reserve a spot?", "t-new-3", external_user_id="88382b")
    contents = [m["content"] for m in state["long_term_memory"]]
    assert "Prefers events in Rio de Janeiro" in contents
    assert any("[experience_info]" in c for c in contents)


def test_follow_up_on_escalated_ticket_stays_with_humans(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    run(graph, "I want to talk to a human", "t-esc", external_user_id="88382b")
    state = run(graph, "How do I reserve a spot for an event?", "t-esc")
    assert state["route"] == "escalation_agent"
    assert "already escalated" in state["route_reason"]


def test_graceful_degradation_when_llm_is_down(dbs):
    graph = build_graph(llm=ScriptedLLM(handler=broken_brain, calls=[]))
    state = run(graph, "How do I reserve a spot for an event?", "t-down", external_user_id="88382b")
    assert state["classification"]["category"] == "reservation"   # heuristic fallback
    assert state["resolution"]["status"] == "escalated"
    assert "human support specialist" in state["messages"][-1].content
    assert events("t-down", "escalation")[0]["source"].startswith("template")


def test_guest_conversation_without_customer_id(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    state = run(graph, "How do I reset my password?", "t-guest")
    assert state["external_user_id"] is None
    assert state["resolution"]["status"] == "resolved"
    assert udahub_tools.get_ticket("t-guest")["external_user_id"] == "guest"


def test_every_turn_is_logged(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    run(graph, "How do I pause my subscription?", "t-log", external_user_id="f556c0")
    kinds = [e["event"] for e in read_events(ticket_id="t-log")]
    assert kinds[0] == "intake" and kinds[-1] == "ticket_update"
    assert {"classification", "routing"} <= set(kinds)


def test_ticket_metadata_channel_and_urgency(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    state = run(graph, "Can I pause my subscription?", "t-meta", external_user_id="f556c0",
                ticket_metadata={"channel": "email", "urgency": "high"})
    assert state["classification"]["urgency"] == "high"
    assert udahub_tools.get_ticket("t-meta")["channel"] == "email"
    assert "'channel': 'email'" in scripted_llm.calls[0]["messages"][0].content


def test_subscription_change_two_step_confirmation(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    state = run(graph, "I'd like to pause my subscription", "t-pause", external_user_id="88382b")
    assert state["route"] == "account_agent" and state["pending_action"]["tool"] == "pause_my_subscription"
    assert state["resolution"]["status"] == "in_progress" and "Shall I go ahead" in state["messages"][-1].content
    assert cultpass_tools.get_customer_profile("88382b")["subscription"]["status"] == "active"

    state = run(graph, "Yes, go ahead", "t-pause")  # short confirmation, routed by the pending action
    assert state["route"] == "account_agent" and "waiting for the customer's confirmation" in state["route_reason"]
    assert cultpass_tools.get_customer_profile("88382b")["subscription"]["status"] == "paused"
    assert state["resolution"]["actions"][0]["tool"] == "pause_my_subscription"
    assert state["pending_action"] is None and state["resolution"]["status"] == "resolved"


def test_declined_confirmation_changes_nothing(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    run(graph, "Please cancel my subscription", "t-decline", external_user_id="f556c0")
    state = run(graph, "No, keep it", "t-decline")
    assert "won't make that change" in state["messages"][-1].content
    assert cultpass_tools.get_customer_profile("f556c0")["subscription"]["status"] == "active"


def test_profile_update_through_agent(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    state = run(graph, "Please change my name to Catherine Bloom", "t-name", external_user_id="88382b")
    assert state["route"] == "account_agent"
    assert cultpass_tools.get_customer_profile("88382b")["full_name"] == "Catherine Bloom"


def test_email_channel_end_to_end(dbs, scripted_llm):
    from agentic.channels import from_email

    graph = build_graph(llm=scripted_llm)
    payload = from_email("bob.stone@granite.com", "App problem", "The QR code is not showing in the app.\n\nThanks,\nBob")
    state = graph.invoke(payload, config={"configurable": {"thread_id": "t-email"}})
    assert state["external_user_id"] == "f556c0" and state["channel"] == "email"
    reply = state["messages"][-1].content
    assert reply.startswith("Hi Bob,") and reply.endswith("CultPass Support (UDA-Hub)")
    ticket = udahub_tools.get_ticket("t-email")
    assert ticket["channel"] == "email" and ticket["messages"][-1]["content"] == reply


def test_social_channel_reply_is_public_safe(dbs, scripted_llm):
    from agentic.channels import from_social

    graph = build_graph(llm=scripted_llm)
    state = graph.invoke(from_social("@fan", "@cultpass how do I reset my password?"),
                         config={"configurable": {"thread_id": "t-social"}})
    reply = state["messages"][-1].content
    assert state["channel"] == "social" and state["external_user_id"] is None
    assert "DM" in reply and len(reply) <= 280


def test_priority_is_stored_and_escalations_get_a_queue(dbs, scripted_llm):
    graph = build_graph(llm=scripted_llm)
    state = run(graph, "I still can't log in, please help", ALICE_TICKET)
    ticket = udahub_tools.get_ticket(ALICE_TICKET)
    assert ticket["priority"] == state["priority"] and ticket["assigned_queue"] == "trust_safety"
    assert events(ALICE_TICKET, "routing")[0]["priority_factors"]
