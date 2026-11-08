"""Subscription management, refunds, account modifications and staff tools."""
from agentic.tools import cultpass_tools as ct
from agentic.tools import mcp_client
from agentic.tools import udahub_tools as ut
from agentic.tools.agent_tools import build_account_tools


def _status(user_id):
    return ct.get_customer_profile(user_id)["subscription"]["status"]


def test_pause_requires_confirmation_then_resume(dbs):
    preview = ct.pause_subscription("88382b")
    assert preview["requires_confirmation"] and _status("88382b") == "active"   # nothing changed yet
    assert ct.pause_subscription("88382b", confirm=True)["status"] == "paused"
    assert ct.pause_subscription("88382b", confirm=True)["error"] == "invalid_state"
    assert ct.reserve_experience("88382b", "exp-1")["error"] == "subscription_inactive"
    assert ct.resume_subscription("88382b", confirm=True)["status"] == "active"


def test_cancel_subscription_keeps_access_until_end_of_cycle(dbs):
    preview = ct.cancel_subscription("f556c0")
    assert preview["requires_confirmation"] and "No refund" in preview["preview"]
    done = ct.cancel_subscription("f556c0", confirm=True)
    assert done["status"] == "cancelled" and done["access_until"]
    assert ct.cancel_subscription("f556c0", confirm=True)["error"] == "invalid_state"
    assert ct.resume_subscription("f556c0", confirm=True)["ok"]  # reactivation of a cancelled plan


def test_change_tier_rules(dbs):
    assert ct.change_subscription_tier("88382b", "gold")["error"] == "invalid_tier"
    assert ct.change_subscription_tier("88382b", "basic")["error"] == "invalid_state"
    assert ct.change_subscription_tier("88382b", "premium", confirm=True)["tier"] == "premium"
    assert ct.reserve_experience("88382b", "exp-0")["ok"]  # premium event now allowed
    # Bob holds upcoming premium reservations -> downgrade refused
    assert ct.change_subscription_tier("f556c0", "basic", confirm=True)["error"] == "has_premium_reservations"


def test_subscription_changes_blocked_for_blocked_or_unknown_accounts(dbs):
    assert ct.pause_subscription("a4ab87", confirm=True)["error"] == "account_blocked"
    assert ct.cancel_subscription("nobody")["error"] == "user_not_found"


def test_refund_request_validation_dedupe_and_staff_decision(dbs):
    assert ct.request_refund("88382b", "x")["error"] == "invalid_reason"
    assert ct.request_refund("88382b", "charged twice", amount=-5)["error"] == "invalid_amount"
    assert ct.request_refund("88382b", "charged twice", reservation_id="r-bob-1")["error"] == "reservation_not_found"
    first = ct.request_refund("f556c0", "Venue was closed", amount=50.0, reservation_id="r-bob-2")
    assert first["status"] == "pending_approval"
    again = ct.request_refund("f556c0", "Venue was closed", reservation_id="r-bob-2")
    assert again["duplicate"] and again["refund_id"] == first["refund_id"]
    # only a human (staff tool) can decide
    assert ct.decide_refund(first["refund_id"], True, decided_by="")["error"] == "missing_agent"
    assert ct.decide_refund(first["refund_id"], True, decided_by="lead@cultpass", note="ok")["status"] == "approved"
    assert ct.decide_refund(first["refund_id"], False, decided_by="lead@cultpass")["error"] == "already_decided"
    assert ct.list_refund_requests("f556c0")["refunds"][0]["decided_by"] == "lead@cultpass"


def test_profile_and_email_changes(dbs):
    assert ct.update_profile("88382b", "C")["error"] == "invalid_name"
    assert ct.update_profile("88382b", "  Catherine   Bloom ")["full_name"] == "Catherine Bloom"
    assert ct.update_profile("a4ab87", "Alice K")["error"] == "account_blocked"
    assert ct.request_email_change("88382b", "not-an-email")["error"] == "invalid_email"
    assert ct.request_email_change("88382b", "bob.stone@granite.com")["error"] == "email_in_use"
    sent = ct.request_email_change("88382b", "Cathy@New.org")
    assert sent["status"] == "verification_sent"
    assert ct.get_customer_profile("88382b")["email"] == "cathy.bloom@florals.org"  # unchanged until verified


def test_staff_block_unblock(dbs):
    assert ct.set_account_block("a4ab87", False, "ok", decided_by="lead")["error"] == "missing_fields"
    assert ct.set_account_block("a4ab87", False, "identity verified", decided_by="lead")["is_blocked"] is False
    assert ct.get_customer_profile("a4ab87")["is_blocked"] is False


def test_ticket_priority_queue_and_staff_work_queue(dbs):
    for tid, prio in [("t-a", "P3"), ("t-b", "P1"), ("t-c", "P2")]:
        ut.create_ticket("cultpass", "88382b", ticket_id=tid)
        ut.update_ticket_metadata(tid, status="escalated", priority=prio, assigned_queue="billing")
    assert ut.update_ticket_metadata("t-a", priority="P9")["error"] == "invalid_priority"
    assert ut.assign_ticket("t-a", "nowhere")["error"] == "invalid_queue"
    queue = ut.list_tickets(status="escalated", assigned_queue="billing")["tickets"]
    assert [t["ticket_id"] for t in queue] == ["t-b", "t-c", "t-a"]  # most urgent first
    ut.assign_ticket("t-a", "technical", priority="P2")
    assert ut.get_ticket("t-a")["assigned_queue"] == "technical"


def test_mcp_exposes_all_operations_but_ai_gets_no_staff_tools(dbs):
    cultpass_tools = set(mcp_client.list_tools("cultpass"))
    assert {"pause_subscription", "resume_subscription", "cancel_subscription", "change_subscription_tier",
            "request_refund", "list_refund_requests", "update_profile", "request_email_change",
            "decide_refund", "set_account_block"} <= cultpass_tools
    assert {"list_tickets", "assign_ticket"} <= set(mcp_client.list_tools("udahub"))
    ai_tools = {t.name for t in build_account_tools("cultpass", "88382b")}
    assert not ai_tools & {"decide_refund", "set_account_block", "find_customer", "list_tickets", "assign_ticket"}
    paused = mcp_client.call_tool("cultpass", "pause_subscription", {"user_id": "88382b", "confirm": True})
    assert paused["status"] == "paused"
