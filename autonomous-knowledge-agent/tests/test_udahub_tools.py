"""Ticket lifecycle and long-term memory tools (core DB)."""
from agentic.tools import udahub_tools as ut
from tests.seed import ALICE_TICKET


def test_create_ticket_message_and_metadata(dbs):
    created = ut.create_ticket("cultpass", "88382b", "Cathy Bloom", ticket_id="t-1")
    assert created["ok"]
    assert ut.create_ticket("cultpass", "88382b", ticket_id="t-1")["error"] == "ticket_exists"
    assert ut.add_ticket_message("t-1", "user", "hello")["ok"]
    assert ut.add_ticket_message("t-1", "ai", "hi!")["ok"]
    ut.update_ticket_metadata("t-1", status="resolved", main_issue_type="general_question", tags=["greeting"])
    meta = ut.update_ticket_metadata("t-1", tags=["greeting", "chat"])
    assert meta["tags"] == "greeting, chat"
    ticket = ut.get_ticket("t-1")
    assert ticket["status"] == "resolved" and [m["role"] for m in ticket["messages"]] == ["user", "ai"]


def test_invalid_status_and_unknown_ticket(dbs):
    assert ut.update_ticket_metadata(ALICE_TICKET, status="weird")["error"] == "invalid_status"
    assert ut.get_ticket("missing")["error"] == "ticket_not_found"
    assert ut.add_ticket_message("missing", "user", "x")["error"] == "ticket_not_found"


def test_existing_ticket_maps_to_customer(dbs):
    ticket = ut.get_ticket(ALICE_TICKET)
    assert ticket["external_user_id"] == "a4ab87" and ticket["messages"][0]["role"] == "user"


def test_memory_save_search_and_dedupe(dbs):
    ut.save_memory("cultpass", "88382b", "Prefers museum visits", memory_type="preference", user_name="Cathy")
    again = ut.save_memory("cultpass", "88382b", "prefers museum visits", memory_type="preference")
    assert again["deduplicated"]
    ut.save_memory("cultpass", "88382b", "[reservation] booked samba night -> resolved")
    ut.save_memory("cultpass", "88382b", "[login_access] password reset -> resolved")
    mems = ut.search_memories("cultpass", "88382b", query="password problem")["memories"]
    assert mems[0]["memory_type"] == "preference"
    assert "password" in mems[1]["content"]
    assert ut.search_memories("cultpass", "unknown-user")["memories"] == []


def test_ticket_history_excludes_current(dbs):
    history = ut.get_customer_ticket_history("cultpass", "f556c0")["tickets"]
    assert history[0]["ticket_id"] == "t-bob-past" and history[0]["status"] == "resolved"
    assert ut.get_customer_ticket_history("cultpass", "f556c0", exclude_ticket_id="t-bob-past")["tickets"] == []
