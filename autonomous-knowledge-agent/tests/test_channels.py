"""Multi-channel input adapters and reply formatting."""
from agentic.channels import clean_email_body, format_reply, from_chat, from_email, from_social


def test_clean_email_body_strips_quotes_and_signature():
    body = "Hi,\nmy QR code is missing.\n\nThanks,\nBob\n\nOn Mon, Support wrote:\n> old text"
    assert clean_email_body(body) == "Hi,\nmy QR code is missing."


def test_from_email_identifies_customer_by_sender(dbs):
    payload = from_email("Bob Stone <Bob.Stone@granite.com>", "QR code", "It does not show.\n-- \nBob")
    assert payload["external_user_id"] == "f556c0"
    assert payload["ticket_metadata"]["channel"] == "email" and payload["ticket_metadata"]["customer_identified"]
    assert payload["messages"][0].content == "Subject: QR code\n\nIt does not show."


def test_from_email_unknown_sender_is_not_identified(dbs):
    payload = from_email("stranger@nowhere.com", "Hello", "Question")
    assert "external_user_id" not in payload and not payload["ticket_metadata"]["customer_identified"]


def test_from_social_and_chat():
    social = from_social("@fan", "@cultpass my app keeps crashing")
    assert social["messages"][0].content == "my app keeps crashing"
    assert social["ticket_metadata"] == {"channel": "social", "platform": "x", "handle": "@fan", "public": True}
    assert "external_user_id" not in social
    assert from_chat("hi", "88382b")["external_user_id"] == "88382b"


def test_format_reply_per_channel():
    email = format_reply("Your reservation is confirmed.", "email", "Bob")
    assert email.startswith("Hi Bob,") and email.endswith("CultPass Support (UDA-Hub)")
    social = format_reply("Write to bob.stone@granite.com about reservation e3d965 on 2026-11-11.", "social")
    assert "granite.com" not in social and "e3d965" not in social and "2026-11-11" in social
    assert "DM" in social and len(format_reply("x " * 400, "social")) <= 280
    assert format_reply("Short answer.", "chat") == "Short answer."
    doubled = format_reply("Hi Bob,\n\nDone.\n\nBest regards,\n[Your Name]\nUDA-Hub Account Operations", "email", "Bob")
    assert doubled == "Hi Bob,\n\nDone.\n\nBest regards,\nCultPass Support (UDA-Hub)"
