"""Multi-channel support: normalise incoming tickets and adapt replies per channel.

Inbound adapters turn raw platform payloads into graph input::

    orchestrator.invoke(from_email(sender, subject, body), config={"configurable": {"thread_id": tid}})

* ``email``  - the customer is identified by the sender address (CultPass lookup);
  quoted replies / signatures are stripped; replies get a greeting and sign-off.
* ``chat``   - short conversational replies.
* ``social`` - public posts: the customer is never identified publicly, replies
  are short, personal data (e-mails, ids) is redacted and the customer is
  invited to continue in a private message.
"""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from agentic.tools.mcp_client import call_tool

CHANNELS = ("chat", "email", "social")
SOCIAL_MAX_CHARS = 280

STYLE = {
    "chat": "Channel: live chat. Reply in 1-4 short sentences, conversational, no greeting or signature.",
    "email": "Channel: e-mail. Reply as a complete e-mail body in short paragraphs (the greeting and signature are added automatically).",
    "social": (
        "Channel: PUBLIC social-media reply. Max 2 short sentences. Never include personal data "
        "(e-mail, reservation or account ids, plan details); invite the customer to send a private message."
    ),
}

_QUOTE_MARKERS = re.compile(r"^(On .+ wrote:|-----Original Message-----|From: .+|Sent from my .+)$", re.I | re.M)
_SIGNATURE = re.compile(r"^(--\s*|Best,?|Regards,?|Cheers,?|Thanks,?|Kind regards,?)\s*$", re.I | re.M)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
# identifiers such as "e3d965", "rf-1a2b3c", "f556c0": mixed letters+digits, >= 5 chars (dates have no letters)
_IDS = re.compile(r"\b(?=[\w-]*\d)(?=[\w-]*[a-z])[a-z0-9][a-z0-9-]{4,}\b", re.I)


# --------------------------------------------------------------------------- #
# Inbound adapters
# --------------------------------------------------------------------------- #
def clean_email_body(body: str) -> str:
    """Drop quoted history, '>' lines and signatures from an e-mail body."""
    lines = []
    for line in body.strip().splitlines():
        if _QUOTE_MARKERS.match(line.strip()) or _SIGNATURE.match(line.strip()):
            break
        if line.lstrip().startswith(">"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def from_email(sender: str, subject: str, body: str, account_id: str = "cultpass") -> dict:
    """Graph input for an inbound e-mail; the customer is resolved from the sender address."""
    match = _EMAIL.search(sender or "")
    address = match.group(0).lower() if match else None
    external_user_id = None
    if address:
        matches = call_tool("cultpass", "find_customer", {"identifier": address}).get("matches", [])
        external_user_id = matches[0]["user_id"] if len(matches) == 1 else None  # ambiguous -> treat as unknown
    text = f"Subject: {subject.strip()}\n\n{clean_email_body(body)}" if subject else clean_email_body(body)
    payload: dict = {
        "messages": [HumanMessage(text)],
        "account_id": account_id,
        "ticket_metadata": {"channel": "email", "sender": address, "subject": subject,
                            "customer_identified": external_user_id is not None},
    }
    if external_user_id:
        payload["external_user_id"] = external_user_id
    return payload


def from_chat(text: str, external_user_id: str | None = None, account_id: str = "cultpass") -> dict:
    """Graph input for an in-app chat message (the app knows the logged-in customer)."""
    payload: dict = {"messages": [HumanMessage(text.strip())], "account_id": account_id,
                     "ticket_metadata": {"channel": "chat"}}
    if external_user_id:
        payload["external_user_id"] = external_user_id
    return payload


def from_social(handle: str, text: str, platform: str = "x", account_id: str = "cultpass") -> dict:
    """Graph input for a public social-media mention (customer not identified: privacy)."""
    cleaned = re.sub(r"(^|\s)@\w+", " ", text).strip()
    return {
        "messages": [HumanMessage(cleaned)],
        "account_id": account_id,
        "ticket_metadata": {"channel": "social", "platform": platform, "handle": handle, "public": True},
    }


# --------------------------------------------------------------------------- #
# Outbound formatting
# --------------------------------------------------------------------------- #
def channel_of(state: dict) -> str:
    channel = state.get("channel") or (state.get("ticket_metadata") or {}).get("channel") or "chat"
    return channel if channel in CHANNELS else "chat"


def style_for(state: dict) -> str:
    return STYLE[channel_of(state)]


def format_reply(text: str, channel: str, first_name: str = "there") -> str:
    """Deterministic last-mile formatting so every reply fits its channel."""
    text = (text or "").strip()
    if channel == "email":
        # drop any sign-off / placeholder the model wrote itself; ours is appended below
        text = re.split(r"\n\s*(?:best regards|kind regards|regards|sincerely|cheers|warm regards)\b",
                        text, maxsplit=1, flags=re.I)[0].rstrip()
        text = re.sub(r"\[(?:your name|name|agent name)\]", "", text, flags=re.I).rstrip()
        if not re.match(r"^(hi|hello|dear|olá)\b", text, re.I):
            text = f"Hi {first_name},\n\n{text}"
        if "CultPass Support" not in text:
            text += "\n\nBest regards,\nCultPass Support (UDA-Hub)"
        return text
    if channel == "social":
        text = _EMAIL.sub("[private]", text)
        text = _IDS.sub(" ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        dm = " Please send us a DM so we can help privately."
        if "DM" not in text and "private message" not in text.lower():
            text = text.rstrip() + dm
        if len(text) > SOCIAL_MAX_CHARS:
            text = text[: SOCIAL_MAX_CHARS - len(dm) - 1].rsplit(" ", 1)[0] + "…" + dm
        return text
    return text
