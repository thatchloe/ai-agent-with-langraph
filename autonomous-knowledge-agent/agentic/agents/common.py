"""Helpers shared by all agents (prompt context formatting, LLM access)."""
from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

LLMProvider = Callable[[], BaseChatModel]


def conversation(state: dict, max_messages: int = 12) -> list[AnyMessage]:
    """Customer-visible conversation only (no system/tool messages)."""
    msgs = [
        m
        for m in state.get("messages", [])
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.content and not m.tool_calls)
    ]
    return msgs[-max_messages:]


def latest_user_message(state: dict) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else json.dumps(m.content)
    return ""


def format_customer(state: dict) -> str:
    ctx = state.get("customer_context") or {}
    if not ctx.get("ok"):
        return "Customer: not linked to a CultPass account (guest)."
    sub = ctx.get("subscription") or {}
    lines = [
        f"Customer: {ctx.get('full_name')} (CultPass id {ctx.get('user_id')}, email {ctx.get('email')})",
        f"Account blocked: {'YES' if ctx.get('is_blocked') else 'no'}",
    ]
    if sub:
        lines.append(
            f"Subscription: status={sub.get('status')}, tier={sub.get('tier')}, "
            f"monthly_quota={sub.get('monthly_quota')}, remaining_this_month={sub.get('remaining_quota')}"
        )
    else:
        lines.append("Subscription: none")
    return "\n".join(lines)


def format_memories(state: dict) -> str:
    mems = state.get("long_term_memory") or []
    prior = state.get("prior_ticket_messages") or []
    lines: list[str] = []
    if mems:
        lines.append("Long-term memory about this customer (from previous tickets):")
        lines += [f"- ({m['memory_type']}) {m['content']}" for m in mems]
    if prior:
        lines.append("Earlier messages already logged on this ticket:")
        lines += [f"- {m['role']}: {m['content']}" for m in prior[-6:]]
    return "\n".join(lines) or "No previous history with this customer."


def channel_guidance(state: dict) -> str:
    from agentic.channels import style_for

    return style_for(state)


def known_first_name(state: dict) -> str | None:
    name = (state.get("customer_context") or {}).get("full_name") or state.get("user_name") or ""
    return name.split()[0] if name and name != "guest" else None


def first_name(state: dict) -> str:
    """First name for greetings ("Hi there," when unknown)."""
    return known_first_name(state) or "there"


def addressee(state: dict) -> str:
    """How prompts should tell the LLM to address the customer."""
    name = known_first_name(state)
    return f"the customer by first name ({name})" if name else "the customer without a name (name unknown)"


def safe_json(data: Any, limit: int = 4000) -> str:
    text = json.dumps(data, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"
