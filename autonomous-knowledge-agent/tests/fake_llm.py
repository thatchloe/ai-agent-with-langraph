"""A scripted chat model for offline tests.

``ScriptedLLM`` implements the parts of the chat-model interface the agents use
(``invoke``, ``bind_tools``, ``with_structured_output`` via function calling).
``support_brain`` mimics a well-behaved LLM with simple rules so that the real
graph, MCP tools and databases are exercised end to end without network access.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from agentic.agents.classifier import heuristic_classify
from agentic.tools.retrieval import tokenize

Handler = Callable[[list[BaseMessage], list[str]], AIMessage]


class ScriptedLLM(BaseChatModel):
    handler: Any
    bound_tools: list[str] = []
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.model_copy(update={"bound_tools": names})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append({"tools": list(self.bound_tools), "messages": messages})
        message = self.handler(messages, self.bound_tools)
        return ChatResult(generations=[ChatGeneration(message=message)])


def _call(name: str, args: dict, idx: int = 0) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{idx}"}])


def _last_human(messages) -> str:
    return next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))


def _tool_results(messages) -> list[tuple[str, dict]]:
    """Tool results produced after the last human message (the current turn)."""
    out = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out = []
        elif isinstance(m, ToolMessage):
            out.append((m.name, json.loads(m.content)))
    return out


def _classification(text: str) -> AIMessage:
    cls = heuristic_classify(text).model_dump()
    cls["confidence"] = max(cls["confidence"], 0.8) if cls["category"] != "general_question" else 0.7
    if "rio" in text.lower():
        cls["customer_preferences"] = ["Prefers events in Rio de Janeiro"]
    return _call("TicketClassification", cls)


def _knowledge(messages) -> AIMessage:
    system = next(m.content for m in messages if isinstance(m, SystemMessage))
    question = _last_human(messages)
    titles = re.findall(r"^### (.+)$", system, flags=re.M)
    body = system.split("Knowledge-base articles:", 1)[-1]
    phrasing = re.findall(r'\*\*Suggested phrasing:\*\*\n"(.+?)"', body, flags=re.S)
    first_article = body.split("### ")[1] if "### " in body else ""
    overlap = set(tokenize(question)) & set(tokenize(first_article))
    answerable = bool(overlap)
    return _call(
        "KnowledgeAnswer",
        {
            "answer": phrasing[0] if phrasing else "Here is what I found.",
            "answerable": answerable,
            "confidence": 0.9 if answerable else 0.2,
            "used_article_titles": titles[:1],
        },
    )


def _escalation() -> AIMessage:
    return _call(
        "EscalationHandoff",
        {
            "internal_summary": "Customer needs human help.",
            "priority": "medium",
            "recommended_action": "Review and reply.",
            "customer_message": "I've passed your request to a human support specialist who will reply by e-mail.",
        },
    )


_SUBSCRIPTION_INTENTS = [
    ("pause", "pause_my_subscription", {}),
    ("reactivate", "resume_my_subscription", {}),
    ("resume", "resume_my_subscription", {}),
    ("cancel my subscription", "cancel_my_subscription", {}),
    ("upgrade", "change_my_plan", {"new_tier": "premium"}),
    ("downgrade", "change_my_plan", {"new_tier": "basic"}),
]


def _pending_action(messages) -> dict | None:
    system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
    marker = "Pending action awaiting the customer's confirmation: "
    if marker not in system:
        return None
    raw = system.split(marker, 1)[1].strip()
    return None if raw.startswith("none") else json.JSONDecoder().raw_decode(raw)[0]


def _account(messages) -> AIMessage:
    text = _last_human(messages).lower()
    results = _tool_results(messages)
    step = len(results)
    last_name, last = results[-1] if results else (None, {})

    def final(prefix: str = "") -> AIMessage:
        if last.get("ok") is False:
            return AIMessage(content=f"Sorry, I couldn't do that: {last.get('message')}")
        if last.get("requires_confirmation"):
            return AIMessage(content=f"{last.get('preview')} Shall I go ahead?")
        return AIMessage(content=prefix or last.get("message") or json.dumps(last)[:300])

    pending = _pending_action(messages)
    if pending and re.match(r"^(yes|yep|sure|ok|confirm|go ahead|please do)", text):
        if step == 0:
            return _call(pending["tool"], pending["args"])
        return final()
    if pending and re.match(r"^(no|nope|don't|cancel that)", text):
        return AIMessage(content="No problem, I won't make that change.")

    if any(w in text for w in ("refund", "charged", "money back")):
        return _call("request_refund", {"reason": _last_human(messages)})  # executor escalates

    for keyword, tool, args in _SUBSCRIPTION_INTENTS:
        if keyword in text:
            return _call(tool, {**args, "confirm": False}) if step == 0 else final()

    name = re.search(r"change my name to ([a-z ]+)", text)
    if name:
        return _call("update_my_name", {"full_name": name.group(1).title()}) if step == 0 else final()
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    if email and "email" in text:
        return _call("request_email_change", {"new_email": email.group(0)}) if step == 0 else final()

    if "cancel" in text and "reservation" in text:
        if step == 0:
            return _call("list_my_reservations", {})
        if last_name == "list_my_reservations":
            target = next(
                (r for r in last["reservations"] if any(w in r["title"].lower() for w in text.split() if len(w) > 3)),
                last["reservations"][0],
            )
            return _call("cancel_reservation", {"reservation_id": target["reservation_id"]}, step)
        if last_name == "cancel_reservation" and last.get("error") == "late_cancellation":
            return _call("escalate_to_human", {"reason": "late cancellation requested (<24h)"}, step)
        return final()

    if any(w in text for w in ("book", "reserve")):
        if step == 0:
            keyword = next((w for w in ("samba", "paddleboarding", "masp", "opera", "olinda", "jazz") if w in text), "")
            return _call("search_experiences", {"query": keyword}, step)
        if last_name == "search_experiences":
            if not last["experiences"]:
                return AIMessage(content="I couldn't find an available experience matching that.")
            return _call("reserve_experience", {"experience_id": last["experiences"][0]["experience_id"]}, step)
        return final()

    if "my reservations" in text or "booked" in text:
        if step == 0:
            return _call("list_my_reservations", {})
        titles = ", ".join(r["title"] for r in last["reservations"])
        return AIMessage(content=f"You have these reservations: {titles}.")

    # default: check the account first
    if step == 0:
        return _call("get_my_account", {})
    if last_name == "get_my_account" and last.get("is_blocked"):
        return _call("escalate_to_human", {"reason": "account is blocked; customer wants access restored"}, step)
    sub = last.get("subscription") or {}
    return AIMessage(
        content=f"Your {sub.get('tier')} subscription is {sub.get('status')} with {sub.get('remaining_quota')} experiences left this month."
    )


def _route(messages) -> AIMessage:
    """Scripted LLM router: account categories -> account, everything else -> knowledge (never escalates)."""
    system = next(m.content for m in messages if isinstance(m, SystemMessage))
    category = (re.search(r"'category': '(\w+)'", system) or [None, "general_question"])[1]
    route = "account_agent" if category in {"reservation", "subscription", "account_status", "experience_info"} else "knowledge_agent"
    return _call("RouteDecision", {"route": route, "reason": f"scripted router for {category}"})


def support_brain(messages: list[BaseMessage], tools: list[str]) -> AIMessage:
    if "RouteDecision" in tools:
        return _route(messages)
    if "TicketClassification" in tools:
        return _classification(_last_human(messages))
    if "KnowledgeAnswer" in tools:
        return _knowledge(messages)
    if "EscalationHandoff" in tools:
        return _escalation()
    if "get_my_account" in tools:
        return _account(messages)
    raise AssertionError(f"unexpected LLM call with tools={tools}")


def broken_brain(messages, tools) -> AIMessage:
    raise ConnectionError("LLM service unavailable")
