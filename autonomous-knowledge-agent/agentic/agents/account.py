"""Account Operations agent: a hand-built tool-calling loop (no prebuilt ReAct agent).

Graph shape:  account_agent --(tool calls)--> account_tools --> account_agent ...
              account_agent --(final reply)--> finalize
              account_tools --(escalate_to_human)--> escalation_agent

The loop's intermediate AI/tool messages live in ``state["scratchpad"]`` so the
customer-facing ``messages`` history only contains the final reply.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from agentic.agents.common import (
    LLMProvider,
    conversation,
    addressee,
    format_customer,
    channel_guidance,
    format_memories,
    safe_json,
)
from agentic.agents.knowledge import relevant_articles
from agentic.logging_utils import log_event
from agentic.tools.agent_tools import (
    CONFIRMABLE_TOOLS,
    ESCALATION_TOOL_NAME,
    REFUND_TOOL_NAME,
    build_account_tools,
)

MAX_TOOL_ROUNDS = 6
MUTATING_TOOLS = {
    "reserve_experience", "cancel_reservation", "pause_my_subscription", "resume_my_subscription",
    "cancel_my_subscription", "change_my_plan", "request_refund", "update_my_name", "request_email_change",
}

SYSTEM_PROMPT = """You are the Account Operations agent of UDA-Hub, serving CultPass customers.
You can read and change the customer's CultPass data ONLY through your tools.

Rules:
1. Always look facts up with tools before stating them (account, subscription, reservations, events). Never invent IDs, dates or availability.
2. Only reserve or cancel when the customer clearly asked for that specific action. If the request is ambiguous (e.g. several matching events or reservations), ask which one instead of guessing.
3. If a tool returns ok=false, explain the reason in plain language and offer the next step.
   If a search returns no results, retry once with a broader search (one keyword, no location) before answering,
   and never claim an event is sold out unless a tool explicitly returned error=sold_out.
4. Subscription changes (pause, resume, cancel, change plan) are TWO-STEP: call the tool with confirm=false, tell the
   customer exactly what will happen (the preview) and ask them to confirm. Only call with confirm=true when the
   customer's latest message explicitly confirms the pending action below.
5. Refunds: you cannot approve refunds. Call request_refund with the customer's reason (and amount / reservation_id
   if given); the ticket is then handed to the billing team automatically.
6. E-mail changes use request_email_change (a confirmation link is sent); name changes use update_my_name.
7. Call escalate_to_human (instead of answering) when: the account is blocked and the customer wants access
   restored, a late cancellation (<24h) is requested, or a tool keeps failing.
8. Never answer policy / how-to questions (guests, cancellations, rules, what is included...) from your own
   knowledge: call search_knowledge_base and base the answer ONLY on the returned articles. If they don't cover
   it, say so and offer a human.
9. Be concise and friendly; address {addressee}. Mention reservation IDs and dates when you act.
{channel_style}

Pending action awaiting the customer's confirmation: {pending}

Help-center articles relevant to this message (policy answers must come from these or search_knowledge_base):
{articles}

{customer}

Ticket classification: {classification}

{memory}"""


class AccountAgent:
    name = "account_agent"

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def __call__(self, state: dict) -> dict:
        ticket_id = state.get("ticket_id")
        scratch = list(state.get("scratchpad") or [])
        rounds = sum(isinstance(m, AIMessage) for m in scratch)
        if rounds >= MAX_TOOL_ROUNDS:
            reason = f"Account agent exceeded {MAX_TOOL_ROUNDS} tool rounds without resolving the request"
            return {"escalation_reason": reason, "scratchpad": []}

        tools = build_account_tools(state["account_id"], state.get("external_user_id"))
        cls = state.get("classification") or {}
        updates: dict = {}
        articles = state.get("retrieved_articles") or []
        if rounds == 0:  # retrieve once per turn; later rounds reuse it from state
            articles = relevant_articles(state)
            updates["retrieved_articles"] = articles
        prompt = SystemMessage(
            SYSTEM_PROMPT.format(
                addressee=addressee(state),
                customer=format_customer(state),
                classification=f"{cls.get('category')} - {cls.get('summary')}",
                memory=format_memories(state),
                channel_style=channel_guidance(state),
                pending=safe_json(state.get("awaiting_confirmation")) if state.get("awaiting_confirmation") else "none",
                articles="\n\n".join(f"### {a['title']}\n{a['content']}" for a in articles) or "none found",
            )
        )
        try:
            # grounding guarantee: the first step of every turn must be a tool call (look facts up, never guess)
            bound = self.llm().bind_tools(tools, tool_choice="required") if rounds == 0 else self.llm().bind_tools(tools)
            response = bound.invoke([prompt, *conversation(state), *scratch])
        except Exception as exc:  # noqa: BLE001
            return {
                "escalation_reason": f"Account agent unavailable ({type(exc).__name__})",
                "scratchpad": [],
            }

        if response.tool_calls:
            record = log_event(
                "tool_decision",
                ticket_id,
                self.name,
                tool_calls=[{"name": tc["name"], "args": tc["args"]} for tc in response.tool_calls],
            )
            return {**updates, "scratchpad": scratch + [response], "trace": [record]}

        actions = _actions_from_scratchpad(scratch)
        record = log_event("answer", ticket_id, self.name, tool_rounds=rounds, actions=actions)
        pending = state.get("pending_action")
        return {
            **updates,
            "messages": [AIMessage(response.content, name=self.name)],
            "scratchpad": [],
            # waiting for the customer to confirm a previewed change -> ticket stays in progress
            "resolution": {"status": "in_progress" if pending else "resolved", "agent": self.name, "actions": actions},
            "trace": [record],
        }


class AccountToolExecutor:
    """Executes the tool calls requested by the account agent (custom ToolNode)."""

    name = "account_tools"

    def __call__(self, state: dict) -> dict:
        ticket_id = state.get("ticket_id")
        scratch = list(state.get("scratchpad") or [])
        last = scratch[-1]
        tools = {t.name: t for t in build_account_tools(state["account_id"], state.get("external_user_id"))}
        updates: dict = {}
        records = []
        for call in last.tool_calls:
            tool = tools.get(call["name"])
            if tool is None:
                result = {"ok": False, "error": "unknown_tool", "message": f"No tool named {call['name']}"}
            else:
                try:
                    result = tool.invoke(call["args"])
                except Exception as exc:  # noqa: BLE001
                    result = {"ok": False, "error": "tool_exception", "message": str(exc)}
            if call["name"] == ESCALATION_TOOL_NAME:
                updates["escalation_reason"] = call["args"].get("reason", "Escalation requested by account agent")
            elif call["name"] == REFUND_TOOL_NAME and isinstance(result, dict) and result.get("ok"):
                # policy enforced in code: every refund request goes to a human approver
                updates["escalation_reason"] = (
                    f"refund request {result['refund_id']} filed (pending_approval); needs support-lead decision"
                )
            elif call["name"] in CONFIRMABLE_TOOLS and isinstance(result, dict):
                if result.get("requires_confirmation"):
                    updates["pending_action"] = {"tool": call["name"], "args": {**call["args"], "confirm": True},
                                                 "preview": result.get("preview")}
                elif call["args"].get("confirm"):
                    updates["pending_action"] = None  # executed (or refused): nothing pending any more
            scratch.append(
                ToolMessage(content=safe_json(result), tool_call_id=call["id"], name=call["name"])
            )
            records.append(
                log_event(
                    "tool_call",
                    ticket_id,
                    "account_agent",
                    tool=call["name"],
                    args=call["args"],
                    ok=result.get("ok") if isinstance(result, dict) else None,
                    error=result.get("error") if isinstance(result, dict) else None,
                    outcome=(result.get("message") or "")[:200] if isinstance(result, dict) else None,
                )
            )
        return {**updates, "scratchpad": [] if "escalation_reason" in updates else scratch, "trace": records}


def _actions_from_scratchpad(scratch: list) -> list[dict]:
    """Successful state-changing tool calls performed during this turn."""
    actions = []
    calls = {}
    for msg in scratch:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                calls[tc["id"]] = tc
        elif isinstance(msg, ToolMessage) and msg.name in MUTATING_TOOLS:
            try:
                ok = json.loads(msg.content).get("ok")
            except (ValueError, AttributeError):
                ok = False
            try:
                preview_only = json.loads(msg.content).get("requires_confirmation", False)
            except (ValueError, AttributeError):
                preview_only = False
            if ok and not preview_only:
                actions.append({"tool": msg.name, "args": calls.get(msg.tool_call_id, {}).get("args")})
    return actions


def route_after_account_agent(state: dict) -> str:
    if state.get("escalation_reason"):
        return "escalation_agent"
    scratch = state.get("scratchpad") or []
    if scratch and isinstance(scratch[-1], AIMessage) and scratch[-1].tool_calls:
        return "account_tools"
    return "finalize"


def route_after_tools(state: dict) -> str:
    return "escalation_agent" if state.get("escalation_reason") else "account_agent"
