"""Escalation agent: hands the ticket to a human support agent.

Produces (a) an internal hand-off note for the human team (summary, priority,
recommended next step), stored on the ticket as a ``system`` message, and
(b) a customer-facing message setting expectations. Falls back to a template
if the LLM is unavailable, so escalation never fails.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic.agents.common import (
    LLMProvider,
    channel_guidance,
    conversation,
    addressee,
    known_first_name,
    format_customer,
    format_memories,
)
from agentic.logging_utils import log_event
from agentic.tools.mcp_client import call_tool


class EscalationHandoff(BaseModel):
    """Hand-off to the human support team."""

    internal_summary: str = Field(description="2-3 sentences for the human agent: issue, what was tried, what is needed.")
    priority: Literal["low", "medium", "high"]
    recommended_action: str = Field(description="Concrete next step for the human agent.")
    customer_message: str = Field(
        description="Short, empathetic message to the customer saying a human specialist will follow up. No promises of outcome."
    )


SYSTEM_PROMPT = """You are the Escalation agent of UDA-Hub (CultPass account).
The ticket must be handed to a human support specialist. Reason: {reason}
Write the hand-off note and a customer message (address {addressee}, max 70 words,
say a human specialist will review the ticket and reply by e-mail, do NOT promise refunds or outcomes).
{channel_style}

{customer}

Classification: {classification}

{memory}"""

_PRIORITY_BY_URGENCY = {"high": "high", "medium": "medium", "low": "low"}


def choose_queue(state: dict) -> str:
    """Human queue that should own the escalated ticket."""
    cls = state.get("classification") or {}
    category = cls.get("category")
    reason = (state.get("escalation_reason") or "").lower()
    if category == "billing_refund" or "refund" in reason or "charge" in reason:
        return "billing"
    if category in {"account_status", "login_access"} or (state.get("customer_context") or {}).get("is_blocked"):
        return "trust_safety"
    if category == "technical_issue":
        return "technical"
    return "general"


class EscalationAgent:
    name = "escalation_agent"

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def _fallback(self, state: dict, reason: str) -> EscalationHandoff:
        cls = state.get("classification") or {}
        return EscalationHandoff(
            internal_summary=f"{cls.get('summary', 'Customer request')} | Escalation reason: {reason}",
            priority=_PRIORITY_BY_URGENCY.get(cls.get("urgency", "medium"), "medium"),  # type: ignore[arg-type]
            recommended_action="Review the conversation and contact the customer.",
            customer_message=(
                f"Thanks for your patience{', ' + known_first_name(state) if known_first_name(state) else ''}. I've passed your request to a human "
                "support specialist, who will review it and get back to you by e-mail as soon as possible."
            ),
        )

    def __call__(self, state: dict) -> dict:
        ticket_id = state.get("ticket_id")
        cls = state.get("classification") or {}
        reason = state.get("escalation_reason") or f"routed by supervisor: {state.get('route_reason') or cls.get('category')}"
        if state.get("ticket_status") == "escalated" and not state.get("escalation_reason"):
            reason = "follow-up message on an already escalated ticket"

        try:
            prompt = SystemMessage(
                SYSTEM_PROMPT.format(
                    reason=reason,
                    addressee=addressee(state),
                    customer=format_customer(state),
                    classification=cls,
                    memory=format_memories(state),
                    channel_style=channel_guidance(state),
                )
            )
            structured = self.llm().with_structured_output(EscalationHandoff, method="function_calling")
            handoff = structured.invoke([prompt, *conversation(state)])
            if not isinstance(handoff, EscalationHandoff):
                handoff = EscalationHandoff.model_validate(handoff)
            source = "llm"
        except Exception as exc:  # noqa: BLE001
            handoff, source = self._fallback(state, reason), f"template ({type(exc).__name__})"

        # urgent issues are never downgraded by the LLM
        if cls.get("urgency") == "high":
            handoff = handoff.model_copy(update={"priority": "high"})

        queue = choose_queue(state)
        note = (
            f"[ESCALATION] priority={state.get('priority') or handoff.priority} ({handoff.priority}) queue={queue}\n"
            f"reason: {reason}\n"
            f"summary: {handoff.internal_summary}\nrecommended action: {handoff.recommended_action}"
        )
        call_tool("udahub", "add_ticket_message", {"ticket_id": ticket_id, "role": "system", "content": note})
        record = log_event(
            "escalation",
            ticket_id,
            self.name,
            reason=reason,
            priority=handoff.priority,
            queue=choose_queue(state),
            source=source,
        )
        return {
            "messages": [AIMessage(handoff.customer_message, name=self.name)],
            "escalation_reason": reason,
            "resolution": {
                "status": "escalated",
                "agent": self.name,
                "priority": handoff.priority,
                "reason": reason,
                "queue": queue,
            },
            "trace": [record],
        }
