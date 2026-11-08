"""Classifier agent: turns a raw customer message into structured ticket metadata.

The LLM produces a ``TicketClassification`` (structured output). Two
deterministic safeguards wrap it:
  * an explicit request for a human always wins (``HUMAN_REQUEST``), and
  * if the LLM is unavailable/fails, a keyword classifier is used so the
    ticket is still routed instead of crashing the conversation.
"""
from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agentic.agents.common import LLMProvider, conversation, format_customer, latest_user_message
from agentic.logging_utils import log_event

Category = Literal[
    "login_access",
    "account_status",
    "reservation",
    "subscription",
    "billing_refund",
    "experience_info",
    "technical_issue",
    "general_question",
    "human_request",
    "complaint",
    "closing",
]

CATEGORY_GUIDE = """\
- login_access: cannot log in, password reset, verification e-mail
- account_status: account blocked / suspended / locked, profile, name or e-mail change
- reservation: book, cancel or check a reservation for an experience
- subscription: plan, tier, quota, pause/resume/cancel subscription, upgrade/downgrade, what is included
  (a short confirmation like "yes, go ahead" keeps the category of the request it confirms)
- billing_refund: refunds, charges, failed payments, invoices, money back
- experience_info: what events/experiences exist, availability, premium events
- technical_issue: app crashes, QR code not showing, bugs
- general_question: other questions about how CultPass works
- human_request: the customer explicitly asks for a human/agent/representative
- complaint: angry or dissatisfied feedback without a concrete request
- closing: thanks/goodbye/confirmation that the problem is solved"""


class TicketClassification(BaseModel):
    """Structured classification of the customer's latest message."""

    category: Category = Field(description="Main issue type of the latest customer message.")
    urgency: Literal["low", "medium", "high"] = Field(description="How time-sensitive the issue is.")
    sentiment: Literal["positive", "neutral", "negative", "angry"] = Field(
        description="angry only for explicit anger / strong frustration (insults, 'worst', 'ridiculous', shouting, "
        "repeated '!!'); mild dissatisfaction or a problem report is 'negative'."
    )
    confidence: float = Field(ge=0, le=1, description="Confidence in the category, 0-1.")
    summary: str = Field(description="One sentence restating what the customer needs (used as search query).")
    tags: list[str] = Field(default_factory=list, description="2-4 short lowercase tags.")
    customer_preferences: list[str] = Field(
        default_factory=list,
        description="Durable preferences the customer stated (e.g. 'prefers events in Rio de Janeiro'). Empty if none.",
    )


HUMAN_REQUEST = re.compile(
    r"\b(human|real person|representative|live agent|(speak|talk|chat) (to|with) (an? |the )?(agent|someone|somebody|person|manager|supervisor))\b",
    re.I,
)

_KEYWORDS: list[tuple[str, str]] = [
    ("billing_refund", r"refund|money back|charged|charge|billing|invoice|payment|credit card|reimburs"),
    ("account_status", r"blocked|suspended|locked out|banned|disabled|deactivated|(change|update) (my )?(e-?mail|name|profile)"),
    ("login_access", r"log ?in|sign ?in|password|can't access|cannot access|verification"),
    ("reservation", r"reserv|book|booking|cancel (my|the) (spot|ticket|reservation)|my spot|attend"),
    ("subscription", r"subscription|plan|pause|tier|quota|upgrade|downgrade|membership"),
    ("technical_issue", r"crash|bug|error|qr code|not loading|freez|app (is|keeps)"),
    ("experience_info", r"event|experience|concert|museum|tour|what's on|available"),
    ("complaint", r"terrible|awful|unacceptable|worst|ridiculous|disappointed|angry"),
    ("closing", r"^(thanks|thank you|thx|great|perfect|that'?s all|bye|goodbye|ok thanks)\b"),
]
_URGENT = re.compile(r"urgent|asap|immediately|right now|today|tonight", re.I)
_ANGRY = re.compile(r"terrible|awful|angry|unacceptable|worst|ridiculous|disgusting|!!", re.I)
_NEGATIVE = re.compile(r"frustrat|annoy|disappointed|not working|doesn't work|can't|cannot|problem|missing", re.I)


def heuristic_classify(text: str) -> TicketClassification:
    lowered = text.lower().strip()
    category, confidence = "general_question", 0.3
    if HUMAN_REQUEST.search(lowered):
        category, confidence = "human_request", 0.9
    else:
        for cat, pattern in _KEYWORDS:
            if re.search(pattern, lowered):
                category, confidence = cat, 0.6
                break
    angry = bool(_ANGRY.search(lowered))
    negative = angry or bool(_NEGATIVE.search(lowered))
    return TicketClassification(
        category=category,  # type: ignore[arg-type]
        urgency="high" if _URGENT.search(lowered) or angry else "medium",
        sentiment="angry" if angry else "negative" if negative else ("positive" if category == "closing" else "neutral"),
        confidence=confidence,
        summary=text.strip()[:200],
        tags=[category.replace("_", " ")],
    )


SYSTEM_PROMPT = """You are the Classifier agent of UDA-Hub, a customer-support platform. \
The current client account is CultPass, a cultural-experiences subscription (museums, concerts, tours in Brazil).
Classify ONLY the customer's latest message, using the conversation for context.

Categories:
{categories}

Rules:
- Use human_request only when the customer explicitly asks for a person.
- urgency=high for same-day events, being charged incorrectly, or strong frustration.
- sentiment=angry only for explicit anger; a customer calmly reporting a problem is "negative".
- customer_preferences: only lasting preferences (favourite city, event type, language, contact preference).

Customer context:
{customer}

Ticket metadata from the source platform: {metadata}"""


class ClassifierAgent:
    name = "classifier"

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def classify(self, state: dict) -> tuple[TicketClassification, str]:
        text = latest_user_message(state)
        try:
            prompt = SystemMessage(
                SYSTEM_PROMPT.format(
                    categories=CATEGORY_GUIDE,
                    customer=format_customer(state),
                    metadata=state.get("ticket_metadata") or {},
                )
            )
            structured = self.llm().with_structured_output(TicketClassification, method="function_calling")
            result = structured.invoke([prompt, *conversation(state, 8)])
            if not isinstance(result, TicketClassification):
                result = TicketClassification.model_validate(result)
            source = "llm"
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            result, source = heuristic_classify(text), f"heuristic ({type(exc).__name__})"
        # urgency set by the source platform (e.g. a VIP queue) is a floor, never lowered
        if (state.get("ticket_metadata") or {}).get("urgency") == "high" and result.urgency != "high":
            result = result.model_copy(update={"urgency": "high"})
            source += "+metadata_urgency"
        # explicit anger is never under-rated by the LLM
        if _ANGRY.search(text) and result.sentiment != "angry":
            result = result.model_copy(update={"sentiment": "angry"})
            source += "+anger_guardrail"
        if HUMAN_REQUEST.search(text) and result.category != "human_request":
            result = result.model_copy(update={"category": "human_request", "confidence": 0.95})
            source += "+human_request_guardrail"
        return result, source

    def __call__(self, state: dict) -> dict:
        result, source = self.classify(state)
        record = log_event(
            "classification",
            state.get("ticket_id"),
            self.name,
            source=source,
            **result.model_dump(),
        )
        return {"classification": result.model_dump(), "trace": [record]}
