"""Supervisor: routes a classified ticket to a specialist agent and sets its priority.

Two interchangeable routing strategies exist so they can be A/B tested
(see ``agentic/experiments.py``):

* ``rules`` - deterministic rules on classification + ticket metadata + customer context.
* ``llm``   - an LLM router choosing the specialist from the same inputs.

Both strategies sit behind the same hard **policies** (``policy_route``): those
can never be overridden by a strategy (e.g. a human request always reaches a human).
"""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agentic.agents.common import LLMProvider, format_customer, latest_user_message
from agentic.experiments import EXPERIMENT_NAME, assign_variant, strategy_setting
from agentic.logging_utils import log_event

KNOWLEDGE = "knowledge_agent"
ACCOUNT = "account_agent"
ESCALATION = "escalation_agent"
ROUTES = (KNOWLEDGE, ACCOUNT, ESCALATION)

ACCOUNT_CATEGORIES = {"reservation", "subscription", "account_status", "experience_info"}
MIN_CLASSIFIER_CONFIDENCE = 0.35
MAX_AI_TURNS = 5  # complexity: longer conversations go to a human


# --------------------------------------------------------------------------- #
# Priority (sentiment analysis + urgency + customer value)
# --------------------------------------------------------------------------- #
def compute_priority(state: dict) -> tuple[str, int, list[str]]:
    """P1 (most urgent) .. P4. Returns (priority, score, explanation)."""
    cls = state.get("classification") or {}
    customer = state.get("customer_context") or {}
    metadata = state.get("ticket_metadata") or {}
    score, why = 0, []
    urgency = cls.get("urgency", "medium")
    score += {"high": 3, "medium": 2, "low": 1}.get(urgency, 2)
    why.append(f"urgency={urgency}")
    sentiment = cls.get("sentiment", "neutral")
    if sentiment == "angry":
        score += 3
        why.append("angry customer")
    elif sentiment == "negative":
        score += 2
        why.append("negative sentiment")
    elif sentiment == "positive":
        score -= 1
    if ((customer.get("subscription") or {}).get("tier")) == "premium":
        score += 1
        why.append("premium customer")
    if metadata.get("urgency") == "high":
        score += 1
        why.append("platform marked urgent")
    if cls.get("category") in {"billing_refund", "account_status"}:
        score += 1
        why.append("money/access at stake")
    if cls.get("category") == "closing":
        score = 0
    priority = "P1" if score >= 6 else "P2" if score >= 4 else "P3" if score >= 2 else "P4"
    return priority, score, why


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def policy_route(state: dict) -> tuple[str, str] | None:
    """Hard policies shared by every routing strategy."""
    cls = state.get("classification") or {}
    category = cls.get("category", "general_question")
    if state.get("ticket_status") == "escalated":
        return ESCALATION, "ticket is already escalated to a human agent"
    if category == "human_request":
        return ESCALATION, "policy: customer asked for a human"
    if (state.get("customer_context") or {}).get("is_blocked") and category in {"login_access", "account_status"}:
        return ESCALATION, "policy: blocked accounts can only be reviewed by trust & safety (AI cannot unblock)"
    if category == "billing_refund":
        if state.get("external_user_id"):
            return ACCOUNT, "policy: file a refund request, then hand it to the billing team"
        return ESCALATION, "policy: refund request from an unidentified customer"
    if state.get("awaiting_confirmation") and category not in {"complaint"}:
        return ACCOUNT, "a previewed account change is waiting for the customer's confirmation"
    return None


def decide_route(state: dict) -> tuple[str, str]:
    """Strategy ``rules``: return (route, reason)."""
    forced = policy_route(state)
    if forced:
        return forced
    cls = state.get("classification") or {}
    category = cls.get("category", "general_question")

    if category == "complaint" and (cls.get("urgency") == "high" or cls.get("sentiment") in {"negative", "angry"}):
        return ESCALATION, "negative / urgent complaint"
    if category != "closing" and cls.get("sentiment") == "angry" and cls.get("urgency") == "high":
        return ESCALATION, "sentiment: angry customer with an urgent issue is prioritised to a human"
    if category != "closing" and (state.get("turn") or 0) > MAX_AI_TURNS:
        return ESCALATION, f"complexity: still unresolved after {MAX_AI_TURNS} AI turns"
    if category in ACCOUNT_CATEGORIES:
        return ACCOUNT, f"'{category}' needs customer data or an account action"
    if category != "closing" and cls.get("confidence", 1.0) < MIN_CLASSIFIER_CONFIDENCE:
        return KNOWLEDGE, "low classifier confidence: try the knowledge base (escalates if nothing relevant)"
    return KNOWLEDGE, f"'{category}' can be answered from the knowledge base"


class RouteDecision(BaseModel):
    """Which specialist should handle the ticket."""

    route: Literal["knowledge_agent", "account_agent", "escalation_agent"]
    reason: str = Field(description="One short sentence.")


LLM_ROUTER_PROMPT = """You are the routing supervisor of UDA-Hub (CultPass customer support).
Choose ONE specialist for the customer's latest message:
- knowledge_agent: how-to and policy questions answerable from help-center articles (login help, app issues, rules).
- account_agent: anything needing this customer's data or an action on their account (reservations, events,
  subscription, plan changes, profile, refund requests).
- escalation_agent: a human is needed (angry/urgent customers, complex or sensitive cases, repeated failures).

Ticket: turn {turn}, classification {classification}, platform metadata {metadata}
{customer}"""


def llm_route(state: dict, llm: LLMProvider) -> tuple[str, str]:
    """Strategy ``llm``: the LLM chooses the route (hard policies still apply first)."""
    forced = policy_route(state)
    if forced:
        return forced
    prompt = SystemMessage(
        LLM_ROUTER_PROMPT.format(
            turn=state.get("turn"),
            classification=state.get("classification"),
            metadata=state.get("ticket_metadata") or {},
            customer=format_customer(state),
        )
    )
    decision = llm().with_structured_output(RouteDecision, method="function_calling").invoke(
        [prompt, HumanMessage(latest_user_message(state))]
    )
    if not isinstance(decision, RouteDecision):
        decision = RouteDecision.model_validate(decision)
    return decision.route, f"llm router: {decision.reason}"


class Supervisor:
    name = "supervisor"

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm

    def __call__(self, state: dict, config: RunnableConfig = None) -> dict:  # type: ignore[assignment]
        ticket_id = state.get("ticket_id") or ""
        variant = assign_variant(ticket_id, strategy_setting(config))
        source = variant
        if variant == "llm" and self.llm is not None:
            try:
                route, reason = llm_route(state, self.llm)
            except Exception as exc:  # noqa: BLE001 - fall back to rules
                route, reason = decide_route(state)
                source = f"rules (llm router failed: {type(exc).__name__})"
        else:
            route, reason = decide_route(state)
        priority, score, why = compute_priority(state)
        record = log_event(
            "routing",
            ticket_id,
            self.name,
            route=route,
            reason=reason,
            category=(state.get("classification") or {}).get("category"),
            experiment=EXPERIMENT_NAME,
            variant=variant,
            strategy_used=source,
            priority=priority,
            priority_score=score,
            priority_factors=why,
        )
        return {
            "route": route,
            "route_reason": reason,
            "priority": priority,
            "experiment": {"name": EXPERIMENT_NAME, "variant": variant},
            "trace": [record],
        }


def route_from_supervisor(state: dict) -> str:
    return state["route"]
