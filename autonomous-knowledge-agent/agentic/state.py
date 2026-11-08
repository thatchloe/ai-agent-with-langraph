"""Shared LangGraph state for the UDA-Hub workflow."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class UdaHubState(TypedDict, total=False):
    # --- conversation (short-term memory, persisted per thread_id by the checkpointer)
    messages: Annotated[list[AnyMessage], add_messages]

    # --- ticket / customer identity (set on the first turn, kept across turns)
    ticket_id: str
    account_id: str
    external_user_id: Optional[str]
    user_name: str
    turn: int
    ticket_status: str                      # open | in_progress | resolved | escalated
    ticket_metadata: dict[str, Any]         # input metadata, e.g. {"channel": "email", "urgency": "high"}

    # --- context loaded every turn
    customer_context: dict[str, Any]       # CultPass profile + subscription
    long_term_memory: list[dict[str, Any]]  # recalled CustomerMemory rows
    prior_ticket_messages: list[dict[str, Any]]  # messages logged on the ticket before this thread

    # --- per-turn working fields (reset by the ingest node)
    classification: dict[str, Any]
    route: str
    route_reason: str
    retrieved_articles: list[dict[str, Any]]
    scratchpad: list[AnyMessage]            # tool-calling loop of the account agent
    escalation_reason: Optional[str]
    pending_action: Optional[dict[str, Any]]       # change previewed this turn, waiting for confirmation
    awaiting_confirmation: Optional[dict[str, Any]]  # pending_action carried over from the previous turn
    channel: str                            # chat | email | social
    priority: str                           # P1..P4 (sentiment / urgency / tier based)
    experiment: dict[str, Any]              # A/B test assignment {name, variant}
    resolution: dict[str, Any]              # {status, agent, confidence, ...}

    # --- observability: every decision made in this thread
    trace: Annotated[list[dict[str, Any]], operator.add]
