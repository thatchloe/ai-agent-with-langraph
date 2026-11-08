"""Structured, searchable logging of agent decisions.

Every routing decision, classification, retrieval, tool call and escalation is
written as one JSON object per line to ``logs/agent_events.jsonl`` (path set by
``UDAHUB_LOG_PATH``) and also returned so it can be stored in the graph state
(``state["trace"]``) for inspection via ``orchestrator.get_state(...)``.

Example search:  ``grep '"event": "routing"' logs/agent_events.jsonl``
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from agentic import config

logger = logging.getLogger("udahub")


def log_event(event: str, ticket_id: str | None, agent: str, **data: Any) -> dict:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "ticket_id": ticket_id,
        "agent": agent,
        **data,
    }
    try:
        path = config.log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:  # logging must never break a conversation
        logger.warning("could not write agent log: %s", exc)
    logger.info("%s | %s | %s", event, agent, json.dumps(data, default=str)[:300])
    return record


def read_events(ticket_id: str | None = None, event: str | None = None) -> list[dict]:
    """Load logged events, optionally filtered by ticket and/or event type."""
    path = config.log_path()
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if ticket_id and rec.get("ticket_id") != ticket_id:
                continue
            if event and rec.get("event") != event:
                continue
            out.append(rec)
    return out
