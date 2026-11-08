"""A/B testing of routing strategies.

* **Assignment** - ``assign_variant(ticket_id, setting)``: with setting ``ab`` each
  ticket is deterministically hashed into ``rules`` or ``llm`` (50/50, sticky for
  the whole ticket). ``rules`` / ``llm`` force one strategy. The setting comes from
  ``config["configurable"]["routing_strategy"]`` or ``UDAHUB_ROUTING_STRATEGY``
  (default ``rules``).
* **Online measurement** - every routing event logs ``experiment`` and ``variant``;
  ``ab_report()`` joins them with ticket outcomes from the structured log.
* **Offline evaluation** - ``agentic/evaluation.py`` replays a labelled ticket set
  through each strategy and scores routing accuracy and outcomes.
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict

EXPERIMENT_NAME = "routing_strategy_v1"
VARIANTS = ("rules", "llm")


def strategy_setting(config: dict | None = None) -> str:
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    return (configurable.get("routing_strategy") or os.getenv("UDAHUB_ROUTING_STRATEGY", "rules")).lower()


def assign_variant(ticket_id: str, setting: str = "rules") -> str:
    if setting in VARIANTS:
        return setting
    if setting == "ab":
        digest = hashlib.sha256(f"{EXPERIMENT_NAME}:{ticket_id}".encode()).hexdigest()
        return VARIANTS[int(digest, 16) % len(VARIANTS)]
    return "rules"


def ab_report(events: list[dict] | None = None) -> dict:
    """Per-variant outcome metrics computed from the structured agent log."""
    if events is None:
        from agentic.logging_utils import read_events

        events = read_events()
    tickets: dict[str, dict] = defaultdict(lambda: {"turns": 0, "tool_calls": 0, "routes": [], "variant": None,
                                                     "final_status": None, "escalated": False})
    for ev in events:
        t = tickets[ev.get("ticket_id")]
        kind = ev.get("event")
        if kind == "intake":
            t["turns"] += 1
        elif kind == "routing":
            t["variant"] = t["variant"] or ev.get("variant")
            t["routes"].append(ev.get("route"))
        elif kind == "tool_call":
            t["tool_calls"] += 1
        elif kind == "escalation":
            t["escalated"] = True
        elif kind == "ticket_update":
            t["final_status"] = ev.get("status")

    report: dict = {}
    for variant in VARIANTS:
        rows = [t for t in tickets.values() if t["variant"] == variant]
        n = len(rows)
        turns = sum(t["turns"] for t in rows)
        report[variant] = {
            "tickets": n,
            "turns": turns,
            "resolution_rate": round(sum(t["final_status"] == "resolved" for t in rows) / n, 3) if n else None,
            "escalation_rate": round(sum(t["escalated"] for t in rows) / n, 3) if n else None,
            "avg_turns_per_ticket": round(turns / n, 2) if n else None,
            "tool_calls_per_turn": round(sum(t["tool_calls"] for t in rows) / turns, 2) if turns else None,
            "route_mix": dict(Counter(r for t in rows for r in t["routes"])),
        }
    return report
