"""Offline A/B evaluation of routing strategies on a labelled ticket set.

Each strategy replays every ticket in ``data/evaluation/labelled_tickets.jsonl``
through the full graph, on its own throw-away copy of the databases (so account
actions in one run cannot influence the other), and is scored on:

* routing accuracy  - first route is one of the ticket's acceptable routes
* outcome accuracy  - final ticket status is one of the acceptable statuses
* escalation rate, tool calls per ticket, latency

Run from ``solution/`` (needs an API key; ~2 x 24 graph runs)::

    python -m agentic.evaluation                      # rules vs llm, writes agentic/design/ab_test_report.md
    python -m agentic.evaluation --limit 5 --out -    # quick run, print only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from agentic import config
from agentic.channels import from_chat, from_email, from_social

DATASET = config.DATA_DIR / "evaluation" / "labelled_tickets.jsonl"
DEFAULT_REPORT = config.SOLUTION_ROOT / "agentic" / "design" / "ab_test_report.md"


def load_dataset(path: Path = DATASET) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@contextmanager
def sandbox_databases():
    """Point the tools at temporary copies of the current databases."""
    tmp = Path(tempfile.mkdtemp(prefix="udahub-eval-"))
    saved = {k: os.environ.get(k) for k in ("UDAHUB_DB_PATH", "CULTPASS_DB_PATH", "UDAHUB_LOG_PATH")}
    try:
        shutil.copy(config.udahub_db_path(), tmp / "udahub.db")
        shutil.copy(config.cultpass_db_path(), tmp / "cultpass.db")
        os.environ.update(
            UDAHUB_DB_PATH=str(tmp / "udahub.db"),
            CULTPASS_DB_PATH=str(tmp / "cultpass.db"),
            UDAHUB_LOG_PATH=str(tmp / "events.jsonl"),
        )
        yield tmp
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(tmp, ignore_errors=True)


def ticket_payload(ticket: dict) -> dict:
    if ticket["channel"] == "email":
        payload = from_email("customer@example.com", "Support request", ticket["text"])
        payload.pop("external_user_id", None)
    elif ticket["channel"] == "social":
        payload = from_social("@someone", ticket["text"])
    else:
        payload = from_chat(ticket["text"])
    if ticket.get("external_user_id"):
        payload["external_user_id"] = ticket["external_user_id"]
    return payload


def run_strategy(strategy: str, tickets: list[dict], llm=None) -> list[dict]:
    from agentic.workflow import build_graph

    rows = []
    with sandbox_databases():
        graph = build_graph(llm=llm)
        for ticket in tickets:
            thread = f"eval-{strategy}-{ticket['id']}"
            started = time.perf_counter()
            error = None
            try:
                state = graph.invoke(
                    ticket_payload(ticket),
                    config={"configurable": {"thread_id": thread, "routing_strategy": strategy}},
                )
            except Exception as exc:  # noqa: BLE001 - a crash is a failed ticket, not a failed evaluation
                state, error = {}, f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            trace = state.get("trace", [])
            first_route = next((e["route"] for e in trace if e.get("event") == "routing"), None)
            status = state.get("ticket_status")
            rows.append(
                {
                    "id": ticket["id"],
                    "text": ticket["text"],
                    "strategy": strategy,
                    "route": first_route,
                    "status": status,
                    "route_ok": first_route in ticket["acceptable_routes"],
                    "outcome_ok": status in ticket["acceptable_statuses"],
                    "escalated": status == "escalated",
                    "tool_calls": sum(e.get("event") == "tool_call" for e in trace),
                    "latency_s": round(elapsed, 2),
                    "error": error,
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "tickets": len(rows),
        "routing_accuracy": round(sum(r["route_ok"] for r in rows) / n, 3),
        "outcome_accuracy": round(sum(r["outcome_ok"] for r in rows) / n, 3),
        "fully_correct": round(sum(r["route_ok"] and r["outcome_ok"] for r in rows) / n, 3),
        "escalation_rate": round(sum(r["escalated"] for r in rows) / n, 3),
        "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 2),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
        "errors": sum(bool(r["error"]) for r in rows),
    }


def evaluate(strategies=("rules", "llm"), tickets: list[dict] | None = None, llm=None) -> dict:
    tickets = tickets if tickets is not None else load_dataset()
    results = {s: run_strategy(s, tickets, llm=llm) for s in strategies}
    return {"summary": {s: summarize(r) for s, r in results.items()}, "rows": results}


def to_markdown(result: dict) -> str:
    summary, rows = result["summary"], result["rows"]
    strategies = list(summary)
    metrics = ["tickets", "routing_accuracy", "outcome_accuracy", "fully_correct", "escalation_rate",
               "avg_tool_calls", "avg_latency_s", "errors"]
    lines = [
        "# A/B test: routing strategies",
        "",
        f"Experiment `routing_strategy_v1` - labelled set `data/evaluation/labelled_tickets.jsonl` "
        f"({summary[strategies[0]]['tickets']} tickets), model `{os.getenv('UDAHUB_MODEL', config.DEFAULT_MODEL)}`.",
        "Generated by `python -m agentic.evaluation`.",
        "",
        "| metric | " + " | ".join(strategies) + " |",
        "|---|" + "---|" * len(strategies),
    ]
    for m in metrics:
        lines.append(f"| {m} | " + " | ".join(str(summary[s][m]) for s in strategies) + " |")
    best = max(strategies, key=lambda s: (summary[s]["fully_correct"], -summary[s]["avg_latency_s"]))
    lines += ["", f"**Winner on this set: `{best}`** (highest share of tickets with correct route and outcome; latency breaks ties).", ""]
    lines += ["## Per-ticket results", "", "| id | ticket | " + " | ".join(f"{s} route / status" for s in strategies) + " |",
              "|---|---|" + "---|" * len(strategies)]
    by_id = {s: {r["id"]: r for r in rows[s]} for s in strategies}
    for rid in [r["id"] for r in rows[strategies[0]]]:
        cells = []
        for s in strategies:
            r = by_id[s][rid]
            mark = "✅" if r["route_ok"] and r["outcome_ok"] else "❌"
            cells.append(f"{mark} {(r['route'] or '-').replace('_agent', '')} / {r['status']}")
        text = by_id[strategies[0]][rid]["text"].replace("|", "/")[:70]
        lines.append(f"| {rid} | {text} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(config.SOLUTION_ROOT / ".env")
    parser = argparse.ArgumentParser(description="A/B evaluation of routing strategies")
    parser.add_argument("--strategies", nargs="+", default=["rules", "llm"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(DEFAULT_REPORT), help="markdown report path, or '-' to only print")
    args = parser.parse_args()
    tickets = load_dataset()[: args.limit] if args.limit else load_dataset()
    result = evaluate(args.strategies, tickets)
    report = to_markdown(result)
    print(report)
    if args.out != "-":
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
