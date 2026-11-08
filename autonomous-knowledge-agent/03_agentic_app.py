"""UDA-Hub - run the multi-agent customer-support app from the command line.

Run from the ``solution/`` folder (after the two setup notebooks):

    python 03_agentic_app.py --demo                        # scripted scenarios, prints routing + outcome
    python 03_agentic_app.py --user f556c0                 # interactive chat as Bob Stone (new ticket)
    python 03_agentic_app.py --ticket <existing-ticket-id> # continue an existing ticket
    python 03_agentic_app.py --list                        # show customers and tickets

Requires VOCAREUM_API_KEY or OPENAI_API_KEY in ``.env`` (see ``.env.example``).
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from langchain_core.messages import HumanMessage  # noqa: E402

from agentic import config  # noqa: E402

DEMO = [
    # (customer id, messages in one ticket, what it demonstrates)
    ("88382b", ["The QR code for my event is not showing in the app"], "knowledge base (RAG) answer"),
    ("f556c0", ["Which reservations do I have?", "Please also book Samba Night at Lapa for me"], "account tools + action, multi-turn"),
    ("88382b", ["Book the Carnival History Tour in Olinda please"], "business rule: premium event on basic tier"),
    ("a4ab87", ["I can't log in to my account, can you unblock me?"], "blocked account -> escalation"),
    ("f556c0", ["I was charged twice this month and want a refund now!"], "refund filed -> billing escalation"),
    ("88382b", ["Please pause my subscription", "Yes, go ahead"], "subscription change with confirmation"),
    ("e6376d", ["Can you recommend a good pizza recipe?"], "no relevant article -> escalation"),
    ("f556c0", ["Any music events coming up that I'd like?", "Thanks, that's all!"], "long-term memory + closing"),
]


def _check_setup() -> None:
    missing = [p for p in (config.udahub_db_path(), config.cultpass_db_path()) if not p.exists()]
    if missing:
        sys.exit(f"Missing database(s): {missing}. Run 01_external_db_setup.ipynb and 02_core_db_setup.ipynb first.")
    if not config.has_llm_credentials():
        sys.exit("No LLM credentials. Copy .env.example to .env and set VOCAREUM_API_KEY or OPENAI_API_KEY.")


def run_demo() -> None:
    from agentic.workflow import orchestrator

    for user_id, messages, purpose in DEMO:
        ticket_id = f"demo-{uuid.uuid4().hex[:8]}"
        print(f"\n=== {purpose} | customer {user_id} | ticket {ticket_id}")
        for i, text in enumerate(messages):
            payload: dict = {"messages": [HumanMessage(text)]}
            if i == 0:
                payload["external_user_id"] = user_id
            state = orchestrator.invoke(payload, config={"configurable": {"thread_id": ticket_id}})
            cls = state.get("classification") or {}
            print(f"User: {text}")
            print(f"Assistant ({state['messages'][-1].name}): {state['messages'][-1].content}")
            print(
                f"  -> category={cls.get('category')} route={state.get('route')} "
                f"status={state.get('ticket_status')} memories_recalled={len(state.get('long_term_memory') or [])}"
            )
    print(f"\nStructured decision log: {config.log_path()}")


def list_entities() -> None:
    from agentic.tools.db import cultpass_engine, session_scope, udahub_engine
    from data.models import cultpass, udahub

    with session_scope(cultpass_engine()) as s:
        print("CultPass customers:")
        for u in s.query(cultpass.User).all():
            sub = u.subscription
            print(f"  {u.user_id}  {u.full_name:16s} blocked={u.is_blocked!s:5s} "
                  f"subscription={sub.status}/{sub.tier} quota={sub.monthly_quota}")
    with session_scope(udahub_engine()) as s:
        print("UDA-Hub tickets:")
        for t in s.query(udahub.Ticket).all():
            meta = t.ticket_metadata
            print(f"  {t.ticket_id}  user={t.user.external_user_id} status={meta.status if meta else None}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="run the scripted demo scenarios")
    parser.add_argument("--user", help="CultPass customer id for a new ticket (e.g. f556c0)")
    parser.add_argument("--ticket", help="ticket id / thread id (new or existing)")
    parser.add_argument("--list", action="store_true", help="list customers and tickets")
    args = parser.parse_args()

    if args.list:
        list_entities()
        return
    _check_setup()
    if args.demo:
        run_demo()
        return

    from agentic.workflow import orchestrator
    from utils import chat_interface

    ticket_id = args.ticket or f"chat-{uuid.uuid4().hex[:8]}"
    chat_interface(orchestrator, ticket_id, external_user_id=args.user)
    snapshot = orchestrator.get_state({"configurable": {"thread_id": ticket_id}})
    if snapshot.values:
        print("\nDecision trace for this ticket:")
        for rec in snapshot.values.get("trace", []):
            print("  " + json.dumps({k: v for k, v in rec.items() if k != "ts"}, default=str)[:200])


if __name__ == "__main__":
    main()
