"""Deterministic test databases (fixed IDs/dates) built from the same models and
JSONL files the setup notebooks use."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agentic.config import DATA_DIR
from data.models import cultpass, udahub

EXTERNAL = DATA_DIR / "external"

PERSONAS = {
    "a4ab87": dict(status="active", tier="premium", monthly_quota=4),   # Alice (blocked)
    "f556c0": dict(status="active", tier="premium", monthly_quota=8),   # Bob
    "88382b": dict(status="active", tier="basic", monthly_quota=4),     # Cathy
    "888fb2": dict(status="active", tier="basic", monthly_quota=4),     # David (blocked)
    "f1f10d": dict(status="cancelled", tier="basic", monthly_quota=4),  # Eva
    "e6376d": dict(status="active", tier="basic", monthly_quota=2),     # Frank
}
# experience ids: exp-0 .. exp-6 (from jsonl, exp-i is premium when i is even),
# exp-soon (in 3 hours, standard), exp-full (sold out, standard)
ALICE_TICKET = "t-alice-login"
BOB_PAST_TICKET = "t-bob-past"


def _load(name: str) -> list[dict]:
    with open(EXTERNAL / name, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _session(path: Path, base):
    engine = create_engine(f"sqlite:///{path}")
    base.metadata.create_all(engine)
    return sessionmaker(bind=engine)(), engine


def seed_cultpass(path: Path) -> None:
    session, engine = _session(path, cultpass.Base)
    now = datetime.now()
    for i, exp in enumerate(_load("cultpass_experiences.jsonl")):
        session.add(
            cultpass.Experience(
                experience_id=f"exp-{i}",
                title=exp["title"],
                description=exp["description"],
                location=exp["location"],
                when=now + timedelta(days=i + 2),
                slots_available=10,
                is_premium=(i % 2 == 0),
            )
        )
    session.add(cultpass.Experience(experience_id="exp-soon", title="Jazz Brunch in Pinheiros",
                                    description="Live jazz with brunch.", location="São Paulo, Brazil",
                                    when=now + timedelta(hours=3), slots_available=5, is_premium=False))
    session.add(cultpass.Experience(experience_id="exp-full", title="Sold Out Opera Night",
                                    description="Opera gala.", location="São Paulo, Brazil",
                                    when=now + timedelta(days=5), slots_available=0, is_premium=False))
    for user in _load("cultpass_users.jsonl"):
        session.add(cultpass.User(user_id=user["id"], full_name=user["name"], email=user["email"],
                                  is_blocked=user["is_blocked"]))
        session.add(cultpass.Subscription(subscription_id=f"sub-{user['id']}", user_id=user["id"],
                                          started_at=now, **PERSONAS[user["id"]]))
    for rid, uid, eid in [
        ("r-alice-1", "a4ab87", "exp-1"),
        ("r-bob-1", "f556c0", "exp-6"),      # Ibirapuera Park Bike Ride (premium)
        ("r-bob-2", "f556c0", "exp-5"),      # Modern Art at MASP
        ("r-bob-late", "f556c0", "exp-soon"),
        ("r-frank-1", "e6376d", "exp-3"),    # Samba Night at Lapa
    ]:
        session.add(cultpass.Reservation(reservation_id=rid, user_id=uid, experience_id=eid,
                                         status="reserved", created_at=now))
    session.commit()
    session.close()
    engine.dispose()


def seed_udahub(path: Path) -> None:
    session, engine = _session(path, udahub.Base)
    session.add(udahub.Account(account_id="cultpass", account_name="CultPass Card"))
    for i, art in enumerate(_load("cultpass_articles.jsonl")):
        session.add(udahub.Knowledge(article_id=f"kb-{i}", account_id="cultpass", title=art["title"],
                                     content=art["content"], tags=art["tags"]))
    alice = udahub.User(user_id="u-alice", account_id="cultpass", external_user_id="a4ab87", user_name="Alice Kingsley")
    bob = udahub.User(user_id="u-bob", account_id="cultpass", external_user_id="f556c0", user_name="Bob Stone")
    past = datetime.now() - timedelta(days=20)
    session.add_all([
        alice, bob,
        udahub.Ticket(ticket_id=ALICE_TICKET, account_id="cultpass", user_id="u-alice", channel="chat"),
        udahub.TicketMetadata(ticket_id=ALICE_TICKET, status="open", tags="login, access"),
        udahub.TicketMessage(message_id=str(uuid.uuid4()), ticket_id=ALICE_TICKET, role="user",
                             content="I can't log in to my Cultpass account."),
        udahub.Ticket(ticket_id=BOB_PAST_TICKET, account_id="cultpass", user_id="u-bob", channel="email", created_at=past),
        udahub.TicketMetadata(ticket_id=BOB_PAST_TICKET, status="resolved", main_issue_type="experience_info"),
        udahub.TicketMessage(message_id=str(uuid.uuid4()), ticket_id=BOB_PAST_TICKET, role="user",
                             content="Are premium experiences included in my plan?", created_at=past),
        udahub.CustomerMemory(memory_id="m-bob-1", account_id="cultpass", user_id="u-bob", ticket_id=BOB_PAST_TICKET,
                              memory_type="interaction", created_at=past,
                              content="[experience_info] Asked whether premium experiences are included -> resolved"),
        udahub.CustomerMemory(memory_id="m-bob-2", account_id="cultpass", user_id="u-bob", ticket_id=BOB_PAST_TICKET,
                              memory_type="preference", created_at=past,
                              content="Enjoys samba / music events in Rio de Janeiro"),
    ])
    session.commit()
    session.close()
    engine.dispose()
