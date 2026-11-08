"""Tools that abstract the core UDA-Hub database.

Covers: knowledge-base retrieval (RAG), ticket lifecycle (create, message log,
metadata/status) and long-term customer memory. Exposed through the ``udahub``
MCP server (``agentic/tools/mcp_servers/udahub_server.py``).
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime

from sqlalchemy import func

from agentic import config
from agentic.tools.db import session_scope, to_dict, udahub_engine
from agentic.tools.retrieval import Document, HybridRetriever, embedding_text
from data.models import udahub

VALID_STATUSES = {"open", "in_progress", "resolved", "escalated"}
VALID_PRIORITIES = {"P1", "P2", "P3", "P4"}
VALID_QUEUES = {"billing", "trust_safety", "technical", "general"}
_INDEX_CACHE: dict[tuple, HybridRetriever] = {}


# --------------------------------------------------------------------------- #
# Knowledge base (RAG retrieval)
# --------------------------------------------------------------------------- #
def _knowledge_index(account_id: str) -> HybridRetriever:
    with session_scope(udahub_engine()) as session:
        count, last_update = (
            session.query(func.count(udahub.Knowledge.article_id), func.max(udahub.Knowledge.updated_at))
            .filter(udahub.Knowledge.account_id == account_id)
            .one()
        )
        embeddings = config.get_embeddings()
        key = (str(config.udahub_db_path()), account_id, count, str(last_update), embeddings is not None)
        if key not in _INDEX_CACHE:
            rows = session.query(udahub.Knowledge).filter_by(account_id=account_id).all()
            docs = [Document(r.article_id, r.title, r.content, r.tags or "") for r in rows]
            vectors = _load_or_create_embeddings(session, docs, embeddings) if embeddings is not None else None
            _INDEX_CACHE[key] = HybridRetriever(docs, embeddings=embeddings, doc_embeddings=vectors)
        return _INDEX_CACHE[key]


def _load_or_create_embeddings(session, docs: list[Document], embeddings) -> list[list[float]]:
    """Read article vectors from the ``knowledge_embeddings`` table; embed only new/changed articles."""
    model = config.embedding_model()
    stored = {
        row.article_id: row
        for row in session.query(udahub.KnowledgeEmbedding).filter(
            udahub.KnowledgeEmbedding.model == model,
            udahub.KnowledgeEmbedding.article_id.in_([d.doc_id for d in docs]),
        )
    }
    hashes = {d.doc_id: hashlib.sha256(embedding_text(d).encode()).hexdigest() for d in docs}
    missing = [d for d in docs if d.doc_id not in stored or stored[d.doc_id].content_hash != hashes[d.doc_id]]
    if missing:
        new_vectors = embeddings.embed_documents([embedding_text(d) for d in missing])
        for doc, vec in zip(missing, new_vectors):
            row = stored.get(doc.doc_id) or udahub.KnowledgeEmbedding(article_id=doc.doc_id, model=model)
            row.content_hash, row.vector = hashes[doc.doc_id], json.dumps(vec)
            session.add(row)
            stored[doc.doc_id] = row
    return [json.loads(stored[d.doc_id].vector) for d in docs]


def search_knowledge(query: str, account_id: str = config.DEFAULT_ACCOUNT_ID, k: int = 3) -> dict:
    """Return the k most relevant knowledge-base articles for a query with relevance scores."""
    index = _knowledge_index(account_id)
    results = index.search(query, k=k)
    return {
        "ok": True,
        "query": query,
        "mode": index.mode,
        "min_relevance": index.threshold,
        "results": [
            {"article_id": d.doc_id, "title": d.title, "content": d.content, "tags": d.tags, "score": s}
            for d, s in results
        ],
    }


# --------------------------------------------------------------------------- #
# Users & tickets
# --------------------------------------------------------------------------- #
def _get_or_create_user(session, account_id: str, external_user_id: str, user_name: str) -> udahub.User:
    user = (
        session.query(udahub.User)
        .filter_by(account_id=account_id, external_user_id=external_user_id)
        .first()
    )
    if user is None:
        user = udahub.User(
            user_id=str(uuid.uuid4()),
            account_id=account_id,
            external_user_id=external_user_id,
            user_name=user_name or external_user_id,
        )
        session.add(user)
        session.flush()
    return user


def get_ticket(ticket_id: str) -> dict:
    """Return ticket details, metadata and full message log (None fields if missing)."""
    with session_scope(udahub_engine()) as session:
        ticket = session.get(udahub.Ticket, ticket_id)
        if ticket is None:
            return {"ok": False, "error": "ticket_not_found", "ticket_id": ticket_id}
        meta = ticket.ticket_metadata
        return {
            "ok": True,
            "ticket_id": ticket.ticket_id,
            "account_id": ticket.account_id,
            "channel": ticket.channel,
            "external_user_id": ticket.user.external_user_id,
            "user_name": ticket.user.user_name,
            "status": meta.status if meta else None,
            "main_issue_type": meta.main_issue_type if meta else None,
            "tags": meta.tags if meta else None,
            "priority": meta.priority if meta else None,
            "assigned_queue": meta.assigned_queue if meta else None,
            "messages": [
                {"role": m.role.value, "content": m.content, "created_at": str(m.created_at)}
                for m in sorted(ticket.messages, key=lambda m: m.created_at)
            ],
        }


def create_ticket(
    account_id: str,
    external_user_id: str,
    user_name: str = "",
    ticket_id: str | None = None,
    channel: str = "chat",
) -> dict:
    """Open a new ticket (creating the UDA-Hub user mapping if needed)."""
    with session_scope(udahub_engine()) as session:
        ticket_id = ticket_id or str(uuid.uuid4())
        if session.get(udahub.Ticket, ticket_id) is not None:
            return {"ok": False, "error": "ticket_exists", "ticket_id": ticket_id}
        if session.get(udahub.Account, account_id) is None:
            return {"ok": False, "error": "account_not_found", "account_id": account_id}
        user = _get_or_create_user(session, account_id, external_user_id, user_name)
        session.add(
            udahub.Ticket(ticket_id=ticket_id, account_id=account_id, user_id=user.user_id, channel=channel)
        )
        session.add(udahub.TicketMetadata(ticket_id=ticket_id, status="open", main_issue_type=None, tags=""))
        return {"ok": True, "ticket_id": ticket_id, "user_id": user.user_id}


def add_ticket_message(ticket_id: str, role: str, content: str) -> dict:
    """Append a message (role: user | ai | agent | system) to a ticket's log."""
    with session_scope(udahub_engine()) as session:
        if session.get(udahub.Ticket, ticket_id) is None:
            return {"ok": False, "error": "ticket_not_found", "ticket_id": ticket_id}
        msg = udahub.TicketMessage(
            message_id=str(uuid.uuid4()),
            ticket_id=ticket_id,
            role=udahub.RoleEnum(role),
            content=content,
            created_at=datetime.now(),
        )
        session.add(msg)
        return {"ok": True, "message_id": msg.message_id}


def update_ticket_metadata(
    ticket_id: str,
    status: str | None = None,
    main_issue_type: str | None = None,
    tags: list[str] | None = None,
    priority: str | None = None,
    assigned_queue: str | None = None,
) -> dict:
    """Update status / issue type / priority / queue and merge tags on a ticket."""
    if status is not None and status not in VALID_STATUSES:
        return {"ok": False, "error": "invalid_status", "allowed": sorted(VALID_STATUSES)}
    if priority is not None and priority not in VALID_PRIORITIES:
        return {"ok": False, "error": "invalid_priority", "allowed": sorted(VALID_PRIORITIES)}
    if assigned_queue is not None and assigned_queue not in VALID_QUEUES:
        return {"ok": False, "error": "invalid_queue", "allowed": sorted(VALID_QUEUES)}
    with session_scope(udahub_engine()) as session:
        meta = session.get(udahub.TicketMetadata, ticket_id)
        if meta is None:
            return {"ok": False, "error": "ticket_not_found", "ticket_id": ticket_id}
        if status:
            meta.status = status
        if main_issue_type:
            meta.main_issue_type = main_issue_type
        if priority:
            meta.priority = priority
        if assigned_queue:
            meta.assigned_queue = assigned_queue
        if tags:
            existing = [t.strip() for t in (meta.tags or "").split(",") if t.strip()]
            for tag in tags:
                tag = tag.strip().lower()
                if tag and tag not in existing:
                    existing.append(tag)
            meta.tags = ", ".join(existing)
        meta.updated_at = datetime.now()
        return {"ok": True, **to_dict(meta)}


def list_tickets(
    account_id: str = config.DEFAULT_ACCOUNT_ID,
    status: str | None = None,
    assigned_queue: str | None = None,
    limit: int = 50,
) -> dict:
    """STAFF: work queue - tickets filtered by status / queue, most urgent (P1) first."""
    with session_scope(udahub_engine()) as session:
        q = session.query(udahub.Ticket, udahub.TicketMetadata).join(udahub.TicketMetadata).filter(
            udahub.Ticket.account_id == account_id
        )
        if status:
            q = q.filter(udahub.TicketMetadata.status == status)
        if assigned_queue:
            q = q.filter(udahub.TicketMetadata.assigned_queue == assigned_queue)
        rows = sorted(q.all(), key=lambda tm: (tm[1].priority or "P9", -(tm[0].created_at or datetime.min).timestamp()))
        return {
            "ok": True,
            "tickets": [
                {
                    "ticket_id": t.ticket_id, "channel": t.channel, "status": m.status, "priority": m.priority,
                    "assigned_queue": m.assigned_queue, "issue_type": m.main_issue_type,
                    "customer": t.user.external_user_id, "created_at": str(t.created_at),
                }
                for t, m in rows[:limit]
            ],
        }


def assign_ticket(ticket_id: str, assigned_queue: str, priority: str | None = None) -> dict:
    """STAFF: (re)assign a ticket to a human queue and optionally change its priority."""
    return update_ticket_metadata(ticket_id, assigned_queue=assigned_queue, priority=priority)


def get_customer_ticket_history(
    account_id: str, external_user_id: str, exclude_ticket_id: str | None = None, limit: int = 5
) -> dict:
    """Summaries of the customer's previous tickets (most recent first)."""
    with session_scope(udahub_engine()) as session:
        user = session.query(udahub.User).filter_by(account_id=account_id, external_user_id=external_user_id).first()
        if user is None:
            return {"ok": True, "tickets": []}
        tickets = sorted(user.tickets, key=lambda t: t.created_at or datetime.min, reverse=True)
        out = []
        for t in tickets:
            if t.ticket_id == exclude_ticket_id:
                continue
            first = min(t.messages, key=lambda m: m.created_at, default=None)
            out.append(
                {
                    "ticket_id": t.ticket_id,
                    "created_at": str(t.created_at),
                    "status": t.ticket_metadata.status if t.ticket_metadata else None,
                    "issue_type": t.ticket_metadata.main_issue_type if t.ticket_metadata else None,
                    "first_message": first.content if first else None,
                }
            )
            if len(out) >= limit:
                break
        return {"ok": True, "tickets": out}


# --------------------------------------------------------------------------- #
# Long-term memory
# --------------------------------------------------------------------------- #
def save_memory(
    account_id: str,
    external_user_id: str,
    content: str,
    memory_type: str = "interaction",
    ticket_id: str | None = None,
    tags: list[str] | None = None,
    user_name: str = "",
) -> dict:
    """Persist a durable fact about the customer (interaction summary or preference)."""
    with session_scope(udahub_engine()) as session:
        user = _get_or_create_user(session, account_id, external_user_id, user_name)
        if memory_type == "preference":
            dup = (
                session.query(udahub.CustomerMemory)
                .filter_by(user_id=user.user_id, memory_type="preference")
                .filter(func.lower(udahub.CustomerMemory.content) == content.lower())
                .first()
            )
            if dup:
                return {"ok": True, "memory_id": dup.memory_id, "deduplicated": True}
        mem = udahub.CustomerMemory(
            memory_id=str(uuid.uuid4()),
            account_id=account_id,
            user_id=user.user_id,
            ticket_id=ticket_id,
            memory_type=memory_type,
            content=content,
            tags=", ".join(tags or []),
            created_at=datetime.now(),
        )
        session.add(mem)
        return {"ok": True, "memory_id": mem.memory_id}


def search_memories(account_id: str, external_user_id: str, query: str = "", k: int = 5) -> dict:
    """Recall long-term memories: all preferences + the most relevant/recent interactions."""
    with session_scope(udahub_engine()) as session:
        user = session.query(udahub.User).filter_by(account_id=account_id, external_user_id=external_user_id).first()
        if user is None:
            return {"ok": True, "memories": []}
        rows = (
            session.query(udahub.CustomerMemory)
            .filter_by(user_id=user.user_id)
            .order_by(udahub.CustomerMemory.created_at.desc())
            .all()
        )
        prefs = [r for r in rows if r.memory_type == "preference"]
        interactions = [r for r in rows if r.memory_type != "preference"]
        if query and interactions:
            docs = [Document(r.memory_id, "", r.content, r.tags or "", {"row": r}) for r in interactions]
            ranked = HybridRetriever(docs).search(query, k=len(docs))
            # relevance first, recency (original order) as tie-breaker
            interactions = [d.metadata["row"] for d, _ in ranked]
        selected = prefs + interactions[: max(k - len(prefs), 2)]
        return {
            "ok": True,
            "memories": [
                {
                    "memory_type": r.memory_type,
                    "content": r.content,
                    "ticket_id": r.ticket_id,
                    "created_at": r.created_at.isoformat(sep=" ", timespec="minutes") if r.created_at else None,
                }
                for r in selected
            ],
        }
