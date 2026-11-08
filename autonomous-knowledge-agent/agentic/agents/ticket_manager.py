"""Ticket manager: the first and last node of every turn (memory + persistence).

``TicketIntake`` (first node)
  * resolves the ticket (thread_id == ticket_id) and the customer behind it,
    creating the ticket in UDA-Hub when it does not exist yet;
  * logs the customer message on the ticket;
  * loads fresh customer context from CultPass and recalls long-term memory;
  * resets the per-turn working fields of the state.

``TicketFinalizer`` (last node)
  * logs the AI reply on the ticket, updates status / issue type / tags;
  * writes long-term memory (interaction summary + stated preferences).
"""
from __future__ import annotations

from datetime import date

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agentic import config as app_config
from agentic.agents.common import first_name, latest_user_message
from agentic.channels import channel_of, format_reply
from agentic.logging_utils import log_event
from agentic.tools.mcp_client import call_tool

GUEST_ID = "guest"


class TicketIntake:
    name = "ticket_intake"

    def __call__(self, state: dict, config: RunnableConfig) -> dict:
        configurable = (config or {}).get("configurable", {})
        ticket_id = state.get("ticket_id") or configurable.get("thread_id")
        if not ticket_id:
            raise ValueError("A thread_id (used as ticket_id) is required in config['configurable'].")
        account_id = state.get("account_id") or configurable.get("account_id") or app_config.DEFAULT_ACCOUNT_ID
        external_user_id = state.get("external_user_id") or configurable.get("external_user_id")
        turn = (state.get("turn") or 0) + 1

        ticket = call_tool("udahub", "get_ticket", {"ticket_id": ticket_id})
        prior_messages: list[dict] = state.get("prior_ticket_messages") or []
        created = False
        if ticket.get("ok"):
            account_id = ticket["account_id"]
            if ticket["external_user_id"] != GUEST_ID:
                external_user_id = ticket["external_user_id"]
            if turn == 1:  # ticket existed before this conversation thread
                prior_messages = ticket["messages"]
            status = ticket["status"] or "open"
        else:
            profile = (
                call_tool("cultpass", "get_customer_profile", {"user_id": external_user_id})
                if external_user_id
                else {"ok": False}
            )
            created_ticket = call_tool(
                "udahub",
                "create_ticket",
                {
                    "account_id": account_id,
                    "external_user_id": external_user_id or GUEST_ID,
                    "user_name": profile.get("full_name", GUEST_ID),
                    "ticket_id": ticket_id,
                    "channel": (state.get("ticket_metadata") or {}).get("channel")
                    or configurable.get("channel", "chat"),
                },
            )
            if not created_ticket.get("ok"):
                raise RuntimeError(f"Could not create ticket {ticket_id}: {created_ticket}")
            status, created = "open", True

        customer = (
            call_tool("cultpass", "get_customer_profile", {"user_id": external_user_id})
            if external_user_id
            else {"ok": False, "error": "no_customer"}
        )
        metadata = state.get("ticket_metadata") or {}
        channel = ticket.get("channel") if ticket.get("ok") else None
        channel = channel or metadata.get("channel") or configurable.get("channel") or "chat"
        user_msg = latest_user_message(state)
        call_tool("udahub", "add_ticket_message", {"ticket_id": ticket_id, "role": "user", "content": user_msg})
        if status == "resolved":  # customer came back: reopen
            call_tool("udahub", "update_ticket_metadata", {"ticket_id": ticket_id, "status": "in_progress"})
            status = "in_progress"

        memories = (
            call_tool(
                "udahub",
                "search_memories",
                {"account_id": account_id, "external_user_id": external_user_id, "query": user_msg, "k": 5},
            ).get("memories", [])
            if external_user_id
            else []
        )
        record = log_event(
            "intake",
            ticket_id,
            self.name,
            turn=turn,
            ticket_metadata=state.get("ticket_metadata") or {},
            ticket_created=created,
            external_user_id=external_user_id,
            ticket_status=status,
            memories_recalled=len(memories),
        )
        return {
            "ticket_id": ticket_id,
            "account_id": account_id,
            "external_user_id": external_user_id,
            "user_name": customer.get("full_name", GUEST_ID),
            "channel": channel,
            "turn": turn,
            "ticket_status": status,
            "customer_context": customer,
            "long_term_memory": memories,
            "prior_ticket_messages": prior_messages,
            # reset per-turn fields
            "classification": {},
            "route": "",
            "route_reason": "",
            "retrieved_articles": [],
            "scratchpad": [],
            "escalation_reason": None,
            "resolution": {},
            "awaiting_confirmation": state.get("pending_action"),
            "pending_action": None,
            "trace": [record],
        }


class TicketFinalizer:
    name = "ticket_finalizer"

    def __call__(self, state: dict) -> dict:
        ticket_id = state["ticket_id"]
        cls = state.get("classification") or {}
        resolution: dict = state.get("resolution") or {"status": "in_progress", "agent": "unknown"}
        status = resolution.get("status", "in_progress")

        updates: dict = {}
        reply = next((m for m in reversed(state.get("messages", [])) if isinstance(m, AIMessage)), None)
        if reply is not None:
            # last-mile formatting for the channel (e-mail greeting/sign-off, public social redaction)
            formatted = format_reply(reply.content, channel_of(state), first_name(state))
            if formatted != reply.content:
                updates["messages"] = [AIMessage(formatted, name=reply.name, id=reply.id)]  # same id -> replaced
            call_tool("udahub", "add_ticket_message", {"ticket_id": ticket_id, "role": "ai", "content": formatted})
        call_tool(
            "udahub",
            "update_ticket_metadata",
            {
                "ticket_id": ticket_id,
                "status": status,
                "main_issue_type": cls.get("category"),
                "tags": cls.get("tags") or [],
                "priority": state.get("priority"),
                "assigned_queue": resolution.get("queue"),
            },
        )

        saved = 0
        ext = state.get("external_user_id")
        if ext:
            outcome = {
                "resolved": f"resolved by {resolution.get('agent')}",
                "escalated": f"escalated to a human ({resolution.get('reason')})",
            }.get(status, status)
            actions: list[dict] = list(resolution.get("actions") or [])
            if actions:
                outcome += "; actions: " + ", ".join(
                    f"{a['tool']}({', '.join(f'{k}={v}' for k, v in (a.get('args') or {}).items())})" for a in actions
                )
            summary = f"{date.today().isoformat()} [{cls.get('category')}] {cls.get('summary', '')} -> {outcome}"
            base = {"account_id": state["account_id"], "external_user_id": ext, "ticket_id": ticket_id}
            if cls.get("category") != "closing":
                call_tool("udahub", "save_memory", {**base, "content": summary, "memory_type": "interaction", "tags": cls.get("tags") or []})
                saved += 1
            for pref in cls.get("customer_preferences") or []:
                call_tool("udahub", "save_memory", {**base, "content": pref, "memory_type": "preference"})
                saved += 1

        record = log_event(
            "ticket_update",
            ticket_id,
            self.name,
            status=status,
            issue_type=cls.get("category"),
            handled_by=resolution.get("agent"),
            priority=state.get("priority"),
            queue=resolution.get("queue"),
            channel=channel_of(state),
            variant=(state.get("experiment") or {}).get("variant"),
            memories_saved=saved,
        )
        return {**updates, "ticket_status": status, "trace": [record]}
