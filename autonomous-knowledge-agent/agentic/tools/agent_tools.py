"""LangChain tools handed to the Account Operations agent.

Tools are built *per request* and bound to the customer that owns the ticket:
the LLM never receives a ``user_id`` parameter, so it cannot read or modify
another customer's data even if prompted to. Each tool forwards to an MCP
server through ``mcp_client.call_tool``.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from agentic.tools.mcp_client import call_tool

ESCALATION_TOOL_NAME = "escalate_to_human"
REFUND_TOOL_NAME = "request_refund"
CONFIRMABLE_TOOLS = {"pause_my_subscription", "resume_my_subscription", "cancel_my_subscription", "change_my_plan"}


def build_account_tools(account_id: str, external_user_id: str | None) -> list[BaseTool]:
    def _needs_customer() -> dict | None:
        if not external_user_id:
            return {"ok": False, "error": "no_customer", "message": "This conversation is not linked to a CultPass customer."}
        return None

    def get_my_account() -> dict:
        """Get the customer's profile: name, e-mail, blocked status, subscription (status, tier, monthly quota, remaining quota)."""
        return _needs_customer() or call_tool("cultpass", "get_customer_profile", {"user_id": external_user_id})

    def list_my_reservations(include_cancelled: bool = False) -> dict:
        """List the customer's reservations (reservation_id, experience, date, location)."""
        return _needs_customer() or call_tool(
            "cultpass", "list_reservations", {"user_id": external_user_id, "include_cancelled": include_cancelled}
        )

    def search_experiences(query: str = "", location: str = "") -> dict:
        """Search upcoming experiences with free slots. query: keywords from the title/description (e.g. 'samba', 'museum'). location: optional city or state (e.g. 'Rio de Janeiro'); leave empty when unsure."""
        return call_tool("cultpass", "search_experiences", {"query": query, "location": location})

    def reserve_experience(experience_id: str) -> dict:
        """Reserve an experience for the customer. Use the experience_id from search_experiences. Only call when the customer clearly asked to book it."""
        return _needs_customer() or call_tool(
            "cultpass", "reserve_experience", {"user_id": external_user_id, "experience_id": experience_id}
        )

    def cancel_reservation(reservation_id: str) -> dict:
        """Cancel one of the customer's reservations. Use the reservation_id from list_my_reservations. Only call when the customer clearly asked to cancel it."""
        return _needs_customer() or call_tool(
            "cultpass", "cancel_reservation", {"user_id": external_user_id, "reservation_id": reservation_id}
        )

    # --- subscription management (two-step: call without confirm, ask the customer, then confirm=True)
    def pause_my_subscription(confirm: bool = False) -> dict:
        """Pause the customer's subscription. First call with confirm=false to get a preview; call with confirm=true only after the customer explicitly confirmed in a later message."""
        return _needs_customer() or call_tool("cultpass", "pause_subscription", {"user_id": external_user_id, "confirm": confirm})

    def resume_my_subscription(confirm: bool = False) -> dict:
        """Reactivate a paused or cancelled subscription. Preview with confirm=false; confirm=true only after explicit customer confirmation."""
        return _needs_customer() or call_tool("cultpass", "resume_subscription", {"user_id": external_user_id, "confirm": confirm})

    def cancel_my_subscription(confirm: bool = False) -> dict:
        """Cancel the subscription at the end of the billing cycle (no refund). Preview with confirm=false; confirm=true only after explicit customer confirmation."""
        return _needs_customer() or call_tool("cultpass", "cancel_subscription", {"user_id": external_user_id, "confirm": confirm})

    def change_my_plan(new_tier: str, confirm: bool = False) -> dict:
        """Upgrade/downgrade the plan: new_tier is 'basic' or 'premium'. Preview with confirm=false; confirm=true only after explicit customer confirmation."""
        return _needs_customer() or call_tool(
            "cultpass", "change_subscription_tier", {"user_id": external_user_id, "new_tier": new_tier, "confirm": confirm}
        )

    # --- refunds (filed for human approval; never approved by the AI)
    def request_refund(reason: str, amount: float | None = None, reservation_id: str | None = None) -> dict:
        """File a refund request for the billing team (status pending_approval). Include the customer's reason, and the amount / reservation_id if they gave them. The ticket is then escalated automatically."""
        return _needs_customer() or call_tool(
            "cultpass", "request_refund",
            {"user_id": external_user_id, "reason": reason, "amount": amount, "reservation_id": reservation_id},
        )

    def list_my_refund_requests() -> dict:
        """List the customer's refund requests and their status."""
        return _needs_customer() or call_tool("cultpass", "list_refund_requests", {"user_id": external_user_id})

    # --- account modifications
    def update_my_name(full_name: str) -> dict:
        """Change the customer's display name."""
        return _needs_customer() or call_tool("cultpass", "update_profile", {"user_id": external_user_id, "full_name": full_name})

    def request_email_change(new_email: str) -> dict:
        """Start a login e-mail change: sends a confirmation link to the new address (the change applies after the customer clicks it)."""
        return _needs_customer() or call_tool(
            "cultpass", "request_email_change", {"user_id": external_user_id, "new_email": new_email}
        )

    def search_knowledge_base(query: str) -> dict:
        """Search CultPass support policies / how-to articles (subscription rules, cancellation policy, premium events, etc.)."""
        result = call_tool("udahub", "search_knowledge", {"query": query, "account_id": account_id, "k": 2})
        return result

    def get_my_past_tickets() -> dict:
        """Get summaries of the customer's previous support tickets."""
        return _needs_customer() or call_tool(
            "udahub", "get_customer_ticket_history", {"account_id": account_id, "external_user_id": external_user_id}
        )

    def escalate_to_human(reason: str) -> dict:
        """Hand the ticket to a human support agent. Use when a request needs approval (refunds, late cancellations, unblocking an account), when tools fail repeatedly, or when the customer asks for a person."""
        return {"ok": True, "escalated": True, "reason": reason}

    fns = [
        get_my_account,
        list_my_reservations,
        search_experiences,
        reserve_experience,
        cancel_reservation,
        pause_my_subscription,
        resume_my_subscription,
        cancel_my_subscription,
        change_my_plan,
        request_refund,
        list_my_refund_requests,
        update_my_name,
        request_email_change,
        search_knowledge_base,
        get_my_past_tickets,
        escalate_to_human,
    ]
    return [StructuredTool.from_function(fn) for fn in fns]
