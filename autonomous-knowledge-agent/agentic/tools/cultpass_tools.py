"""Tools that abstract the external CultPass database (customer system of record).

Every function is plain Python, returns JSON-serialisable dicts and never
raises for business-rule failures: it returns ``{"ok": False, "error": ...}``
so an agent can explain the outcome to the customer. These functions are
exposed to agents through the ``cultpass`` MCP server
(``agentic/tools/mcp_servers/cultpass_server.py``).
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta

from sqlalchemy import func, or_

from agentic.tools.db import cultpass_engine, session_scope, to_dict
from data.models import cultpass

LATE_CANCELLATION_WINDOW = timedelta(hours=24)


def _fail(error: str, message: str, **extra) -> dict:
    return {"ok": False, "error": error, "message": message, **extra}


def _reservations_this_month(session, user_id: str) -> int:
    start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return (
        session.query(func.count(cultpass.Reservation.reservation_id))
        .filter(
            cultpass.Reservation.user_id == user_id,
            cultpass.Reservation.status == "reserved",
            cultpass.Reservation.created_at >= start_of_month,
        )
        .scalar()
    )


def get_customer_profile(user_id: str) -> dict:
    """Return a customer's profile, block status, subscription and quota usage."""
    with session_scope(cultpass_engine()) as session:
        user = session.get(cultpass.User, user_id)
        if user is None:
            return _fail("user_not_found", f"No CultPass customer with id '{user_id}'.")
        profile = {
            "ok": True,
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "is_blocked": bool(user.is_blocked),
            "subscription": None,
        }
        if user.subscription is not None:
            sub = to_dict(user.subscription)
            used = _reservations_this_month(session, user_id)
            sub["reservations_this_month"] = used
            sub["remaining_quota"] = max(sub["monthly_quota"] - used, 0)
            profile["subscription"] = sub
        return profile


def list_reservations(user_id: str, include_cancelled: bool = False) -> dict:
    """List a customer's reservations with experience details."""
    with session_scope(cultpass_engine()) as session:
        if session.get(cultpass.User, user_id) is None:
            return _fail("user_not_found", f"No CultPass customer with id '{user_id}'.")
        query = session.query(cultpass.Reservation).filter(cultpass.Reservation.user_id == user_id)
        if not include_cancelled:
            query = query.filter(cultpass.Reservation.status == "reserved")
        items = []
        for res in query.order_by(cultpass.Reservation.created_at).all():
            exp = res.experience
            items.append(
                {
                    "reservation_id": res.reservation_id,
                    "status": res.status,
                    "experience_id": exp.experience_id,
                    "title": exp.title,
                    "location": exp.location,
                    "when": exp.when.isoformat(sep=" ", timespec="minutes"),
                    "is_premium": bool(exp.is_premium),
                }
            )
        return {"ok": True, "user_id": user_id, "reservations": items}


def search_experiences(query: str = "", location: str = "", only_available: bool = True, limit: int = 10) -> dict:
    """Search upcoming experiences by keyword (title/description) and/or location (city or state)."""
    with session_scope(cultpass_engine()) as session:
        q = session.query(cultpass.Experience).filter(cultpass.Experience.when >= datetime.now())
        if only_available:
            q = q.filter(cultpass.Experience.slots_available > 0)
        rows = q.order_by(cultpass.Experience.when).all()
        catalogue = list(rows)
        if location:
            by_location = [e for e in rows if location.lower() in e.location.lower()]
            if by_location:
                rows = by_location
            else:  # e.g. a neighbourhood like "Lapa": match it as a keyword instead
                query = f"{query} {location}".strip()
        if query:
            terms = [t for t in query.lower().split() if len(t) > 2]
            scored = []
            for exp in rows:
                text = f"{exp.title} {exp.description} {exp.location}".lower()
                hits = sum(t in text for t in terms)
                if hits:
                    scored.append((hits, exp))
            scored.sort(key=lambda pair: (-pair[0], pair[1].when))
            rows = [exp for _, exp in scored]
        experiences = [to_dict(exp) for exp in rows[:limit]]
        for exp in experiences:
            exp.pop("created_at", None)
            exp.pop("updated_at", None)
        result = {"ok": True, "count": len(experiences), "experiences": experiences}
        if not experiences:
            # keywords like "music" may not appear literally: offer the catalogue so the agent can match
            # it against what the customer wants (e.g. a samba night for "music")
            suggestions = [
                {k: v for k, v in to_dict(e).items() if k not in ("created_at", "updated_at")} for e in catalogue[:limit]
            ]
            result["suggestions"] = suggestions
            result["message"] = (
                "No experience matched these keywords literally. This does NOT mean a specific event is sold out. "
                "'suggestions' lists all upcoming experiences with free slots: pick the ones that fit the request."
            )
        return result


def reserve_experience(user_id: str, experience_id: str) -> dict:
    """Reserve an experience for a customer, enforcing CultPass business rules."""
    with session_scope(cultpass_engine()) as session:
        user = session.get(cultpass.User, user_id)
        if user is None:
            return _fail("user_not_found", f"No CultPass customer with id '{user_id}'.")
        if user.is_blocked:
            return _fail("account_blocked", "The account is blocked; reservations are disabled until support reviews it.")
        sub = user.subscription
        if sub is None or sub.status != "active":
            return _fail("subscription_inactive", "An active subscription is required to reserve experiences.")
        exp = session.get(cultpass.Experience, experience_id)
        if exp is None:
            return _fail("experience_not_found", f"No experience with id '{experience_id}'.")
        if exp.when < datetime.now():
            return _fail("experience_in_past", "This experience has already happened.")
        if exp.slots_available <= 0:
            return _fail("sold_out", f"'{exp.title}' has no slots left.")
        if exp.is_premium and sub.tier != "premium":
            return _fail(
                "premium_requires_upgrade",
                f"'{exp.title}' is a premium experience. Basic members can book it in the app by paying the extra fee shown there, or upgrade to Premium.",
            )
        duplicate = (
            session.query(cultpass.Reservation)
            .filter_by(user_id=user_id, experience_id=experience_id, status="reserved")
            .first()
        )
        if duplicate:
            return _fail("already_reserved", f"Already reserved (reservation {duplicate.reservation_id}).")
        used = _reservations_this_month(session, user_id)
        if used >= sub.monthly_quota:
            return _fail("quota_exceeded", f"Monthly quota of {sub.monthly_quota} experiences already used ({used}).")

        reservation = cultpass.Reservation(
            reservation_id=str(uuid.uuid4())[:6],
            user_id=user_id,
            experience_id=experience_id,
            status="reserved",
            created_at=datetime.now(),
        )
        exp.slots_available -= 1
        session.add(reservation)
        return {
            "ok": True,
            "reservation_id": reservation.reservation_id,
            "experience_id": exp.experience_id,
            "title": exp.title,
            "location": exp.location,
            "when": exp.when.isoformat(sep=" ", timespec="minutes"),
            "remaining_quota": sub.monthly_quota - used - 1,
            "message": f"Reserved '{exp.title}'. Arrive 15 minutes early with your QR code.",
        }


def cancel_reservation(user_id: str, reservation_id: str) -> dict:
    """Cancel one of the customer's reservations (>= 24h before the event)."""
    with session_scope(cultpass_engine()) as session:
        res = session.get(cultpass.Reservation, reservation_id)
        if res is None or res.user_id != user_id:
            return _fail("reservation_not_found", f"Reservation '{reservation_id}' was not found for this customer.")
        if res.status != "reserved":
            return _fail("not_active", f"Reservation '{reservation_id}' is already {res.status}.")
        exp = res.experience
        if exp.when - datetime.now() < LATE_CANCELLATION_WINDOW:
            return _fail(
                "late_cancellation",
                "Cancellations less than 24 hours before the event need approval from a human agent.",
            )
        res.status = "cancelled"
        exp.slots_available += 1
        return {
            "ok": True,
            "reservation_id": reservation_id,
            "title": exp.title,
            "message": f"Reservation for '{exp.title}' cancelled; the credit was returned to your monthly quota.",
        }


def find_customer(identifier: str) -> dict:
    """Look a customer up by id, e-mail or (partial) name. Used by support staff tools."""
    with session_scope(cultpass_engine()) as session:
        rows = (
            session.query(cultpass.User)
            .filter(
                or_(
                    cultpass.User.user_id == identifier,
                    cultpass.User.email == identifier.lower(),
                    cultpass.User.full_name.ilike(f"%{identifier}%"),
                )
            )
            .all()
        )
        return {"ok": True, "matches": [{"user_id": u.user_id, "full_name": u.full_name} for u in rows]}


# --------------------------------------------------------------------------- #
# Subscription management (two-step: preview, then confirm=True)
# --------------------------------------------------------------------------- #
TIERS = ("basic", "premium")


def _end_of_billing_cycle(now: datetime) -> datetime:
    first_next_month = (now.replace(day=1) + timedelta(days=32)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return first_next_month - timedelta(seconds=1)


def _confirmation(action: str, preview: str, **extra) -> dict:
    return {
        "ok": True,
        "requires_confirmation": True,
        "action": action,
        "preview": preview,
        "message": f"{preview} Ask the customer to confirm, then call again with confirm=true.",
        **extra,
    }


def _load_subscription(session, user_id: str):
    user = session.get(cultpass.User, user_id)
    if user is None:
        return None, _fail("user_not_found", f"No CultPass customer with id '{user_id}'.")
    if user.is_blocked:
        return None, _fail("account_blocked", "The account is blocked; subscription changes require a human agent.")
    if user.subscription is None:
        return None, _fail("no_subscription", "This customer has no subscription.")
    return user.subscription, None


def pause_subscription(user_id: str, confirm: bool = False) -> dict:
    """Pause an active subscription (data is kept; no new reservations until resumed)."""
    with session_scope(cultpass_engine()) as session:
        sub, err = _load_subscription(session, user_id)
        if err:
            return err
        if sub.status != "active":
            return _fail("invalid_state", f"Only active subscriptions can be paused (current: {sub.status}).")
        if not confirm:
            return _confirmation("pause_subscription", "The subscription will be paused immediately; existing reservations stay valid.")
        sub.status = "paused"
        return {"ok": True, "status": "paused", "message": "Subscription paused. Resume it any time."}


def resume_subscription(user_id: str, confirm: bool = False) -> dict:
    """Reactivate a paused or cancelled subscription."""
    with session_scope(cultpass_engine()) as session:
        sub, err = _load_subscription(session, user_id)
        if err:
            return err
        if sub.status == "active":
            return _fail("invalid_state", "The subscription is already active.")
        if not confirm:
            return _confirmation("resume_subscription", f"The {sub.tier} subscription will be reactivated and billed from today.")
        sub.status, sub.ended_at = "active", None
        return {"ok": True, "status": "active", "message": f"Your {sub.tier} subscription is active again."}


def cancel_subscription(user_id: str, confirm: bool = False) -> dict:
    """Cancel the subscription at the end of the current billing cycle (no refunds)."""
    with session_scope(cultpass_engine()) as session:
        sub, err = _load_subscription(session, user_id)
        if err:
            return err
        if sub.status == "cancelled":
            return _fail("invalid_state", "The subscription is already cancelled.")
        end = _end_of_billing_cycle(datetime.now())
        if not confirm:
            return _confirmation(
                "cancel_subscription",
                f"The subscription will be cancelled; access continues until {end:%Y-%m-%d}. No refund is issued.",
            )
        sub.status, sub.ended_at = "cancelled", end
        return {"ok": True, "status": "cancelled", "access_until": f"{end:%Y-%m-%d}",
                "message": f"Subscription cancelled. You keep access until {end:%Y-%m-%d}."}


def change_subscription_tier(user_id: str, new_tier: str, confirm: bool = False) -> dict:
    """Upgrade or downgrade between 'basic' and 'premium' (effective immediately)."""
    new_tier = (new_tier or "").strip().lower()
    if new_tier not in TIERS:
        return _fail("invalid_tier", f"Tier must be one of {list(TIERS)}.")
    with session_scope(cultpass_engine()) as session:
        sub, err = _load_subscription(session, user_id)
        if err:
            return err
        if sub.status != "active":
            return _fail("invalid_state", f"Only active subscriptions can change tier (current: {sub.status}).")
        if sub.tier == new_tier:
            return _fail("invalid_state", f"The subscription is already {new_tier}.")
        if new_tier == "basic":
            premium_bookings = (
                session.query(cultpass.Reservation)
                .join(cultpass.Experience)
                .filter(cultpass.Reservation.user_id == user_id, cultpass.Reservation.status == "reserved",
                        cultpass.Experience.is_premium.is_(True), cultpass.Experience.when >= datetime.now())
                .count()
            )
            if premium_bookings:
                return _fail("has_premium_reservations",
                             f"Cancel the {premium_bookings} upcoming premium reservation(s) before downgrading to basic.")
        if not confirm:
            return _confirmation("change_subscription_tier", f"The plan will change from {sub.tier} to {new_tier} immediately.",
                                 new_tier=new_tier)
        old, sub.tier = sub.tier, new_tier
        return {"ok": True, "old_tier": old, "tier": new_tier, "message": f"Plan changed from {old} to {new_tier}."}


# --------------------------------------------------------------------------- #
# Refunds (filed by the AI, decided only by a human support lead)
# --------------------------------------------------------------------------- #
MAX_REFUND_AMOUNT = 1000.0


def request_refund(user_id: str, reason: str, amount: float | None = None, reservation_id: str | None = None) -> dict:
    """File a refund request. It is always created as 'pending_approval'; it is never auto-approved."""
    reason = (reason or "").strip()
    if len(reason) < 5:
        return _fail("invalid_reason", "Please describe why the refund is requested.")
    if amount is not None and not (0 < amount <= MAX_REFUND_AMOUNT):
        return _fail("invalid_amount", f"Amount must be between 0 and {MAX_REFUND_AMOUNT:.0f}.")
    with session_scope(cultpass_engine()) as session:
        if session.get(cultpass.User, user_id) is None:
            return _fail("user_not_found", f"No CultPass customer with id '{user_id}'.")
        if reservation_id:
            res = session.get(cultpass.Reservation, reservation_id)
            if res is None or res.user_id != user_id:
                return _fail("reservation_not_found", f"Reservation '{reservation_id}' was not found for this customer.")
        pending = (
            session.query(cultpass.RefundRequest)
            .filter_by(user_id=user_id, status="pending_approval", reservation_id=reservation_id)
            .first()
        )
        if pending:
            return {"ok": True, "refund_id": pending.refund_id, "status": "pending_approval", "duplicate": True,
                    "message": f"A refund request ({pending.refund_id}) is already pending review."}
        refund = cultpass.RefundRequest(
            refund_id=f"rf-{uuid.uuid4().hex[:6]}", user_id=user_id, reservation_id=reservation_id,
            amount=amount, reason=reason, status="pending_approval", created_at=datetime.now(),
        )
        session.add(refund)
        return {"ok": True, "refund_id": refund.refund_id, "status": "pending_approval",
                "message": f"Refund request {refund.refund_id} filed; the billing team decides within 3 business days."}


def list_refund_requests(user_id: str) -> dict:
    """List the customer's refund requests and their status."""
    with session_scope(cultpass_engine()) as session:
        rows = session.query(cultpass.RefundRequest).filter_by(user_id=user_id).order_by(cultpass.RefundRequest.created_at).all()
        return {"ok": True, "refunds": [to_dict(r) for r in rows]}


def decide_refund(refund_id: str, approve: bool, decided_by: str, note: str = "") -> dict:
    """STAFF ONLY: approve or deny a pending refund request."""
    if not decided_by.strip():
        return _fail("missing_agent", "decided_by (the human agent) is required.")
    with session_scope(cultpass_engine()) as session:
        refund = session.get(cultpass.RefundRequest, refund_id)
        if refund is None:
            return _fail("refund_not_found", f"No refund request '{refund_id}'.")
        if refund.status != "pending_approval":
            return _fail("already_decided", f"Refund {refund_id} is already {refund.status}.")
        refund.status = "approved" if approve else "denied"
        refund.decided_by, refund.decision_note, refund.decided_at = decided_by, note, datetime.now()
        return {"ok": True, "refund_id": refund_id, "status": refund.status}


# --------------------------------------------------------------------------- #
# Account modifications
# --------------------------------------------------------------------------- #
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)


def update_profile(user_id: str, full_name: str) -> dict:
    """Update the customer's display name."""
    full_name = " ".join((full_name or "").split())
    if not (2 <= len(full_name) <= 100):
        return _fail("invalid_name", "Name must be between 2 and 100 characters.")
    with session_scope(cultpass_engine()) as session:
        user = session.get(cultpass.User, user_id)
        if user is None:
            return _fail("user_not_found", f"No CultPass customer with id '{user_id}'.")
        if user.is_blocked:
            return _fail("account_blocked", "The account is blocked; profile changes require a human agent.")
        old, user.full_name = user.full_name, full_name
        return {"ok": True, "old_name": old, "full_name": full_name, "message": f"Name updated to {full_name}."}


def request_email_change(user_id: str, new_email: str) -> dict:
    """Start an e-mail change: the address is validated and a confirmation link is sent to it.
    The login e-mail only changes after the customer clicks the link (never directly from chat)."""
    new_email = (new_email or "").strip().lower()
    if not _EMAIL.match(new_email):
        return _fail("invalid_email", f"'{new_email}' is not a valid e-mail address.")
    with session_scope(cultpass_engine()) as session:
        user = session.get(cultpass.User, user_id)
        if user is None:
            return _fail("user_not_found", f"No CultPass customer with id '{user_id}'.")
        if user.is_blocked:
            return _fail("account_blocked", "The account is blocked; e-mail changes require a human agent.")
        if user.email == new_email:
            return _fail("same_email", "That is already the account e-mail.")
        if session.query(cultpass.User).filter(cultpass.User.email == new_email).first():
            return _fail("email_in_use", "That e-mail is already used by another account.")
        return {"ok": True, "status": "verification_sent", "new_email": new_email,
                "message": f"A confirmation link was sent to {new_email}; the change applies once it is clicked."}


def set_account_block(user_id: str, blocked: bool, reason: str, decided_by: str) -> dict:
    """STAFF ONLY: block or unblock a customer account after a review."""
    if not decided_by.strip() or len((reason or "").strip()) < 5:
        return _fail("missing_fields", "decided_by and a reason are required.")
    with session_scope(cultpass_engine()) as session:
        user = session.get(cultpass.User, user_id)
        if user is None:
            return _fail("user_not_found", f"No CultPass customer with id '{user_id}'.")
        user.is_blocked = blocked
        return {"ok": True, "user_id": user_id, "is_blocked": blocked}
