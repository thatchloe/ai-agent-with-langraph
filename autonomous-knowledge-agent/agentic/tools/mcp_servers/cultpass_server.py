"""MCP server exposing the CultPass customer system (external DB).

Tools tagged ``customer`` are given to the AI account agent (bound to the ticket's
customer). Tools tagged ``staff`` are for human support agents only (approving
refunds, blocking accounts, looking customers up) and are never given to the AI.

Run standalone (stdio transport) from the ``solution/`` folder:

    python -m agentic.tools.mcp_servers.cultpass_server
"""
from fastmcp import FastMCP

from agentic.tools import cultpass_tools as ct

mcp = FastMCP(
    name="cultpass",
    instructions=(
        "Customer operations for the CultPass account: profile, subscription management "
        "(pause/resume/cancel/change tier, two-step confirm), reservations, experience catalogue, "
        "refund requests and account changes. Tools tagged 'staff' require a human agent."
    ),
)

CUSTOMER_TOOLS = (
    ct.get_customer_profile,
    ct.list_reservations,
    ct.search_experiences,
    ct.reserve_experience,
    ct.cancel_reservation,
    ct.pause_subscription,
    ct.resume_subscription,
    ct.cancel_subscription,
    ct.change_subscription_tier,
    ct.request_refund,
    ct.list_refund_requests,
    ct.update_profile,
    ct.request_email_change,
)
STAFF_TOOLS = (ct.find_customer, ct.decide_refund, ct.set_account_block)

for _fn in CUSTOMER_TOOLS:
    mcp.tool(_fn, tags={"customer"})
for _fn in STAFF_TOOLS:
    mcp.tool(_fn, tags={"staff"})


if __name__ == "__main__":
    mcp.run(show_banner=False)
