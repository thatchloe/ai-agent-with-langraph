"""MCP server exposing the core UDA-Hub platform (knowledge base, tickets, memory).

Run standalone (stdio transport) from the ``solution/`` folder:

    python -m agentic.tools.mcp_servers.udahub_server
"""
from fastmcp import FastMCP

from agentic.tools import udahub_tools

mcp = FastMCP(
    name="udahub",
    instructions=(
        "UDA-Hub core services: knowledge-base search (RAG), ticket lifecycle and "
        "long-term customer memory."
    ),
)

for _fn in (
    udahub_tools.search_knowledge,
    udahub_tools.get_ticket,
    udahub_tools.create_ticket,
    udahub_tools.add_ticket_message,
    udahub_tools.update_ticket_metadata,
    udahub_tools.get_customer_ticket_history,
    udahub_tools.save_memory,
    udahub_tools.search_memories,
):
    mcp.tool(_fn, tags={"platform"})

for _fn in (udahub_tools.list_tickets, udahub_tools.assign_ticket):
    mcp.tool(_fn, tags={"staff"})


if __name__ == "__main__":
    mcp.run(show_banner=False)
