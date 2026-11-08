"""The tools are served over MCP; agents call them through the MCP client."""
import asyncio

import pytest

from agentic.tools import mcp_client


def test_servers_expose_expected_tools(dbs):
    assert {"get_customer_profile", "reserve_experience", "cancel_reservation", "search_experiences"} <= set(
        mcp_client.list_tools("cultpass")
    )
    assert {"search_knowledge", "create_ticket", "save_memory", "search_memories"} <= set(
        mcp_client.list_tools("udahub")
    )


def test_call_tool_over_mcp_in_memory(dbs):
    profile = mcp_client.call_tool("cultpass", "get_customer_profile", {"user_id": "f556c0"})
    assert profile["ok"] and profile["full_name"] == "Bob Stone"
    kb = mcp_client.call_tool("udahub", "search_knowledge", {"query": "reset password", "k": 1})
    assert kb["results"][0]["title"] == "How to Handle Login Issues?"


def test_invalid_arguments_return_error_instead_of_raising(dbs):
    result = mcp_client.call_tool("cultpass", "reserve_experience", {"user_id": "f556c0"})
    assert result["ok"] is False and result["error"] == "tool_error"


def test_unknown_server():
    with pytest.raises(ValueError):
        mcp_client.call_tool("nope", "x", {})


def test_call_tool_works_inside_running_event_loop(dbs):
    """Jupyter runs an event loop; the sync bridge must still work there."""

    async def inside_loop():
        return mcp_client.call_tool("cultpass", "get_customer_profile", {"user_id": "88382b"})

    assert asyncio.run(inside_loop())["full_name"] == "Cathy Bloom"


@pytest.mark.slow
def test_stdio_transport_subprocess(dbs, monkeypatch):
    monkeypatch.setenv("UDAHUB_MCP_TRANSPORT", "stdio")
    result = mcp_client.call_tool("cultpass", "get_customer_profile", {"user_id": "e6376d"})
    assert result["full_name"] == "Frank Ocean"


@pytest.mark.slow
def test_langchain_mcp_adapters_can_load_servers(dbs):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def load():
        client = MultiServerMCPClient(mcp_client.mcp_servers_config())
        tools = await client.get_tools()
        by_name = {t.name: t for t in tools}
        out = await by_name["search_knowledge"].ainvoke({"query": "wheelchair access", "k": 1})
        return by_name, out

    by_name, out = asyncio.run(load())
    assert "reserve_experience" in by_name and "search_memories" in by_name
    assert "Accessibility" in str(out)
