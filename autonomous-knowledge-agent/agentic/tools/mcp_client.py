"""Client side of the MCP tool layer.

Agents never touch the databases directly: every read/write goes through
``call_tool(server, tool, args)``, which speaks the MCP protocol to one of the
servers in ``agentic/tools/mcp_servers``.

Transports (``UDAHUB_MCP_TRANSPORT``):
  - ``memory`` (default): FastMCP in-memory transport. Full MCP protocol
    (schema validation, JSON serialisation) without spawning processes.
  - ``stdio``: each call launches the server as a subprocess, which is how an
    external MCP host (Claude Desktop, IDEs, ...) would connect to it.

LangGraph nodes are synchronous (``chat_interface`` calls ``invoke``), and a
notebook already runs an event loop, so ``call_tool`` runs the async client on
a short-lived worker thread with its own event loop.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
import threading
from typing import Any

# quiet FastMCP's console warnings (errors are returned to agents as {"ok": False})
os.environ.setdefault("FASTMCP_LOG_LEVEL", "ERROR")

from fastmcp import Client  # noqa: E402
from fastmcp.client.transports import StdioTransport  # noqa: E402

from agentic import config  # noqa: E402


SERVERS: dict[str, str] = {
    "cultpass": "agentic.tools.mcp_servers.cultpass_server",
    "udahub": "agentic.tools.mcp_servers.udahub_server",
}


def _subprocess_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(config.SOLUTION_ROOT), env.get("PYTHONPATH")]))
    # make sure the subprocess points at the same databases as this process
    env["UDAHUB_DB_PATH"] = str(config.udahub_db_path())
    env["CULTPASS_DB_PATH"] = str(config.cultpass_db_path())
    return env


def _client(server: str) -> Client:
    if server not in SERVERS:
        raise ValueError(f"Unknown MCP server '{server}'. Known: {sorted(SERVERS)}")
    if config.mcp_transport() == "stdio":
        return Client(
            StdioTransport(
                command=sys.executable,
                args=["-m", SERVERS[server]],
                env=_subprocess_env(),
                cwd=str(config.SOLUTION_ROOT),
            )
        )
    return Client(importlib.import_module(SERVERS[server]).mcp)


async def acall_tool(server: str, tool: str, args: dict[str, Any]) -> dict:
    async with _client(server) as client:
        result = await client.call_tool(tool, args, raise_on_error=False)
    if result.is_error:
        text = " ".join(getattr(c, "text", "") for c in result.content)
        return {"ok": False, "error": "tool_error", "message": text.strip()}
    if isinstance(result.structured_content, dict):
        return result.structured_content
    return {"ok": True, "result": result.data}


async def alist_tools(server: str) -> list[str]:
    async with _client(server) as client:
        return [t.name for t in await client.list_tools()]


def _run_sync(coro_factory):
    box: dict[str, Any] = {}

    def runner():
        try:
            box["value"] = asyncio.run(coro_factory())
        except BaseException as exc:  # re-raised in caller thread
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def call_tool(server: str, tool: str, args: dict[str, Any] | None = None) -> dict:
    """Synchronously call an MCP tool and return its JSON result."""
    return _run_sync(lambda: acall_tool(server, tool, args or {}))


def list_tools(server: str) -> list[str]:
    return _run_sync(lambda: alist_tools(server))


def mcp_servers_config() -> dict:
    """Connection config for ``langchain_mcp_adapters.MultiServerMCPClient`` (stdio)."""
    return {
        name: {
            "command": sys.executable,
            "args": ["-m", module],
            "transport": "stdio",
            "cwd": str(config.SOLUTION_ROOT),
            "env": _subprocess_env(),
        }
        for name, module in SERVERS.items()
    }
