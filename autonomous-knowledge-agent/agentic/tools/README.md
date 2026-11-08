# Tools

| File | Purpose |
|---|---|
| `cultpass_tools.py` | CultPass operations with business rules: profile, reservations, experiences, reserve and cancel; subscription pause, resume, cancel and tier change (with confirmation); refund requests and staff decisions; profile and e-mail changes; staff block and unblock |
| `udahub_tools.py` | Knowledge search (hybrid RAG with a persistent vector store), ticket lifecycle (status, priority, queue), staff work queue, long-term memory |
| `retrieval.py` | `HybridRetriever`: 0.25 × TF-IDF + 0.75 × embeddings, with calibrated thresholds |
| `mcp_servers/cultpass_server.py`, `mcp_servers/udahub_server.py` | FastMCP servers (26 tools, tagged `customer`, `platform` or `staff`) |
| `mcp_client.py` | Synchronous MCP client used by the agents (in-memory or stdio transport) |
| `agent_tools.py` | The 16 tools given to the account agent, bound to the ticket's customer (no staff tools) |
| `db.py` | Engines and sessions, using absolute paths from `agentic/config.py` |

Run a server standalone (from `solution/`): `python -m agentic.tools.mcp_servers.cultpass_server`
