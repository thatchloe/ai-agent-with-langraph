# Agents

| File | Node(s) | Role |
|---|---|---|
| `ticket_manager.py` | `ticket_intake`, `finalize` | Ticket lifecycle, channel and memory: load context at the start of a turn; at the end, format the reply for its channel and persist the outcome, priority and queue |
| `classifier.py` | `classifier` | LLM structured classification (category, urgency, sentiment, preferences), with a keyword fallback and guardrails |
| `supervisor.py` | `supervisor` | Hard policies, then a routing strategy (`rules` or `llm`, A/B tested), plus P1–P4 priority |
| `knowledge.py` | `knowledge_agent` | Hybrid semantic RAG answers; escalates when articles aren't relevant |
| `account.py` | `account_agent`, `account_tools` | Hand-built tool-calling loop over the customer's MCP tools; two-step confirmation; automatic escalation of refunds |
| `escalation.py` | `escalation_agent` | Human hand-off note, queue choice and customer message |
| `common.py` | – | Shared prompt-context helpers (customer, memory, channel style) |

See `../design/README.md` for the full design.
