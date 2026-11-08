# UDA-Hub: Multi-Agent Customer Support (CultPass)

UDA-Hub is a LangGraph multi-agent system that handles CultPass support tickets end to end, from chat,
e-mail and social media. A classifier labels each message, including its **sentiment** and urgency. A
supervisor gives it a **priority** (P1–P4) and routes it to one of three specialist agents:

- a **knowledge agent**, which answers from the knowledge base using hybrid **semantic RAG** (embeddings plus TF-IDF, with a persistent vector store);
- an **account agent**, which acts on the customer's account through **26 FastMCP tools**: reservations,
  subscription management with a confirmation step, refund requests, profile and e-mail changes;
- an **escalation agent**, which hands the ticket to the right human queue with a summary.

Two routing strategies, deterministic rules and an LLM router, can be **A/B tested** on live traffic and on
a labelled ticket set. The system has short-term memory per ticket, long-term memory per customer, and a
structured log of every decision.

**Design:** [agentic/design/README.md](agentic/design/README.md) covers the architecture diagram, agent
roles, inputs and outputs, routing policies, RAG, memory, tools, priority, A/B testing, channels and
logging. **A/B results:** [agentic/design/ab_test_report.md](agentic/design/ab_test_report.md).

## Project structure

```
solution/
├── agentic/
│   ├── design/                   # design document + A/B test report
│   ├── agents/                   # intake/finalize, classifier, supervisor, knowledge, account, escalation
│   ├── tools/                    # DB tools, hybrid retriever, 2 MCP servers + client, customer-scoped tools
│   ├── channels.py               # chat / e-mail / social adapters and reply formatting
│   ├── experiments.py            # A/B assignment + online report
│   ├── evaluation.py             # offline A/B evaluation on the labelled set
│   ├── config.py                 # absolute paths, LLM + embeddings factory
│   ├── state.py                  # graph state
│   ├── logging_utils.py          # structured JSONL decision log
│   └── workflow.py               # StateGraph built from scratch -> `orchestrator`
├── data/
│   ├── models/                   # SQLAlchemy models (+ CustomerMemory, KnowledgeEmbedding, RefundRequest)
│   ├── external/                 # CultPass JSONL files (16 articles); cultpass.db is generated here
│   ├── evaluation/               # labelled tickets for the A/B evaluation
│   └── core/                     # udahub.db is generated here
├── tests/                        # pytest suite (offline) + live LLM tests
├── 01_external_db_setup.ipynb    # builds data/external/cultpass.db
├── 02_core_db_setup.ipynb        # builds data/core/udahub.db + vector store
├── 03_agentic_app.ipynb          # building blocks + end-to-end scenarios + chat (executed)
├── 03_agentic_app.py             # the same, as a CLI
├── utils.py                      # DB helpers + improved chat_interface()
├── requirements.txt
└── .env.example
```

## Getting started

### Dependencies

The solution was built and tested with **Python 3.11.5**. Packages are pinned in
[requirements.txt](requirements.txt):

| Package | Version | Note |
|---|---|---|
| langgraph | 1.2.20 | graph orchestration and checkpointer |
| langgraph-checkpoint | 4.3.1 | |
| langchain / langchain-core | 1.4.9 / 1.7.6 | |
| langchain-openai | 1.7.4 | `gpt-4o-mini` and `text-embedding-3-small` |
| langchain-mcp-adapters | 0.3.1 | load the MCP servers as LangChain tools |
| fastmcp | 3.4.11 | MCP servers and client |
| mcp | 1.30.0 | **pinned below 2**: `fastmcp>=2.10.6` would otherwise install fastmcp 4 / mcp 2, which breaks langchain-mcp-adapters |
| langgraph-supervisor | 0.0.31 | from the starter requirements; not used (the graph is hand-built) |
| sqlalchemy | 2.1.8 | |
| python-dotenv | 1.2.4 | |
| httpx | 0.28.1 | |
| ipykernel | 7.4.0 | |
| pydantic | 2.14.3 | structured outputs |
| pytest | 9.2.0 | tests |
| nbclient | 0.11.0 | optional, for running the notebooks headless |

### Installation

```bash
cd solution
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then set VOCAREUM_API_KEY (or OPENAI_API_KEY)
```

Build the databases by running the two setup notebooks in order, with `solution/` as the working directory:

1. `01_external_db_setup.ipynb` creates `data/external/cultpass.db`: customers, subscriptions, experiences, reservations, and an empty `refund_requests` table.
2. `02_core_db_setup.ipynb` creates `data/core/udahub.db`: the account, 16 knowledge articles, a ticket, a past ticket with memories, and the vector store (built when an API key is set).

## Running the app

- **Notebook:** `03_agentic_app.ipynb` is executed with outputs.
  - **Part A** works without an LLM. It shows data retrieval, RAG with relevance scores, the MCP tools
    with their validation errors, memory recall, and routing with priority.
  - **Part B** runs complete tickets through the graph:
    - B1: semantic RAG
    - B2: account actions over several turns
    - B3: a subscription pause with confirmation
    - B4: refund filing, billing escalation and a staff approval
    - B5: escalations
    - B6: e-mail and social channels
    - B7: long-term memory
    - B8: the staff work queue
    - B9: A/B routing

    It then runs `chat_interface(...)` (a scripted replay; set `INTERACTIVE = True` to type yourself)
    and shows how to inspect the state and logs.
- **CLI**, from `solution/`:

  ```bash
  python 03_agentic_app.py --demo              # scripted scenarios; prints the category, route and ticket status
  python 03_agentic_app.py --user f556c0       # interactive chat as Bob Stone (new ticket)
  python 03_agentic_app.py --ticket <id>       # continue an existing ticket (e.g. Alice's from notebook 02)
  python 03_agentic_app.py --list              # list customers and tickets
  python -m agentic.evaluation                 # offline A/B evaluation -> agentic/design/ab_test_report.md (~5 min)
  ```

- **From Python:**

  ```python
  from langchain_core.messages import HumanMessage
  from agentic.workflow import orchestrator
  from agentic.channels import from_email

  # chat (or any custom source)
  state = orchestrator.invoke(
      {"messages": [HumanMessage("Please book Samba Night at Lapa")], "external_user_id": "f556c0",
       "ticket_metadata": {"channel": "chat"}},
      config={"configurable": {"thread_id": "ticket-42", "routing_strategy": "rules"}},  # rules | llm | ab
  )
  # e-mail: the customer is identified from the sender address
  state = orchestrator.invoke(from_email("bob.stone@granite.com", "QR code", "My QR code is missing"),
                              config={"configurable": {"thread_id": "ticket-43"}})
  ```

Demo customers (`--list`): `a4ab87` Alice (blocked), `f556c0` Bob (premium, has history), `88382b` Cathy
(basic), `888fb2` David (blocked), `f1f10d` Eva (cancelled), `e6376d` Frank (basic, small quota).

Every decision is appended to `logs/agent_events.jsonl`, for example:

```bash
grep '"event": "routing"' logs/agent_events.jsonl
```

The MCP servers can also run standalone for any MCP host. Staff tools (`decide_refund`,
`set_account_block`, `list_tickets`, `assign_ticket`) are meant for human agents:

```bash
python -m agentic.tools.mcp_servers.cultpass_server
python -m agentic.tools.mcp_servers.udahub_server
```

### Configuration (`.env`)

| Variable | Default | Effect |
|---|---|---|
| `VOCAREUM_API_KEY` / `OPENAI_API_KEY` | – | LLM and embeddings credentials |
| `UDAHUB_MODEL` | `gpt-4o-mini` | chat model |
| `UDAHUB_EMBEDDINGS` | `auto` | `auto` (on with a key), `openai` or `off` (TF-IDF only) |
| `UDAHUB_ROUTING_STRATEGY` | `rules` | `rules`, `llm` or `ab` (50/50, sticky per ticket) |
| `UDAHUB_MCP_TRANSPORT` | `memory` | `memory` or `stdio` |
| `UDAHUB_DB_PATH`, `CULTPASS_DB_PATH`, `UDAHUB_LOG_PATH` | under `data/`, `logs/` | override locations |

## Testing

```bash
cd solution
pytest -m "not live"   # 128 offline tests, ~9 s, no API key or network needed
pytest -m live -v      # 11 tests against the real LLM and embeddings (~2 min; needs a key in .env)
pytest                 # everything (live tests are skipped automatically without a key)
```

The offline tests need **no API key and no network**:

- Every test builds fresh, deterministic copies of `cultpass.db` and `udahub.db` in a temp folder.
- Only the LLM is replaced, by `tests/fake_llm.py::ScriptedLLM`, a rule-based chat model that supports
  tool calls and structured output.
- Everything else is real: the compiled graph, the MCP servers, the SQLite databases and the logging.

### Test breakdown

| File | What it checks |
|---|---|
| `test_data_setup.py` | At least 14 unique articles with diverse topics. The databases built by the notebooks contain the articles, the memory table and the reservations. |
| `test_retrieval.py` | 16 realistic queries return the correct article first (lexical mode). Off-topic queries fall below the threshold. Stemming works. Hybrid retrieval finds a paraphrase that TF-IDF misses. |
| `test_cultpass_tools.py` | Profile and quota, reservations, experience search (including a neighbourhood used as the location, and an empty result explained as "not sold out"). Booking and cancellation business rules. |
| `test_support_operations.py` | Pausing, resuming and cancelling subscriptions with confirmation; tier changes, including the premium-reservation guard. Refund validation, dedup and the staff decision. Profile and e-mail changes. Staff block and unblock. Ticket priority and queue, and the staff work queue. MCP exposes every operation, but the AI gets no staff tools. |
| `test_udahub_tools.py` | Ticket creation, message log, status and tag merging, long-term memory save, search and dedup, ticket history. |
| `test_mcp_servers.py` | Both servers' tools over MCP (in-memory). Errors come back instead of exceptions. Works inside a running event loop. The **stdio** subprocess transport works. `langchain-mcp-adapters` loads both servers. |
| `test_agents.py` | Classifier (heuristics, LLM structured output, guardrails, fallback). The supervisor routing table (19 cases, including the policies). Account tools are scoped to the customer. The knowledge agent's escalation and grounding. |
| `test_channels.py` | E-mail cleaning and identifying the customer from the sender; unknown senders; social-media and chat adapters; per-channel reply formatting (greeting and sign-off, redaction, 280-character limit). |
| `test_experiments.py` | Sticky, roughly 50/50 variant assignment; strategy selection; the LLM router respects the hard policies and falls back to rules on error; priority scoring; `ab_report` from the logs; the offline evaluation harness (sandboxed). |
| `test_workflow.py` | 23 end-to-end graph scenarios: RAG resolution, reservation action, business-rule refusal, blocked-account policy, late cancellation, refund filed then escalated to billing, off-topic escalation, short-term and long-term memory, escalated-ticket follow-up, LLM outage, guest tickets, metadata, logging, two-step subscription change (confirmed and declined), profile update, e-mail and social channels end to end, priority and queue persisted. |
| `test_live_llm.py` | Against the real model: hybrid retrieval calibration and the persistent vector store; RAG answer; reservation; blocked account; refund; off-topic; multiple turns; subscription pause with confirmation; e-mail and social channels; LLM routing strategy. |

## Where each rubric criterion is met

| Rubric criterion | Evidence |
|---|---|
| Databases set up, required tables, data retrieval | Notebooks 01 and 02 (executed); notebook 03 §A2; `tests/test_data_setup.py` |
| At least 10 new articles in different categories | 12 new articles (16 in total) in `data/external/cultpass_articles.jsonl`; `test_articles_are_unique_and_cover_diverse_topics` |
| Architecture document with a diagram, roles, flow, inputs and outputs, and a standard pattern | `agentic/design/README.md` §1–4 (supervisor pattern, Mermaid diagram, agent table, input/output table, routing tables) |
| At least 4 specialised agents connected in LangGraph, with state and message passing | `agentic/workflow.py` (classifier, supervisor, knowledge, account and escalation agents, plus intake, tools and finalize nodes); `agentic/state.py` |
| Classification and routing on content and metadata | `agents/classifier.py` and `agents/supervisor.py`. Routing uses category, sentiment, urgency, confidence, ticket status, turn count, blocked status, pending confirmations and platform urgency. See notebook 03 §A6, `test_supervisor_routing` and the A/B report. |
| RAG answers grounded in articles; escalation on no match; confidence scoring | `agents/knowledge.py` (relevance threshold per mode and LLM answer confidence); notebook 03 §A3, §B1 and §B5; `test_retrieval.py`; the live calibration test |
| At least 2 support tools abstracting the CultPass DB, with error handling | 16 CultPass and 10 UDA-Hub tools over MCP; notebook 03 §A4–A4b; `test_cultpass_tools.py`; `test_support_operations.py` |
| Tools integrated into the workflow | account agent ⇄ `account_tools` loop; notebook 03 §B2–B4; `test_workflow.py` |
| Persistent interaction history and personalisation | `ticket_messages` and `customer_memories`; notebook 03 §A5 and §B7; `test_long_term_memory_is_recalled_in_a_new_thread` |
| Short-term memory and state inspection by `thread_id` | checkpointer and `trace` (tool calls with outcomes); notebook 03 "Inspect state"; `test_short_term_memory_across_turns_in_a_thread` |
| End-to-end flow, both resolution and escalation, error handling | notebook 03 Part B (executed), `tests/test_workflow.py`, `tests/test_live_llm.py`, `03_agentic_app.py --demo` |
| Structured, searchable logs of decisions, routing, tools and outcomes | `logs/agent_events.jsonl` via `agentic/logging_utils.py`; notebook 03 last cells; `test_every_turn_is_logged` |
| **Stand-out: advanced knowledge retrieval** | Hybrid embeddings and TF-IDF with a persistent vector store (`knowledge_embeddings`), a calibrated blend and thresholds; design §5 |
| **Stand-out: multiple channels** | `agentic/channels.py` (e-mail, chat, social); notebook 03 §B6; `test_channels.py`; design §10 |
| **Stand-out: sentiment analysis** | classifier sentiment drives routing, P1–P4 priority, escalation priority and work-queue order; design §8 |
| **Stand-out: A/B testing framework** | `agentic/experiments.py`, `agentic/evaluation.py`, labelled set, `ab_test_report.md`; notebook 03 §B9; `test_experiments.py`; design §9 |
| **Stand-out: MCP tools for support operations** | two FastMCP servers with 26 tools: refunds, subscription management, account changes, staff approvals, work queue; design §7 |

## Built with

- [LangGraph](https://langchain-ai.github.io/langgraph/) for the state graph, checkpointer and short-term memory
- [LangChain](https://python.langchain.com/) with OpenAI `gpt-4o-mini` and `text-embedding-3-small`
- [FastMCP](https://gofastmcp.com/) for the MCP servers and client
- [SQLAlchemy](https://www.sqlalchemy.org/) and SQLite for the CultPass and UDA-Hub databases

## Submission notes

- The solution doesn't import or reference anything outside `solution/`.
- `.env`, the `*.db` files, `logs/` and `.venv/` are excluded by `.gitignore`. To regenerate the
  databases, run notebooks 01 and 02.
