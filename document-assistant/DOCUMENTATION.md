# DocDacity Document Assistant — Implementation Documentation

This document explains how the completed assistant works: the implementation
decisions, how state and memory flow through the LangGraph workflow, how
structured outputs are enforced, and example conversations covering every
feature. The original task instructions are in [README.md](README.md).

## Contents

1. [Quick start](#1-quick-start)
2. [Architecture](#2-architecture)
3. [State and memory](#3-state-and-memory)
4. [Structured outputs](#4-structured-outputs)
5. [Tools](#5-tools)
6. [Prompt engineering](#6-prompt-engineering)
7. [Implementation decisions and fixes to the starter code](#7-implementation-decisions-and-fixes-to-the-starter-code)
8. [Example conversations](#8-example-conversations)
9. [Generated logs and sessions](#9-generated-logs-and-sessions)
10. [Known limitations](#10-known-limitations)

---

## 1. Quick start

```bash
cp .env.example .env              # then set OPENAI_API_KEY
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python main.py                    # interactive assistant
python tests/test_offline.py      # 12 offline tests, no API key needed
```

The offline tests cover schema validation (confidence bounds, allowed intent
types, type enforcement), the calculator (results, unsafe input, error
handling, logging), prompt selection, and the compiled graph's nodes and edges.

---

## 2. Architecture

```
                    ┌──────────────────┐
     user input ──▶ │ classify_intent  │  LLM + UserIntent structured output
                    └────────┬─────────┘
                             │ should_continue(state) reads state["next_step"]
          ┌──────────────────┼────────────────────────┐
          ▼                  ▼                        ▼
   ┌────────────┐   ┌──────────────────────┐   ┌───────────────────┐
   │  qa_agent  │   │ summarization_agent  │   │ calculation_agent │
   │ AnswerResp │   │ SummarizationResp    │   │ CalculationResp   │
   └─────┬──────┘   └──────────┬───────────┘   └─────────┬─────────┘
         └─────────────────────┼─────────────────────────┘
                               ▼
                    ┌──────────────────┐
                    │  update_memory   │  LLM + UpdateMemoryResponse
                    └────────┬─────────┘
                             ▼
                            END
```

| Node | File | What it does |
|---|---|---|
| `classify_intent` | `src/agent.py` | Formats the intent prompt with the user input and the last 10 user/assistant messages, calls the LLM with `UserIntent` structured output, and maps the intent to the next node (`qa`→`qa_agent`, `summarization`→`summarization_agent`, `calculation`→`calculation_agent`, anything else→`qa_agent`). |
| `qa_agent`, `summarization_agent`, `calculation_agent` | `src/agent.py` | Each builds a chat prompt for its intent (system prompt + chat history + user input) and runs a ReAct agent (`create_react_agent`) with all four tools and its own response schema. |
| `update_memory` | `src/agent.py` | Summarizes the conversation and extracts the active document IDs with `UpdateMemoryResponse` structured output. |

Routing uses `add_conditional_edges("classify_intent", should_continue, {...})`.
`should_continue` returns `state["next_step"]`, which `classify_intent` set to
a node name. Every agent node has a fixed edge to `update_memory`, which has an
edge to `END`.

The three agent nodes are intentionally identical apart from the prompt and
the response schema. That makes the routing the only thing that differs
between intents, and keeps each agent easy to compare against `qa_agent`.

---

## 3. State and memory

### AgentState

`AgentState` (a `TypedDict` in `src/agent.py`) is the object every node reads
and returns updates to. Nodes return only the keys they change, and LangGraph
merges them into the state.

| Field | Reducer | Set by |
|---|---|---|
| `user_input` | overwrite | `DocumentAssistant.process_message` |
| `messages` | `add_messages` (appends; de-duplicates by message ID) | agent nodes |
| `intent`, `next_step` | overwrite | `classify_intent` (`update_memory` sets `next_step="end"`) |
| `conversation_summary`, `active_documents` | overwrite | `update_memory` |
| `current_response`, `tools_used` | overwrite | agent nodes |
| `session_id`, `user_id` | overwrite | `process_message` |
| `actions_taken` | **`operator.add`** (list concatenation) | every node appends its own name |

Because `actions_taken` uses `operator.add`, each node returns a one-item
list (for example `["classify_intent"]`) and LangGraph concatenates it onto
the existing list instead of replacing it. A single turn therefore produces
`classify_intent -> qa_agent -> update_memory`.

### Persistent memory with InMemorySaver

`create_workflow` compiles the graph with `checkpointer=InMemorySaver()`.
`process_message` passes a config whose `thread_id` is the current session ID,
together with the LLM and tools:

```python
config = {"configurable": {
    "thread_id": self.current_session.session_id,
    "llm": self.llm,
    "tools": self.tools,
}}
```

The checkpointer saves the state after every node, keyed by `thread_id`. On
the next `invoke` with the same `thread_id`, LangGraph loads the saved state
first and then applies the new input. This means:

- **`messages` accumulate across turns.** The new turn passes
  `"messages": []`, and the `add_messages` reducer appends nothing, so the
  full history is preserved and given to the agents as `chat_history`. This
  is how "What is 10 percent of the largest one?" is resolved in the example
  below.
- **`actions_taken` accumulates across turns too**, since its reducer adds the
  (empty) initial list to the saved one. The ACTIONS TAKEN line in the
  example below grows by three entries per turn.
- **The conversation summary is reused.** `_get_conversation_summary` reads
  it back from `workflow.get_state(config)`.

Different sessions use different `thread_id`s, so their memory is isolated.
LLM and tools are passed through `config["configurable"]` rather than stored
in state, so the state remains plain data that the checkpointer can
serialize.

### Two layers of memory

| Layer | Where | Lifetime |
|---|---|---|
| Checkpointer (`InMemorySaver`) | process memory | Lasts as long as the Python process runs; this is what the agents actually use |
| Session file (`sessions/<session_id>.json`) | disk | Permanent record of each turn's final state, saved by `_save_session` |

---

## 4. Structured outputs

Every LLM call that must return data uses `llm.with_structured_output(Schema)`
or `create_react_agent(..., response_format=Schema)`. Both make the model
return JSON that matches the Pydantic schema, which is then validated.

| Where | Schema | Purpose |
|---|---|---|
| `classify_intent` | `UserIntent` | intent type, confidence, reasoning |
| `qa_agent` | `AnswerResponse` | question, answer, source IDs, confidence, timestamp |
| `summarization_agent` | `SummarizationResponse` | summary, key points, document IDs |
| `calculation_agent` | `CalculationResponse` | expression, result, explanation, units |
| `update_memory` | `UpdateMemoryResponse` | conversation summary, active document IDs |

The two schemas implemented for this project (`src/schemas.py`) enforce their
constraints with Pydantic, not only by describing them:

```python
class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)      # rejects 1.5 or -0.1
    timestamp: datetime = Field(default_factory=datetime.now)

class UserIntent(BaseModel):
    intent_type: Literal["qa", "summarization", "calculation", "unknown"]  # rejects "weather"
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
```

`Literal` restricts `intent_type` to the four valid options, and `ge`/`le`
bound confidence to 0–1. Field descriptions are included in the JSON schema
sent to the model, so they also guide what it generates. The offline tests
check that invalid values raise `ValidationError`.

---

## 5. Tools

All tools use the `@tool` decorator, log each use with `ToolLogger`, catch
their own errors and return a string.

| Tool | Purpose |
|---|---|
| `calculator` | Evaluates arithmetic (implemented for this project) |
| `document_search` | Finds documents by keyword, type or amount |
| `document_reader` | Returns a document's full text by ID |
| `document_statistics` | Counts and amount totals across the collection |

### Calculator safety

The calculator uses `eval()`, so input is validated in two layers first:

1. **Character whitelist:** after removing `$` and thousands-separator commas,
   only digits, whitespace, `.`, `( )` and `+ - * / % **` are allowed. This
   rejects names, quotes, brackets and attribute access, so
   `__import__('os')` never reaches `eval`.
2. **Syntax tree check:** the expression is parsed with `ast.parse`, and only
   number, unary-operator and binary-operator nodes are allowed. Exponents
   must be a plain number no larger than 100, which blocks expressions like
   `9**9**9**9` that would freeze the process.

`eval` then runs with `{"__builtins__": {}}`. The result is returned as a
string of the number, such as `"5"` or `"2.5"`; whole-number floats are shown
without `.0`. Errors (division by zero, bad syntax, unsafe input) return a
message starting with `Error` instead of raising, so the agent can recover.
Every call, successful or not, is logged.

### Per-session tool logs

`DocumentAssistant.start_session` calls `ToolLogger.set_session(session_id)`,
so tool calls are written to `logs/session_<session_id>.json`. Resuming a
session reloads its existing log instead of overwriting it.

---

## 6. Prompt engineering

### Intent classification (`get_intent_classification_prompt`)

The classifier prompt contains:

- **Clear categories:** `qa`, `summarization`, `calculation`, `unknown`, each
  with a definition. Questions that need arithmetic go to `calculation` even
  when they look like questions.
- **Examples:** nine labelled requests, including borderline cases such as
  "How much more is INV-003 than INV-002?" (calculation, because it needs a
  subtraction).
- **Follow-up handling:** an instruction to use the conversation history to
  resolve references like "it" or "that one".
- **Confidence scoring:** explicit bands (0.9–1.0 clear, 0.7–0.89 minor
  ambiguity, 0.5–0.69 ambiguous, below 0.5 use `unknown`).
- **Reasoning:** one or two sentences naming the words that decided the
  category and any ambiguity.

### Dynamic chat prompt selection (`get_chat_prompt_template`)

```python
if intent_type == "qa":              system_prompt = QA_SYSTEM_PROMPT
elif intent_type == "summarization": system_prompt = SUMMARIZATION_SYSTEM_PROMPT
elif intent_type == "calculation":   system_prompt = CALCULATION_SYSTEM_PROMPT
else:                                system_prompt = QA_SYSTEM_PROMPT  # fallback
```

It returns `ChatPromptTemplate.from_messages([system prompt,
MessagesPlaceholder("chat_history"), "{input}"])`.

### Calculation system prompt

`CALCULATION_SYSTEM_PROMPT` gives the agent a fixed procedure:

1. Find the document(s) with the needed data, using search if the ID is unknown.
2. Read each one with `document_reader`, copying figures digit-for-digit.
3. Write the expression and check each number against the document.
4. Use the calculator for **every** calculation, however simple.
5. Report the result, expression, explanation and source IDs.

Step 2's "digit-for-digit" rule was added after testing: without it, the model
once typed 69000 instead of 69300. A guideline also covers documents without
an explicit total (INV-001 lists only subtotal and tax), telling the model to
add the components with the calculator.

---

## 7. Implementation decisions and fixes to the starter code

**Decisions**

- `classify_intent` gives the classifier only user and assistant text (last
  10 messages), not system prompts or raw tool output. This keeps the prompt
  short and focused on what was asked.
- An `unknown` intent is routed to `qa_agent`, which can still respond
  politely (see "hello" below) instead of failing.
- `update_memory` also appends `"update_memory"` to `actions_taken`, so the
  trace shows every node that ran.
- The checkpointer is given a serializer that explicitly allows this project's
  schema classes (`allowed_msgpack_modules`). This silences LangGraph's
  "unregistered type" warning that otherwise printed on every turn.

**Starter-code bugs fixed while testing**

| Problem | Fix |
|---|---|
| `prompts.py` imported `langchain.prompts`, removed in LangChain 1.x | Import from `langchain_core.prompts` (works in all versions) |
| `should_continue` and `create_workflow` were indented inside `update_memory` | Moved to module level |
| `default_factory=lambda: list` returned the `list` type, not an empty list | `default_factory=list` (same for `dict`) |
| `tools_used` counted tool calls from the whole chat history | Count only messages produced in the current turn |
| Keyword search could not find a document by its ID (e.g. "CLM-001") | Match document IDs and types |
| Plural searches ("invoices") never matched ("invoice") | Strip a trailing "s" from search words |
| `search_type="all"` results were overwritten by the fallback branch | Changed `if` to `elif` |
| `main.py` never printed SOURCES (read the wrong result key) | Read `result["sources"]`; also print ACTIONS TAKEN |
| `process_message` read the session ID before checking a session exists | Check first, then build the config |
| Tool logs were per process, not per session | `ToolLogger.set_session` |

---

## 8. Example conversations

A real session run with `python main.py` (gpt-4o). Long answers are shortened
with `…`; nothing else is edited. The logs and session file from this run are
committed in `logs/` and `sessions/`.

**Turn 1: Q&A with document lookup**
```
Enter Message: What is the status of claim CLM-001?
🤖 Assistant: The status of claim CLM-001 is "Under Review."
INTENT: qa
SOURCES: CLM-001
TOOLS USED: document_reader
ACTIONS TAKEN: classify_intent -> qa_agent -> update_memory
```

**Turn 2: follow-up that relies on memory ("that claim")**
```
Enter Message: Who filed that claim and what was the largest expense?
🤖 Assistant: The claim CLM-001 was filed by John Doe. The largest expense in the
claim was for a Hospital Visit, amounting to $1,200.
INTENT: qa
SOURCES: CLM-001
ACTIONS TAKEN: classify_intent -> qa_agent -> update_memory -> classify_intent -> qa_agent -> update_memory
CONVERSATION SUMMARY: The conversation focused on the status and details of insurance
claim CLM-001. The claim, filed by John Doe, is for medical expense reimbursement
totaling $2,450. The largest expense was a hospital visit costing $1,200. The claim
is currently under review.
```
No tools were needed: the answer came from the claim already in the message
history.

**Turn 3: summarization**
```
Enter Message: Summarize all contracts
🤖 Assistant: ### Summary of Contract CON-001
**Title:** Service Agreement
**Date:** January 1, 2024
**Parties Involved:**
- **Provider:** DocDacity Solutions Inc.
- **Client:** Healthcare Partners LLC
…
**Financial Details:**
- **Monthly Fee:** $15,000
- **Total Contract Value:** $180,000
**Duration:** 12 months
**Termination Clause:** Either party may terminate the agreement with 60 days written notice.
INTENT: summarization
TOOLS USED: document_search, document_reader
```

**Turn 4: calculation across several documents**
```
Enter Message: Calculate the sum of all invoice totals
🤖 Assistant: The sum of all invoice totals is $305,800.
- **Invoice #12345:** Total $22,000 (Subtotal $20,000 + Tax $2,000)
- **Invoice #12346:** Total $69,300
- **Invoice #12347:** Total $214,500
Document IDs: INV-001, INV-002, INV-003.
INTENT: calculation
TOOLS USED: document_search, document_reader, document_reader, document_reader, calculator
```
Tool log entry: `calculator {'expression': '22000 + 69300 + 214500'} -> {'result': '305800'}`

**Turn 5: calculation reusing figures from memory**
```
Enter Message: How much more is INV-003 than INV-002?
🤖 Assistant: Invoice INV-003 is $145,200 more than Invoice INV-002.
INTENT: calculation
TOOLS USED: calculator
```
The agent reused the figures it had read in turn 4 and still did the
subtraction with the calculator (`214500 - 69300`).

**Turn 6: amount-based search**
```
Enter Message: Find documents with amounts over $50,000
🤖 Assistant: Here are the documents with amounts over $50,000:
1. **Invoice #12347 (ID: INV-003)**
2. **Service Agreement (ID: CON-001)**
3. **Invoice #12346 (ID: INV-002)**
INTENT: qa
TOOLS USED: document_search
```

**Turn 7: follow-up calculation ("the largest one")**
```
Enter Message: What is 10 percent of the largest one?
🤖 Assistant: 10 percent of the largest amount, which is $214,500 from Invoice #12347
(ID: INV-003), is $21,450.
INTENT: calculation
TOOLS USED: calculator
```

**Turn 8: unknown intent**
```
Enter Message: hello
🤖 Assistant: Hello! How can I assist you today with your financial or healthcare documents?
INTENT: unknown
ACTIONS TAKEN: … -> classify_intent -> qa_agent -> update_memory
```
`unknown` falls back to `qa_agent`.

---

## 9. Generated logs and sessions

Both directories are created automatically on first run.

- **`logs/session_<session_id>.json`**: every tool call in that session, with
  timestamp, tool name, input and output. For example:
  ```json
  {"tool_name": "calculator",
   "input": {"expression": "214500 - 69300"},
   "output": "{'result': '145200'}"}
  ```
- **`sessions/<session_id>.json`**: the session record (user ID, timestamps,
  active document IDs) plus the final state of every turn: user input,
  messages, intent, summary, tools used and actions taken.

---

## 10. Known limitations

- **Memory lasts only while the process runs.** `InMemorySaver` keeps state
  in RAM, so restarting `main.py` loses agent memory even though session files
  remain on disk. A persistent checkpointer (such as SQLite) would fix this.
- **Session files grow quickly.** The starter stores the full final state,
  including all messages so far, after every turn. The 8-turn example above
  produces a file of about 0.5 MB.
- **Classification of borderline requests.** "What's the total in INV-002?"
  is usually classified as `calculation`, because the provided category
  definitions send anything that *may* need math there. The answer is still
  correct.
- **LLM variability.** Answers are model-generated. For example, INV-001 has no
  "Total Due" line, and the prompt tells the model to add subtotal and tax,
  but a model may still occasionally treat the subtotal as the total.
