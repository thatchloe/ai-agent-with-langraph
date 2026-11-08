"""UDA-Hub orchestration graph (built from scratch with ``StateGraph``).

    START
      -> ticket_intake        (load ticket, customer, long-term memory)
      -> classifier           (LLM structured classification)
      -> supervisor           (rule-based routing)
           |-> knowledge_agent  -> finalize | escalation_agent
           |-> account_agent <-> account_tools
           |        -> finalize | escalation_agent
           |-> escalation_agent -> finalize
      -> finalize             (persist reply, ticket status, long-term memory)
    END

Short-term memory: the checkpointer stores the full state per ``thread_id``
(thread_id == ticket_id). Long-term memory: ``customer_memories`` table in
udahub.db, recalled at intake and written at finalize.

Usage::

    from agentic.workflow import orchestrator
    orchestrator.invoke(
        {"messages": [HumanMessage("How do I reserve an event?")], "external_user_id": "f556c0"},
        config={"configurable": {"thread_id": "ticket-123"}},
    )
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic import config
from agentic.agents.account import (
    AccountAgent,
    AccountToolExecutor,
    route_after_account_agent,
    route_after_tools,
)
from agentic.agents.classifier import ClassifierAgent
from agentic.agents.escalation import EscalationAgent
from agentic.agents.knowledge import KnowledgeAgent, route_after_knowledge
from agentic.agents.supervisor import Supervisor, route_from_supervisor
from agentic.agents.ticket_manager import TicketFinalizer, TicketIntake
from agentic.state import UdaHubState


@lru_cache(maxsize=1)
def _default_llm() -> BaseChatModel:
    return config.get_llm()


def build_graph(llm: BaseChatModel | None = None, checkpointer=None) -> CompiledStateGraph:
    """Compile the UDA-Hub multi-agent graph.

    ``llm`` is injectable (tests pass a scripted model); by default the OpenAI
    model is created lazily on first use, so importing this module never needs
    credentials.
    """
    provider = (lambda: llm) if llm is not None else _default_llm

    graph = StateGraph(UdaHubState)
    graph.add_node("ticket_intake", TicketIntake())
    graph.add_node("classifier", ClassifierAgent(provider))
    graph.add_node("supervisor", Supervisor(provider))
    graph.add_node("knowledge_agent", KnowledgeAgent(provider))
    graph.add_node("account_agent", AccountAgent(provider))
    graph.add_node("account_tools", AccountToolExecutor())
    graph.add_node("escalation_agent", EscalationAgent(provider))
    graph.add_node("finalize", TicketFinalizer())

    graph.add_edge(START, "ticket_intake")
    graph.add_edge("ticket_intake", "classifier")
    graph.add_edge("classifier", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "knowledge_agent": "knowledge_agent",
            "account_agent": "account_agent",
            "escalation_agent": "escalation_agent",
        },
    )
    graph.add_conditional_edges(
        "knowledge_agent",
        route_after_knowledge,
        {"finalize": "finalize", "escalation_agent": "escalation_agent"},
    )
    graph.add_conditional_edges(
        "account_agent",
        route_after_account_agent,
        {"account_tools": "account_tools", "finalize": "finalize", "escalation_agent": "escalation_agent"},
    )
    graph.add_conditional_edges(
        "account_tools",
        route_after_tools,
        {"account_agent": "account_agent", "escalation_agent": "escalation_agent"},
    )
    graph.add_edge("escalation_agent", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


orchestrator = build_graph()
