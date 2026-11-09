from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
import re
import operator
from schemas import (
    UserIntent, SessionState,
    AnswerResponse, SummarizationResponse, CalculationResponse, UpdateMemoryResponse
)
from prompts import get_intent_classification_prompt, get_chat_prompt_template, MEMORY_SUMMARY_PROMPT


# TODO: The AgentState class is already implemented for you.  Study the
# structure to understand how state flows through the LangGraph
# workflow.  See README.md Task 2.1 for detailed explanations of
# each property.
class AgentState(TypedDict):
    """
    The agent state object
    """
    # Current conversation
    user_input: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

    # Intent and routing
    intent: Optional[UserIntent]
    next_step: str

    # Memory and context
    conversation_summary: str
    active_documents: Optional[List[str]]

    # Current task state
    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]

    # Session management
    session_id: Optional[str]
    user_id: Optional[str]

    # Accumulates the names of each agent node that runs (operator.add reducer)
    actions_taken: Annotated[List[str], operator.add]


def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools) -> (
Dict[str, Any], List[str]):
    llm_with_tools = llm.bind_tools(
        tools
    )

    agent = create_react_agent(
        model=llm_with_tools,  # Use the bound model
        tools=tools,
        response_format=response_schema,
    )

    result = agent.invoke({"messages": messages})
    # Only count tools called during this turn; earlier messages are chat history
    new_messages = result.get("messages", [])[len(messages):]
    tools_used = [t.name for t in new_messages if isinstance(t, ToolMessage)]

    return result, tools_used


# Maps a classified intent to the agent node that handles it
INTENT_TO_NODE = {
    "qa": "qa_agent",
    "summarization": "summarization_agent",
    "calculation": "calculation_agent",
}


def _format_conversation_history(messages: List[BaseMessage], max_messages: int = 10) -> str:
    """Render recent user/assistant messages as plain text for the intent classifier."""
    lines = []
    for message in messages:
        # Skip system prompts, tool calls/results and empty messages
        if isinstance(message, HumanMessage):
            role = "User"
        elif isinstance(message, AIMessage) and message.content:
            role = "Assistant"
        else:
            continue
        lines.append(f"{role}: {message.content}")

    if not lines:
        return "No previous conversation."
    return "\n".join(lines[-max_messages:])


def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """

    llm = config.get("configurable").get("llm")
    history = state.get("messages", [])

    # Configure the llm chat model for structured output
    structured_llm = llm.with_structured_output(UserIntent)

    # Create a formatted prompt with conversation history and user input
    prompt = get_intent_classification_prompt().format(
        user_input=state["user_input"],
        conversation_history=_format_conversation_history(history),
    )

    intent = structured_llm.invoke(prompt)

    # Route to the matching agent; "unknown" (or anything else) falls back to Q&A
    next_step = INTENT_TO_NODE.get(intent.intent_type, "qa_agent")

    return {
        "actions_taken": ["classify_intent"],
        "intent": intent,
        "next_step": next_step,
    }


def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle Q&A tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("qa")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(AnswerResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["qa_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle summarization tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("summarization")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(SummarizationResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["summarization_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle calculation tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("calculation")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(CalculationResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["calculation_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def update_memory(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """

    # Retrieve the LLM from config
    llm = config.get("configurable").get("llm")

    prompt_with_history = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
        MessagesPlaceholder("chat_history"),
    ]).invoke({
        "chat_history": state.get("messages", []),
    })

    structured_llm = llm.with_structured_output(UpdateMemoryResponse)

    response = structured_llm.invoke(prompt_with_history)
    return {
        "conversation_summary": response.summary,
        "active_documents": response.document_ids,
        "next_step": "end",
        "actions_taken": ["update_memory"],
    }


def should_continue(state: AgentState) -> str:
    """Router function"""
    return state.get("next_step", "end")


def create_workflow(llm, tools):
    """
    Creates the LangGraph agents.
    Compiles the workflow with an InMemorySaver checkpointer to persist state.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent",
            "end": END
        }
    )

    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")

    workflow.add_edge("update_memory", END)

    # Register our Pydantic schemas so the checkpointer can restore them
    # (intent and current_response hold these objects) without warnings
    serde = JsonPlusSerializer(allowed_msgpack_modules=[
        ("schemas", schema.__name__)
        for schema in (UserIntent, AnswerResponse, SummarizationResponse,
                       CalculationResponse, UpdateMemoryResponse)
    ])
    return workflow.compile(checkpointer=InMemorySaver(serde=serde))
