import os
import uuid
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from tools import TOOL_KIT, VOCAREUM_BASE_URL

load_dotenv()


DEFAULT_HOME_PROFILE = (
    "Home profile: rooftop solar (about 5 kW peak), Tesla Model 3 EV, central AC (HVAC), "
    "dishwasher, washing machine and dryer. Usage history covers roughly the last 30 days."
)


class AgentState(TypedDict):
    """Graph schema: the conversation so far. add_messages appends new messages instead of replacing them."""
    messages: Annotated[List[AnyMessage], add_messages]


class Agent:
    def __init__(self, instructions: str, model: str = "gpt-4o-mini",
                 default_location: str = "San Francisco, CA", home_profile: str = DEFAULT_HOME_PROFILE,
                 max_steps: int = 25):

        api_key = os.getenv("VOCAREUM_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "VOCAREUM_API_KEY is not set. Add it to a .env file next to agent.py or export it."
            )

        self.instructions = instructions
        self.default_location = default_location
        self.home_profile = home_profile
        self.max_steps = max_steps

        # Initialize the LLM and give it the tool schemas
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            base_url=VOCAREUM_BASE_URL,
            api_key=api_key,
            max_retries=3,
            timeout=60,
        ).bind_tools(TOOL_KIT)

        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Energy Advisor graph:

            START -> energy_advisor --(tool calls?)--> tools -> energy_advisor
                                    \\--(final answer)--> END
        """
        def energy_advisor(state: AgentState) -> Dict[str, Any]:
            # The system instructions are prepended on every call rather than stored in the state
            response = self.llm.invoke([SystemMessage(content=self.instructions)] + state["messages"])
            return {"messages": [response]}

        builder = StateGraph(AgentState)
        builder.add_node("energy_advisor", energy_advisor)
        # ToolNode runs every tool call in the last AI message; tool exceptions are returned to the
        # model as error messages so it can recover instead of crashing the run
        builder.add_node("tools", ToolNode(TOOL_KIT, handle_tool_errors=True))
        builder.add_edge(START, "energy_advisor")
        builder.add_conditional_edges("energy_advisor", tools_condition, {"tools": "tools", END: END})
        builder.add_edge("tools", "energy_advisor")

        # The checkpointer keeps conversation history per thread_id so follow-up questions
        # can build on earlier answers.
        return builder.compile(checkpointer=InMemorySaver(), name="energy_advisor")

    def build_context(self, context: Optional[str] = None) -> str:
        """Context the agent needs to resolve dates and personalize advice."""
        now = datetime.now()
        lines = [
            f"Current date and time: {now.strftime('%A, %Y-%m-%d %H:%M')}",
            f"Tomorrow is {(now + timedelta(days=1)).strftime('%A, %Y-%m-%d')}.",
            "Upcoming days: " + ", ".join(
                (now + timedelta(days=d)).strftime("%a %Y-%m-%d") for d in range(7)
            ),
        ]
        if not context or "location" not in context.lower():
            lines.append(f"Location: {self.default_location}")
        if self.home_profile:
            lines.append(self.home_profile)
        if context:
            lines.append(context)
        return "\n".join(lines)

    def invoke(self, question: str, context: str = None, thread_id: str = None) -> Dict[str, Any]:
        """
        Ask the Energy Advisor a question about energy optimization.

        Args:
            question (str): The user's question about energy optimization
            context (str): Extra context, e.g. "Location: San Francisco, CA"
            thread_id (str): Reuse the same id to continue a conversation; a new one is used if omitted

        Returns:
            Dict[str, Any]: The agent state; the advisor's answer is response["messages"][-1].content
        """
        if not question or not question.strip():
            return self._error_response("Please ask a question about your home energy use.")

        messages = [
            SystemMessage(content=self.build_context(context)),
            HumanMessage(content=question.strip()),
        ]

        config = {
            "configurable": {"thread_id": thread_id or str(uuid.uuid4())},
            "recursion_limit": self.max_steps,
        }

        # Get response from the agent
        try:
            return self.graph.invoke(input={"messages": messages}, config=config)
        except GraphRecursionError:
            return self._error_response(
                "I couldn't finish the analysis within the allowed number of steps. "
                "Try asking a narrower question, e.g. about one device or one day."
            )
        except Exception as e:
            return self._error_response(
                f"Sorry, the Energy Advisor ran into a problem ({type(e).__name__}: {e}). Please try again."
            )

    def ask(self, question: str, context: str = None, thread_id: str = None) -> str:
        """Ask a question and return only the advisor's final answer."""
        return self.invoke(question, context, thread_id)["messages"][-1].content

    @staticmethod
    def _error_response(message: str) -> Dict[str, Any]:
        # Same shape as a successful graph response so callers don't need special handling
        return {"messages": [AIMessage(content=message)], "error": message}

    def get_agent_tools(self):
        """Get list of available tools for the Energy Advisor"""
        return [t.name for t in TOOL_KIT]
