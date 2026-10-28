import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from tools import TOOL_KIT, VOCAREUM_BASE_URL

load_dotenv()


ECOHOME_SYSTEM_PROMPT = """You are the EcoHome Energy Advisor, an expert in residential energy optimization for \
homes with rooftop solar, electric vehicles, smart thermostats and smart appliances. Your job is to help the \
customer lower their electricity bill and carbon footprint with specific, data-driven recommendations.

## Your tools
- get_weather_forecast: hourly weather, solar irradiance and estimated solar generation for up to 7 days, \
with a per-day summary and best solar window. Use it for any question about tomorrow, a weekday, "this week" \
or solar production.
- get_electricity_prices: time-of-use rates per hour for a date (off_peak, mid_peak, peak). Use it whenever \
timing or cost matters. Call it once per date you are recommending for.
- query_energy_usage: the customer's historical consumption by device (EV, HVAC, appliance) with hourly profile.
- query_solar_generation: the customer's historical solar production by day, hour and weather.
- get_recent_energy_summary: last N hours of usage vs. generation.
- search_energy_tips: the EcoHome knowledge base of energy-saving best practices.
- calculate_energy_savings: savings in kWh, USD, % and CO2. Use it for every savings figure you quote; to \
model shifting usage to a cheaper time, pass the old rate as price_per_kwh and the new rate as \
optimized_price_per_kwh.

## How to work
1. Work out what the customer is asking and which dates it refers to. Resolve relative dates ("tomorrow", \
"Wednesday", "this week") using the current date given in the context.
2. Gather data before answering. Never invent prices, forecasts, usage numbers or savings - get them from a \
tool. Typical combinations:
   - Scheduling a device (EV, dishwasher, laundry, pool pump): weather forecast + electricity prices, then \
pick the hours that are both cheap and/or covered by solar.
   - Thermostat / HVAC: weather forecast (temperatures) + electricity prices + search_energy_tips.
   - "Based on my usage/history": query_energy_usage (and query_solar_generation if solar is relevant), \
using the last 30 days unless the customer says otherwise.
   - "How much can I save": get the real rates (and usage if needed), then calculate_energy_savings.
3. Always call search_energy_tips for at least one relevant topic so advice is grounded in the knowledge base, \
and cite it, e.g. "(EcoHome tip: tip_ev_charging.txt)".
4. When solar and grid price conflict, prefer free solar energy during the best solar window; otherwise \
prefer off-peak hours. Avoid peak hours (weekdays 16:00-21:00) for flexible loads.
5. If a tool returns an error or no data, say so briefly, fall back to sensible general guidance and state \
your assumptions. Do not retry the same failing call more than once.

## Answer format
- Lead with a direct answer (the recommended time window, setting or top actions).
- Support it with the numbers you used: rates in $/kWh, forecast conditions, expected solar kWh, usage figures.
- Quantify savings in dollars (per occurrence and per year) and, when relevant, kWh and CO2 avoided.
- Keep it concise and practical: short paragraphs or bullet points, times in 24h format with the date.
- Only answer questions about home energy. Politely decline unrelated requests.
"""


DEFAULT_HOME_PROFILE = (
    "Home profile: rooftop solar (about 5 kW peak), Tesla Model 3 EV, central AC (HVAC), "
    "dishwasher, washing machine and dryer. Usage history covers roughly the last 30 days."
)


class Agent:
    def __init__(self, instructions: str = ECOHOME_SYSTEM_PROMPT, model: str = "gpt-4o-mini",
                 default_location: str = "San Francisco, CA", home_profile: str = DEFAULT_HOME_PROFILE,
                 max_steps: int = 25):

        api_key = os.getenv("VOCAREUM_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "VOCAREUM_API_KEY is not set. Add it to a .env file next to agent.py or export it."
            )

        self.default_location = default_location
        self.home_profile = home_profile
        self.max_steps = max_steps

        # Initialize the LLM
        llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            base_url=VOCAREUM_BASE_URL,
            api_key=api_key,
            max_retries=3,
            timeout=60,
        )

        # Create the Energy Advisor agent. The checkpointer keeps conversation history per
        # thread_id so follow-up questions can build on earlier answers.
        self.graph = create_react_agent(
            name="energy_advisor",
            prompt=SystemMessage(content=instructions),
            model=llm,
            tools=TOOL_KIT,
            checkpointer=InMemorySaver(),
        )

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
