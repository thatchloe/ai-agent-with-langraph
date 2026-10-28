# EcoHome Energy Advisor

An AI-powered energy optimization agent that helps customers reduce electricity costs and environmental impact through personalized recommendations.

## Project Scenario

You've joined EcoHome, a cutting-edge smart-home energy startup that's revolutionizing how customers manage their energy consumption. Your team is building the next generation of AI-powered energy optimization systems that help customers with solar panels, electric vehicles, and smart home devices reduce their electricity costs and environmental impact.

The goal? Build an intelligent Energy Advisor agent that can analyze real-time data, understand energy patterns, and provide personalized recommendations for optimal device scheduling and energy usage.

You'll need to design an agent system that can:

- Analyze energy usage patterns and solar generation data
- Provide personalized recommendations for when to run devices
- Suggest energy-saving strategies based on weather forecasts and electricity prices
- Help customers maximize their solar power usage
- Calculate potential savings from optimizations

Your agent should not only provide advice, it should make data-driven decisions about energy optimization!

## Project Introduction

In this project, you will develop the EcoHome Energy Advisor, an intelligent agent capable of optimizing energy usage across multiple smart home devices and systems.

### Key Capabilities

- **Multi-Tool Integration**: Weather forecasts, electricity pricing, energy usage queries, and solar generation data
- **RAG Pipeline**: Retrieval-augmented generation for energy-saving tips and best practices
- **Historical Analysis**: Query past energy usage patterns for personalized advice
- **Cost Optimization**: Consider time-of-day pricing and solar generation for recommendations
- **Savings Calculations**: Provide specific savings estimates and ROI analysis

### Inputs

- Energy usage data (consumption, device types, costs)
- Solar generation data (production, weather conditions)
- Weather forecasts and electricity pricing
- Energy-saving knowledge base
- User questions about energy optimization

### Deliverables

A LangGraph-powered agent system that:

- Understands energy optimization questions
- Retrieves relevant data using appropriate tools
- Provides personalized recommendations based on data analysis
- Cites relevant energy-saving tips from the knowledge base
- Calculates potential savings and environmental impact

## Project Structure

```
ecohome_solution/
├── models/
│   ├── __init__.py
│   └── energy.py              # Database models for energy data
├── data/
│   ├── documents/             # Knowledge base for the RAG pipeline
│   │   ├── tip_device_best_practices.txt        (starter)
│   │   ├── tip_energy_savings.txt               (starter)
│   │   ├── tip_hvac_optimization.txt            (added)
│   │   ├── tip_smart_home_automation.txt        (added)
│   │   ├── tip_renewable_energy_integration.txt (added)
│   │   ├── tip_seasonal_energy_management.txt   (added)
│   │   ├── tip_energy_storage_optimization.txt  (added)
│   │   ├── tip_ev_charging.txt                  (added)
│   │   └── tip_time_of_use_pricing.txt          (added)
│   ├── energy_data.db         # SQLite database (created by 01_db_setup.ipynb)
│   └── vectorstore/           # ChromaDB vector store (created by 02_rag_setup.ipynb)
├── agent.py                   # Main Energy Advisor agent and system prompt
├── tools.py                   # Agent tools (weather, pricing, database, RAG, savings)
├── requirements.txt           # Python dependencies
├── 01_db_setup.ipynb          # Database setup and sample data
├── 02_rag_setup.ipynb         # RAG pipeline setup
├── 03_run_and_evaluate.ipynb  # Agent running, testing and evaluation
└── README.md                  # This file
```

## Solution Overview

### Tools (`tools.py`)

- **`get_weather_forecast`** (implemented): mock forecast for 1-7 days, deterministic per location and date so repeated calls agree. Returns current conditions, hourly temperature, condition, solar irradiance and **estimated solar generation** for a 5 kW system, plus a per-day summary with total expected solar kWh and the best solar window.
- **`get_electricity_prices`** (implemented): mock time-of-use tariff around a $0.15/kWh base rate. Off-peak 22:00-06:00 (0.6x), peak 16:00-21:00 on weekdays (1.8x, with demand charge), mid-peak otherwise. Returns hourly rates, the rate of each period and the cheapest and most expensive hours. An optional `hour` argument returns the exact rate for a time the customer names (e.g. 7 PM). Invalid dates or hours return an error message.
- **`query_energy_usage` / `query_solar_generation`**: now also return per-device breakdowns, daily totals, average hour-of-day profiles and generation by weather. Individual records are capped at the 48 most recent so long ranges don't flood the model's context (totals still cover every record). The device filter is case-insensitive.
- **`get_recent_energy_summary`**: reports the actual dominant weather (it was always "sunny"), net grid consumption and solar coverage.
- **`search_energy_tips`**: indexes **every** `.txt` file in `data/documents/`, caches the vector store between calls and returns the source file name so the agent can cite it.
- **`calculate_energy_savings`**: supports load shifting (`optimized_price_per_kwh`), configurable frequency (`occurrences_per_year`) and reports monthly/annual savings and CO2 avoided.
- Sample data in `01_db_setup.ipynb` uses realistic daily totals (EV ~10 kWh charged in the evening, HVAC ~15-20 kWh, a few appliance runs) priced with the same time-of-use tariff as the price tool.
- All tools resolve data paths relative to `tools.py` and return `{"error": ...}` instead of raising.

### Agent (`agent.py`)

- **LLM**: `gpt-4o-mini` via the Vocareum endpoint, with the tools bound to it.
- **Graph**: an explicit LangGraph `StateGraph`:
  - **Schema**: `AgentState` with a `messages` list that `add_messages` appends to.
  - **Nodes**: `energy_advisor`, which calls the LLM with the system instructions, and `tools`, a `ToolNode` that runs the requested tool calls.
  - **Edges**: `START -> energy_advisor`. From `energy_advisor`, a conditional edge goes to `tools` when the model requested tools and to `END` otherwise. `tools -> energy_advisor` loops back.
- **Contract**: `Agent(instructions, model)` and `invoke(question, context)`. The answer is in `response["messages"][-1].content`.
- **Error handling**: a clear error if `VOCAREUM_API_KEY` is missing, LLM retries and timeout, tool errors returned to the model instead of crashing, empty-question validation and a step limit. Any failure comes back as a friendly message in the same response shape.
- **Context awareness**: every call adds the current date and time, tomorrow's date, the next 7 days, the location (defaults to San Francisco, CA) and a configurable home profile, so "tomorrow" or "Wednesday" resolve correctly. Passing the same `thread_id` keeps conversation memory for follow-up questions.

### Evaluation (`03_run_and_evaluate.ipynb`)

- **`ECOHOME_SYSTEM_PROMPT`** is written in the notebook and passed to `Agent(instructions=ECOHOME_SYSTEM_PROMPT)`. It covers the advisor's role, key capabilities (tools), the steps to follow, how to make recommendations and example questions.
- 12 test cases: EV charging, thermostat during a price spike, usage-history recommendations, dishwasher savings, pool pump scheduling, solar self-consumption, recent energy balance, EV load-shift savings, HVAC cost, home battery strategy, laundry timing and an off-topic request.
- `test_results` logs each test's final response, the tools the agent called with their arguments and result previews, errors and duration.
- **`evaluate_response()`**: LLM-as-judge (`gpt-4o`, a stronger model than the agent's `gpt-4o-mini`) with strict rules, scoring **accuracy, relevance, completeness, usefulness** and reasoning (1-5), with feedback on strengths, weaknesses and suggestions. The judge also sees the tool outputs so it can check the numbers.
- **`evaluate_tool_usage()`**: **tool appropriateness** (share of the tools called that were the right ones) and **tool completeness** (share of the necessary tools that were called), with feedback on missing, extra, failed and repeated calls.
- **`generate_evaluation_report()`** returns a structured report (overall metrics, per-criterion scores, strengths, weaknesses, recommendations, per-test details) and **`display_evaluation_report()`** presents it.

## Setup Instructions

### 1. Install Dependencies

The Project Workspace already has every package this project needs, so there is nothing to install there.

On your own computer, use Python 3.11, 3.12 or 3.13 and install the pinned versions:

```bash
pip install -r requirements.txt
```

No packages beyond `requirements.txt` are needed. The solution was developed and the non-LLM parts tested locally with Python 3.13.13 and exactly the versions pinned in `requirements.txt`.

### 2. Set Environment Variables

Create a `.env` file with your Vocareum API key:

```bash
VOCAREUM_API_KEY=your_vocareum_api_key_here
```

### 3. Run the Notebooks

Execute the notebooks in order:

1. **01_db_setup.ipynb** - Set up the database and populate with sample data
2. **02_rag_setup.ipynb** - Configure the RAG pipeline for energy tips
3. **03_run_and_evaluate.ipynb** - Run, test and evaluate the agent

## Agent Capabilities

### Tools Available

- **Weather Forecast**: Get hourly weather predictions and solar irradiance
- **Electricity Pricing**: Access time-of-day pricing data
- **Energy Usage Query**: Retrieve historical consumption data
- **Solar Generation Query**: Get past solar production data
- **Energy Tips Search**: Find relevant energy-saving recommendations
- **Savings Calculator**: Compute potential cost savings

### Example Questions

The Energy Advisor can answer questions like:

- "When should I charge my electric car tomorrow to minimize cost and maximize solar power?"
- "What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?"
- "Suggest three ways I can reduce energy use based on my usage history."
- "How much can I save by running my dishwasher during off-peak hours?"

## Database Schema

### Energy Usage Table
- `timestamp`: When the energy was consumed
- `consumption_kwh`: Amount of energy used
- `device_type`: Type of device (EV, HVAC, appliance)
- `device_name`: Specific device name
- `cost_usd`: Cost at time of usage

### Solar Generation Table
- `timestamp`: When the energy was generated
- `generation_kwh`: Amount of solar energy produced
- `weather_condition`: Weather during generation
- `temperature_c`: Temperature at time of generation
- `solar_irradiance`: Solar irradiance level

## Learning Objectives

This project helps students learn:

1. **Database Design**: Creating schemas for energy management systems
2. **API Integration**: Working with external weather and pricing APIs
3. **RAG Implementation**: Building retrieval-augmented generation pipelines
4. **Agent Development**: Creating intelligent agents with tool usage
5. **Evaluation Methods**: Testing and measuring agent performance
6. **Energy Optimization**: Understanding smart home energy management

## Key Technologies

- **LangChain**: Agent framework and tool integration
- **LangGraph**: Agent orchestration and workflow
- **ChromaDB**: Vector database for document retrieval
- **SQLAlchemy**: Database ORM and management
- **OpenAI**: LLM and embeddings
- **SQLite**: Local database storage

## Evaluation Criteria

The agent is evaluated on:

- **Accuracy**: Correct information and calculations
- **Relevance**: Responses address the user's question
- **Completeness**: Comprehensive answers with actionable advice
- **Tool Usage**: Appropriate use of available tools
- **Reasoning**: Clear explanation of recommendations

## Getting Started

1. Clone this repository
2. Install the required dependencies
3. Set up your environment variables
4. Run the notebooks in sequence
5. Test the agent with your own questions

## Contributing

This is a learning project. Feel free to:
- Add new tools and capabilities
- Improve the evaluation metrics
- Enhance the RAG pipeline
- Add more sophisticated optimization algorithms

## License

This project is for educational purposes as part of the Udacity Course 2 curriculum.
