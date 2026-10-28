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
ecohome_starter/
├── models/
│   ├── __init__.py
│   └── energy.py              # Database models for energy data
├── data/
│   └── documents/
│       ├── tip_device_best_practices.txt
│       └── tip_energy_savings.txt
├── agent.py                   # Main Energy Advisor agent
├── tools.py                   # Agent tools (weather, pricing, database, RAG)
├── requirements.txt           # Python dependencies
├── 01_db_setup.ipynb         # Database setup and sample data
├── 02_rag_setup.ipynb        # RAG pipeline setup
├── 03_run_and_evaluate.ipynb # Agent running, testing and evaluation
└── README.md                  # This file
```

## Setup Instructions

### 1. Install Dependencies

The Project Workspace already has every package this project needs, so there is nothing to install there.

On your own computer, use Python 3.11, 3.12 or 3.13 and install the pinned versions:

```bash
pip install -r requirements.txt
```

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
