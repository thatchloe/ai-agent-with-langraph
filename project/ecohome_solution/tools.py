"""
Tools for EcoHome Energy Advisor Agent
"""
import os
import glob
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.energy import DatabaseManager

# Resolve data paths relative to this file so the tools work no matter where
# the notebook or script that imports them is started from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "energy_data.db")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
VECTORSTORE_DIR = os.path.join(DATA_DIR, "vectorstore")

VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"

# Household assumptions shared by the forecast and savings tools.
# The sample data in 01_db_setup.ipynb peaks at ~5 kWh per hour at noon on a sunny day.
SOLAR_SYSTEM_PEAK_KW = 5.0
PEAK_IRRADIANCE_W_M2 = 800.0
# Average US grid carbon intensity (EPA eGRID, ~0.39 kg CO2 per kWh)
GRID_CO2_KG_PER_KWH = 0.39

# Time-of-use tariff
BASE_PRICE_PER_KWH = 0.15
OFF_PEAK_HOURS = set(range(0, 6)) | {22, 23}
PEAK_HOURS = set(range(16, 21))  # 4 PM - 9 PM
PERIOD_MULTIPLIERS = {"off_peak": 0.6, "mid_peak": 1.0, "peak": 1.8}
PERIOD_DEMAND_CHARGES = {"off_peak": 0.0, "mid_peak": 0.02, "peak": 0.05}

# Records returned in full by the database tools are capped so a long date
# range doesn't flood the model's context; aggregates always cover every record.
MAX_RECORDS_RETURNED = 48

# Initialize database manager
db_manager = DatabaseManager(DB_PATH)


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _seeded_rng(*parts: Any) -> random.Random:
    """Deterministic RNG so the same location/date always yields the same mock data."""
    return random.Random("|".join(str(p).lower() for p in parts))


def _price_period(hour: int, is_weekend: bool) -> str:
    if hour in OFF_PEAK_HOURS:
        return "off_peak"
    if hour in PEAK_HOURS and not is_weekend:
        return "peak"
    return "mid_peak"


def _daylight_factor(hour: int) -> float:
    """0 at night, 1 at solar noon; daylight runs 6 AM - 6 PM like the historical data."""
    if 6 <= hour <= 18:
        return max(0.0, math.sin(math.pi * (hour - 6) / 12))
    return 0.0


@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days.
    Includes hourly solar irradiance and an estimate of the home's solar generation,
    so it can be used to plan devices around solar production.

    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast starting today (1-7)

    Returns:
        Dict[str, Any]: Weather forecast data including temperature, conditions, solar irradiance
        and estimated solar generation per hour, plus a per-day summary with the best solar window.
    """
    try:
        if not location or not location.strip():
            return {"error": "A location is required, e.g. 'San Francisco, CA'."}
        days = max(1, min(int(days), 7))

        # Mock weather API: deterministic per location and date
        condition_profiles = {
            "sunny": {"cloud_factor": 1.0, "weight": 0.45},
            "partly_cloudy": {"cloud_factor": 0.6, "weight": 0.3},
            "cloudy": {"cloud_factor": 0.3, "weight": 0.17},
            "rainy": {"cloud_factor": 0.1, "weight": 0.08},
        }
        conditions = list(condition_profiles)
        weights = [condition_profiles[c]["weight"] for c in conditions]

        now = datetime.now()
        rng_now = _seeded_rng(location, now.strftime("%Y-%m-%d"), "current")
        forecast = {
            "location": location,
            "forecast_days": days,
            "generated_at": now.isoformat(timespec="minutes"),
            "solar_system_peak_kw": SOLAR_SYSTEM_PEAK_KW,
            "current": {
                "temperature_c": round(rng_now.uniform(12, 24), 1),
                "condition": rng_now.choices(conditions, weights=weights)[0],
                "humidity": rng_now.randint(40, 85),
                "wind_speed": round(rng_now.uniform(2, 25), 1),
            },
            "daily_summary": [],
            "hourly": [],
        }

        for day_offset in range(days):
            day = (now + timedelta(days=day_offset)).date()
            date_str = day.isoformat()
            rng = _seeded_rng(location, date_str)
            day_condition = rng.choices(conditions, weights=weights)[0]
            low_c = rng.uniform(8, 16)
            high_c = low_c + rng.uniform(5, 13)

            day_hours = []
            for hour in range(24):
                # Hourly condition mostly follows the day, with some variability
                condition = day_condition if rng.random() < 0.8 else rng.choice(conditions)
                cloud_factor = condition_profiles[condition]["cloud_factor"]
                daylight = _daylight_factor(hour)
                # Temperature curve: coolest ~5 AM, warmest ~3 PM
                temp = low_c + (high_c - low_c) * (0.5 + 0.5 * math.sin(math.pi * (hour - 9) / 12))
                irradiance = PEAK_IRRADIANCE_W_M2 * daylight * cloud_factor * rng.uniform(0.9, 1.1)
                generation = SOLAR_SYSTEM_PEAK_KW * irradiance / PEAK_IRRADIANCE_W_M2
                entry = {
                    "date": date_str,
                    "hour": hour,
                    "temperature_c": round(temp, 1),
                    "condition": condition,
                    "solar_irradiance": round(irradiance, 1),
                    "estimated_solar_kwh": round(generation, 2),
                    "humidity": rng.randint(40, 90),
                    "wind_speed": round(rng.uniform(2, 25), 1),
                }
                day_hours.append(entry)
                forecast["hourly"].append(entry)

            producing = [h for h in day_hours if h["estimated_solar_kwh"] >= 0.5 * SOLAR_SYSTEM_PEAK_KW * condition_profiles[day_condition]["cloud_factor"]]
            forecast["daily_summary"].append({
                "date": date_str,
                "weekday": day.strftime("%A"),
                "condition": day_condition,
                "high_c": round(max(h["temperature_c"] for h in day_hours), 1),
                "low_c": round(min(h["temperature_c"] for h in day_hours), 1),
                "estimated_solar_generation_kwh": round(sum(h["estimated_solar_kwh"] for h in day_hours), 1),
                "best_solar_window": (
                    f"{producing[0]['hour']:02d}:00-{producing[-1]['hour'] + 1:02d}:00" if producing else None
                ),
            })

        return forecast
    except Exception as e:
        return {"error": f"Failed to get weather forecast: {str(e)}"}


@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get time-of-use electricity prices for a specific date or current day.

    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates, the period of each hour
        (off_peak, mid_peak, peak), and a summary of the cheapest and most expensive hours.
    """
    try:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        day = _parse_date(date)
        is_weekend = day.weekday() >= 5

        # Mock electricity pricing - in a real implementation this would call a pricing API.
        # Small day-to-day variation around the base rate, deterministic per date.
        rng = _seeded_rng("prices", date)
        daily_base = BASE_PRICE_PER_KWH * rng.uniform(0.95, 1.05)

        hourly_rates = []
        for hour in range(24):
            period = _price_period(hour, is_weekend)
            hourly_rates.append({
                "hour": hour,
                "rate": round(daily_base * PERIOD_MULTIPLIERS[period], 4),
                "period": period,
                "demand_charge": PERIOD_DEMAND_CHARGES[period],
            })

        rates = [r["rate"] for r in hourly_rates]
        cheapest = min(rates)
        most_expensive = max(rates)
        return {
            "date": date,
            "weekday": day.strftime("%A"),
            "pricing_type": "time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "summary": {
                "off_peak_hours": "22:00-06:00",
                "peak_hours": "none (weekend)" if is_weekend else "16:00-21:00",
                "cheapest_rate": cheapest,
                "cheapest_hours": [r["hour"] for r in hourly_rates if r["rate"] == cheapest],
                "most_expensive_rate": most_expensive,
                "most_expensive_hours": [r["hour"] for r in hourly_rates if r["rate"] == most_expensive],
                "average_rate": round(sum(rates) / len(rates), 4),
            },
            "hourly_rates": hourly_rates,
        }
    except ValueError:
        return {"error": f"Invalid date '{date}'. Use the YYYY-MM-DD format."}
    except Exception as e:
        return {"error": f"Failed to get electricity prices: {str(e)}"}


@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter ("EV", "HVAC" or "appliance")

    Returns:
        Dict[str, Any]: Energy usage totals, a breakdown by device and by hour of day,
        and the most recent individual records
    """
    try:
        start_dt = _parse_date(start_date)
        end_dt = _parse_date(end_date) + timedelta(days=1)
        if end_dt <= start_dt:
            return {"error": "end_date must be on or after start_date."}

        records = db_manager.get_usage_by_date_range(start_dt, end_dt)

        if device_type:
            records = [r for r in records if (r.device_type or "").lower() == device_type.lower()]

        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "device_breakdown": {},
            "hourly_profile_kwh": {},
            "records": [],
        }
        if not records:
            usage_data["note"] = "No usage records found for this range. The database may not be set up yet."
            return usage_data

        devices = defaultdict(lambda: {"consumption_kwh": 0.0, "cost_usd": 0.0, "records": 0, "names": set()})
        hourly = defaultdict(float)
        days_covered = len({r.timestamp.date() for r in records})
        for record in records:
            device = devices[record.device_type or "unknown"]
            device["consumption_kwh"] += record.consumption_kwh
            device["cost_usd"] += record.cost_usd or 0
            device["records"] += 1
            if record.device_name:
                device["names"].add(record.device_name)
            hourly[record.timestamp.hour] += record.consumption_kwh

        for name, data in devices.items():
            usage_data["device_breakdown"][name] = {
                "consumption_kwh": round(data["consumption_kwh"], 2),
                "cost_usd": round(data["cost_usd"], 2),
                "average_daily_kwh": round(data["consumption_kwh"] / days_covered, 2),
                "share_of_total_pct": round(100 * data["consumption_kwh"] / max(usage_data["total_consumption_kwh"], 1e-9), 1),
                "device_names": sorted(data["names"]),
            }
        usage_data["hourly_profile_kwh"] = {
            f"{hour:02d}:00": round(total / days_covered, 2) for hour, total in sorted(hourly.items())
        }
        usage_data["average_daily_consumption_kwh"] = round(usage_data["total_consumption_kwh"] / days_covered, 2)

        for record in records[-MAX_RECORDS_RETURNED:]:
            usage_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": round(record.consumption_kwh, 3),
                "device_type": record.device_type,
                "device_name": record.device_name,
                "cost_usd": round(record.cost_usd or 0, 3)
            })
        if len(records) > MAX_RECORDS_RETURNED:
            usage_data["records_note"] = f"Showing the {MAX_RECORDS_RETURNED} most recent of {len(records)} records; totals and breakdowns include all records."

        return usage_data
    except ValueError:
        return {"error": "Invalid date. Use the YYYY-MM-DD format for start_date and end_date."}
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}


@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        Dict[str, Any]: Solar generation totals, daily totals with weather, an average
        hour-of-day production profile, and the most recent individual records
    """
    try:
        start_dt = _parse_date(start_date)
        end_dt = _parse_date(end_date) + timedelta(days=1)
        if end_dt <= start_dt:
            return {"error": "end_date must be on or after start_date."}

        records = db_manager.get_generation_by_date_range(start_dt, end_dt)

        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(sum(r.generation_kwh for r in records) / max(1, (end_dt - start_dt).days), 2),
            "daily_totals": [],
            "hourly_profile_kwh": {},
            "generation_by_weather": {},
            "records": [],
        }
        if not records:
            generation_data["note"] = "No solar records found for this range. The database may not be set up yet."
            return generation_data

        daily = defaultdict(lambda: {"kwh": 0.0, "weather": Counter()})
        hourly = defaultdict(list)
        weather = defaultdict(list)
        for record in records:
            day = daily[record.timestamp.date().isoformat()]
            day["kwh"] += record.generation_kwh
            day["weather"][record.weather_condition or "unknown"] += 1
            hourly[record.timestamp.hour].append(record.generation_kwh)
        for date_str, data in sorted(daily.items()):
            condition = data["weather"].most_common(1)[0][0]
            generation_data["daily_totals"].append({
                "date": date_str,
                "generation_kwh": round(data["kwh"], 2),
                "weather_condition": condition,
            })
            weather[condition].append(data["kwh"])
        generation_data["hourly_profile_kwh"] = {
            f"{hour:02d}:00": round(sum(values) / len(values), 2) for hour, values in sorted(hourly.items())
        }
        generation_data["generation_by_weather"] = {
            condition: {"days": len(values), "average_daily_kwh": round(sum(values) / len(values), 2)}
            for condition, values in weather.items()
        }

        for record in records[-MAX_RECORDS_RETURNED:]:
            generation_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": round(record.generation_kwh, 3),
                "weather_condition": record.weather_condition,
                "temperature_c": round(record.temperature_c, 1) if record.temperature_c is not None else None,
                "solar_irradiance": round(record.solar_irradiance, 1) if record.solar_irradiance is not None else None,
            })
        if len(records) > MAX_RECORDS_RETURNED:
            generation_data["records_note"] = f"Showing the {MAX_RECORDS_RETURNED} most recent of {len(records)} records; totals and profiles include all records."

        return generation_data
    except ValueError:
        return {"error": "Invalid date. Use the YYYY-MM-DD format for start_date and end_date."}
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}


@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.

    Args:
        hours (int): Number of hours to look back (default 24)

    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        hours = max(1, int(hours))
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)

        weather_counts = Counter(r.weather_condition or "unknown" for r in generation_records)
        total_consumption = sum(r.consumption_kwh for r in usage_records)
        total_generation = sum(r.generation_kwh for r in generation_records)

        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(total_consumption, 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {}
            },
            "generation": {
                "total_generation_kwh": round(total_generation, 2),
                "average_weather": weather_counts.most_common(1)[0][0] if generation_records else "unknown"
            },
            "net_grid_consumption_kwh": round(total_consumption - total_generation, 2),
            "solar_coverage_pct": round(100 * total_generation / total_consumption, 1) if total_consumption else 0,
        }

        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += record.consumption_kwh
            summary["usage"]["device_breakdown"][device]["cost_usd"] += record.cost_usd or 0
            summary["usage"]["device_breakdown"][device]["records"] += 1

        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)

        if not usage_records and not generation_records:
            summary["note"] = "No records in this period. The database may not be set up yet."

        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}


def load_tip_documents(documents_dir: str = DOCUMENTS_DIR) -> List:
    """Load every .txt document in the knowledge base folder."""
    documents = []
    for doc_path in sorted(glob.glob(os.path.join(documents_dir, "*.txt"))):
        documents.extend(TextLoader(doc_path, encoding="utf-8").load())
    return documents


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url=VOCAREUM_BASE_URL,
        api_key=os.getenv("VOCAREUM_API_KEY")
    )


def build_vectorstore(persist_directory: str = VECTORSTORE_DIR, documents_dir: str = DOCUMENTS_DIR) -> Chroma:
    """Split the knowledge base into chunks and index them in a persistent Chroma store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(load_tip_documents(documents_dir))
    os.makedirs(persist_directory, exist_ok=True)
    return Chroma.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        persist_directory=persist_directory
    )


_vectorstore: Optional[Chroma] = None


def get_vectorstore() -> Chroma:
    """Load the persisted vector store, building it on first use if 02_rag_setup.ipynb hasn't run."""
    global _vectorstore
    if _vectorstore is None:
        if os.path.exists(os.path.join(VECTORSTORE_DIR, "chroma.sqlite3")):
            # Load existing vector store
            _vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=get_embeddings())
            # An empty store (e.g. from an interrupted build) is rebuilt instead of returning no tips
            if _vectorstore._collection.count() == 0:
                _vectorstore = build_vectorstore()
        else:
            # Vector store doesn't exist: initialize it, load and split the documents, and index them
            _vectorstore = build_vectorstore()
    return _vectorstore


@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.
    The knowledge base covers device scheduling, EV charging, HVAC, smart home automation,
    solar and renewable integration, seasonal energy management and battery storage.

    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return

    Returns:
        Dict[str, Any]: Relevant energy tips and best practices, each with its source document
    """
    try:
        if not query or not query.strip():
            return {"error": "A search query is required."}
        max_results = max(1, min(int(max_results), 10))

        # Search for relevant documents
        docs_and_scores = get_vectorstore().similarity_search_with_score(query, k=max_results)

        results = {
            "query": query,
            "total_results": len(docs_and_scores),
            "tips": []
        }

        for i, (doc, distance) in enumerate(docs_and_scores):
            results["tips"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low",
                "distance": round(float(distance), 4),
            })

        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}


@tool
def calculate_energy_savings(device_type: str, current_usage_kwh: float,
                           optimized_usage_kwh: float, price_per_kwh: float = 0.12,
                           optimized_price_per_kwh: float = None,
                           occurrences_per_year: int = 365) -> Dict[str, Any]:
    """
    Calculate potential energy and cost savings from an optimization.
    Supports both using less energy (lower optimized_usage_kwh) and shifting usage to a
    cheaper time (set optimized_price_per_kwh to the rate in the new time slot).

    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh per occurrence (e.g. per day or per run)
        optimized_usage_kwh (float): Optimized energy usage in kWh per occurrence
        price_per_kwh (float): Current price per kWh (default 0.12)
        optimized_price_per_kwh (float): Price per kWh after the change, e.g. the off-peak rate
            when shifting usage. Defaults to price_per_kwh.
        occurrences_per_year (int): How many times per year the usage occurs (365 for daily, 52 for weekly)

    Returns:
        Dict[str, Any]: Savings in kWh, USD and percent, monthly/annual projections and CO2 avoided
    """
    if current_usage_kwh < 0 or optimized_usage_kwh < 0 or price_per_kwh < 0:
        return {"error": "Usage and price values must be non-negative."}
    if optimized_price_per_kwh is None:
        optimized_price_per_kwh = price_per_kwh

    current_cost = current_usage_kwh * price_per_kwh
    optimized_cost = optimized_usage_kwh * optimized_price_per_kwh
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = current_cost - optimized_cost
    savings_percentage = (savings_usd / current_cost) * 100 if current_cost > 0 else 0
    annual_savings = savings_usd * occurrences_per_year

    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "current_cost_usd": round(current_cost, 2),
        "optimized_cost_usd": round(optimized_cost, 2),
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "optimized_price_per_kwh": optimized_price_per_kwh,
        "occurrences_per_year": occurrences_per_year,
        "monthly_savings_usd": round(annual_savings / 12, 2),
        "annual_savings_usd": round(annual_savings, 2),
        "annual_energy_savings_kwh": round(savings_kwh * occurrences_per_year, 1),
        "annual_co2_avoided_kg": round(max(savings_kwh, 0) * occurrences_per_year * GRID_CO2_KG_PER_KWH, 1),
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings
]
