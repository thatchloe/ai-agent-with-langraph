"""Central configuration for the UDA-Hub agentic system.

All filesystem paths are resolved *absolutely* from the location of this file,
so the tools work no matter which directory the notebook / script / test runner
is started from. Every path can be overridden with an environment variable
(used by the test-suite to point at throw-away databases).
"""
from __future__ import annotations

import os
from pathlib import Path

# <solution>/agentic/config.py -> <solution>
SOLUTION_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = SOLUTION_ROOT / "data"

DEFAULT_ACCOUNT_ID = "cultpass"
DEFAULT_MODEL = "gpt-4o-mini"
VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"


def udahub_db_path() -> Path:
    """Core UDA-Hub database (tickets, knowledge base, long-term memory)."""
    return Path(os.getenv("UDAHUB_DB_PATH", DATA_DIR / "core" / "udahub.db")).resolve()


def cultpass_db_path() -> Path:
    """External CultPass database (customers, subscriptions, experiences)."""
    return Path(os.getenv("CULTPASS_DB_PATH", DATA_DIR / "external" / "cultpass.db")).resolve()


def log_path() -> Path:
    """JSON-lines file where every agent decision / tool call is recorded."""
    return Path(os.getenv("UDAHUB_LOG_PATH", SOLUTION_ROOT / "logs" / "agent_events.jsonl")).resolve()


def mcp_transport() -> str:
    """How agents reach the MCP tool servers: ``memory`` (default) or ``stdio``."""
    return os.getenv("UDAHUB_MCP_TRANSPORT", "memory").lower()


def has_llm_credentials() -> bool:
    return bool(os.getenv("VOCAREUM_API_KEY") or os.getenv("OPENAI_API_KEY"))


def get_llm(model: str | None = None, temperature: float = 0.0):
    """Build the chat model used by every agent.

    Uses the Udacity/Vocareum OpenAI proxy when ``VOCAREUM_API_KEY`` is set,
    otherwise the regular OpenAI API with ``OPENAI_API_KEY``.
    """
    from langchain_openai import ChatOpenAI

    model = model or os.getenv("UDAHUB_MODEL", DEFAULT_MODEL)
    if os.getenv("VOCAREUM_API_KEY"):
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=os.getenv("OPENAI_BASE_URL", VOCAREUM_BASE_URL),
            api_key=os.getenv("VOCAREUM_API_KEY"),
        )
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model=model, temperature=temperature)
    raise RuntimeError(
        "No LLM credentials found. Set VOCAREUM_API_KEY or OPENAI_API_KEY "
        "(see .env.example)."
    )


def embedding_model() -> str:
    return os.getenv("UDAHUB_EMBEDDING_MODEL", "text-embedding-3-small")


def get_embeddings():
    """Embeddings for hybrid RAG.

    ``UDAHUB_EMBEDDINGS``: ``auto`` (default: on when LLM credentials exist), ``openai`` (on), ``off``.
    """
    mode = os.getenv("UDAHUB_EMBEDDINGS", "auto").lower()
    if mode in ("off", "false", "0", "none") or not has_llm_credentials():
        return None
    from langchain_openai import OpenAIEmbeddings

    kwargs: dict = {"model": embedding_model()}
    if os.getenv("VOCAREUM_API_KEY"):
        kwargs.update(
            base_url=os.getenv("OPENAI_BASE_URL", VOCAREUM_BASE_URL),
            api_key=os.getenv("VOCAREUM_API_KEY"),
        )
    return OpenAIEmbeddings(**kwargs)
