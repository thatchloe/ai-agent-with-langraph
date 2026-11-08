import pytest
from dotenv import load_dotenv

from agentic import config
from tests.fake_llm import ScriptedLLM, support_brain
from tests.seed import seed_cultpass, seed_udahub

load_dotenv(config.SOLUTION_ROOT / ".env")


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    monkeypatch.setenv("UDAHUB_LOG_PATH", str(tmp_path / "agent_events.jsonl"))


@pytest.fixture
def dbs(tmp_path, monkeypatch):
    """Fresh, deterministic cultpass.db + udahub.db (and log file) for every test."""
    cultpass_db = tmp_path / "cultpass.db"
    udahub_db = tmp_path / "udahub.db"
    seed_cultpass(cultpass_db)
    seed_udahub(udahub_db)
    monkeypatch.setenv("CULTPASS_DB_PATH", str(cultpass_db))
    monkeypatch.setenv("UDAHUB_DB_PATH", str(udahub_db))
    monkeypatch.setenv("UDAHUB_LOG_PATH", str(tmp_path / "agent_events.jsonl"))
    monkeypatch.setenv("UDAHUB_EMBEDDINGS", "off")  # offline tests: lexical retrieval only
    return {"cultpass": cultpass_db, "udahub": udahub_db, "logs": tmp_path / "agent_events.jsonl"}


@pytest.fixture
def scripted_llm():
    return ScriptedLLM(handler=support_brain, calls=[])
