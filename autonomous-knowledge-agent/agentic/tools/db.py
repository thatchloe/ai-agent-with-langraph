"""Database access helpers shared by every tool.

Engines are cached per absolute path. ``check_same_thread=False`` is required
because MCP servers execute synchronous tools in a worker thread.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from agentic import config


@lru_cache(maxsize=16)
def _engine_for(path: str) -> Engine:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run 01_external_db_setup.ipynb and "
            "02_core_db_setup.ipynb first."
        )
    return create_engine(
        f"sqlite:///{path}", echo=False, connect_args={"check_same_thread": False}
    )


def udahub_engine() -> Engine:
    return _engine_for(str(config.udahub_db_path()))


def cultpass_engine() -> Engine:
    return _engine_for(str(config.cultpass_db_path()))


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def to_dict(instance) -> dict:
    """Serialize a SQLAlchemy row into JSON-friendly primitives."""
    out = {}
    for column in instance.__table__.columns:
        value = getattr(instance, column.name)
        if hasattr(value, "isoformat"):
            value = value.isoformat(sep=" ", timespec="minutes")
        elif hasattr(value, "value"):  # enums
            value = value.value
        out[column.name] = value
    return out
