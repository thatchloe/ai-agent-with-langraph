"""Data requirements: >= 14 diverse knowledge articles; setup notebooks produce the DBs."""
import json

import pytest
from sqlalchemy import create_engine, text

from agentic import config

ARTICLES = config.DATA_DIR / "external" / "cultpass_articles.jsonl"


def _articles():
    with open(ARTICLES, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_at_least_14_articles_with_required_fields():
    articles = _articles()
    assert len(articles) >= 14
    for art in articles:
        assert art["title"].strip() and art["content"].strip() and art["tags"].strip()


def test_articles_are_unique_and_cover_diverse_topics():
    articles = _articles()
    assert len({a["title"] for a in articles}) == len(articles)
    first_tags = {a["tags"].split(",")[0].strip() for a in articles}
    assert len(first_tags) >= 12, first_tags
    all_tags = {t.strip() for a in articles for t in a["tags"].split(",")}
    for topic in ["reservation", "subscription", "login", "refund", "blocked", "premium", "privacy", "accessibility"]:
        assert topic in all_tags, topic


@pytest.mark.skipif(not config.udahub_db_path().exists(), reason="run 02_core_db_setup.ipynb first")
def test_core_db_built_by_notebook_has_all_articles_and_memory_table():
    engine = create_engine(f"sqlite:///{config.udahub_db_path()}")
    with engine.connect() as conn:
        assert conn.execute(text("select count(*) from knowledge")).scalar() >= 14
        assert conn.execute(text("select count(*) from customer_memories")).scalar() >= 1


@pytest.mark.skipif(not config.cultpass_db_path().exists(), reason="run 01_external_db_setup.ipynb first")
def test_external_db_built_by_notebook():
    engine = create_engine(f"sqlite:///{config.cultpass_db_path()}")
    with engine.connect() as conn:
        assert conn.execute(text("select count(*) from users")).scalar() == 6
        assert conn.execute(text("select count(distinct user_id) from reservations")).scalar() >= 3
