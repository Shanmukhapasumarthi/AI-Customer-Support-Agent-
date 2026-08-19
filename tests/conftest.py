from __future__ import annotations
import hashlib
import os
from pathlib import Path

import pytest

# Set a dummy key BEFORE any app module is imported, so `Settings` can be
# constructed in environments without a real .env (CI, a fresh clone).
os.environ.setdefault("GROQ_API_KEY", "gsk_test_key_not_real")

from langchain_core.embeddings import Embeddings  # noqa: E402


class StubEmbeddings(Embeddings):
    """Deterministic fake embeddings for fast, offline tests.

    Implements LangChain's `Embeddings` interface, which is the only reason it
    can be dropped into Chroma in place of the real model. This is the value of
    programming against interfaces rather than concrete classes.
    """

    def __init__(self, dimension: int = 32):
        self.dimension = dimension

    def _vector(self, text: str) -> list[float]:
        # Hash the text, then spread the digest bytes across the dimensions.
        # Same text always gives the same vector, so tests are reproducible.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [
            (digest[i % len(digest)] / 255.0) - 0.5
            for i in range(self.dimension)
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture(scope="session")
def stub_embeddings() -> StubEmbeddings:
    return StubEmbeddings()


@pytest.fixture(scope="session")
def knowledge_base_dir() -> Path:
    """Path to the real knowledge base documents."""
    from app.config import settings

    return settings.knowledge_base_path


@pytest.fixture(scope="session")
def loaded_documents(knowledge_base_dir: Path):
    """Load the real knowledge base once for the whole session.

    Session scope because reading nine files repeatedly is wasted work; the
    documents are read-only as far as these tests are concerned.
    """
    from app.rag.loader import load_knowledge_base

    return load_knowledge_base(knowledge_base_dir)


@pytest.fixture(scope="session")
def chunks(loaded_documents):
    from app.rag.splitter import split_documents

    return split_documents(loaded_documents)


@pytest.fixture(scope="session")
def seeded_database(tmp_path_factory):
    """Seed a throwaway SQLite database and point the app at it.

    We monkeypatch the module-level `engine` and `SessionLocal` so tests never
    touch your development database. Session-scoped so seeding happens once.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.database.database as db_module
    from app.database.models import Base

    db_path = tmp_path_factory.mktemp("db") / "test_support.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True,
                           connect_args={"check_same_thread": False})

    db_module.engine = engine
    db_module.SessionLocal = sessionmaker(bind=engine, expire_on_commit=False,
                                          future=True)

    Base.metadata.create_all(engine)

    import scripts.seed_database as seeder
    seeder.SessionLocal = db_module.SessionLocal
    seeder.engine = engine
    seeder.seed(reset=False)

    yield db_path


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: test requires a real Groq API key and network access"
    )


def pytest_collection_modifyitems(config, items):
    """Skip live tests unless explicitly selected with -m live."""
    if config.getoption("-m") == "live":
        return
    skip_live = pytest.mark.skip(reason="needs a real API key; run with -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
