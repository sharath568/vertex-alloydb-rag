import os

import pytest

# Tests run fully offline: fake embedder + scripted generator + local Postgres with pgvector.
os.environ.setdefault("EMBEDDER", "fake")
os.environ.setdefault("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag_test")


@pytest.fixture(scope="session")
def settings():
    from rag.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def embedder(settings):
    from rag.embed import get_embedder

    return get_embedder(settings)


@pytest.fixture(scope="session")
def indexed(settings, embedder):
    import psycopg
    from rag.ingest import ingest_dir

    with psycopg.connect(settings.database_url) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, documents CASCADE")
    return ingest_dir("data/docs", settings, embedder)
