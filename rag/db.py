"""Connection pool. psycopg 3 + pgvector adapter. Same code for local Postgres and AlloyDB."""

from __future__ import annotations

import atexit
from functools import lru_cache
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg_pool import ConnectionPool

from .config import Settings

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def _configure(conn: Connection) -> None:
    register_vector(conn)
    conn.commit()  # the pool requires connections to come back idle, not mid-transaction
    # pgvector >= 0.8: keep scanning the HNSW graph when a WHERE filter (tenant_id) removes
    # most neighbors. Without it, filtered queries can return fewer than LIMIT rows.
    # Harmless no-op error on servers that don't have the setting, so we ignore failures.
    try:
        with conn.cursor() as cur:
            cur.execute("SET hnsw.iterative_scan = relaxed_order")
        conn.commit()
    except Exception:  # noqa: BLE001
        conn.rollback()


@lru_cache(maxsize=4)
def get_pool(dsn: str) -> ConnectionPool:
    p = ConnectionPool(dsn, min_size=1, max_size=10, configure=_configure, open=True)
    atexit.register(p.close)
    return p


def pool(settings: Settings) -> ConnectionPool:
    return get_pool(settings.database_url)


def apply_schema(settings: Settings) -> None:
    """Base schema. Plain connection, not the pool: the pool's configure step needs the vector type."""
    with psycopg.connect(settings.database_url) as conn:
        conn.execute((SQL_DIR / "001_schema.sql").read_text())


def build_scann_index(settings: Settings) -> str:
    """AlloyDB only, after loading data. Falls back to HNSW if ScaNN can't be built
    (for example, a demo table that is too small for ScaNN to partition)."""
    try:
        with psycopg.connect(settings.database_url) as conn:
            conn.execute((SQL_DIR / "002_alloydb_scann.sql").read_text())
        return "scann"
    except psycopg.Error as e:
        print(f"ScaNN index not built, keeping pgvector HNSW: {e}")
        return "hnsw"
