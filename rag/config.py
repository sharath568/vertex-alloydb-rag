"""Settings, all read from environment variables so the same image runs locally and on Cloud Run."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Postgres-compatible DSN. Locally: the docker-compose pgvector container.
    # On Cloud Run: AlloyDB private IP (Direct VPC egress), password from Secret Manager.
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"
        )
    )

    # Models. Check what your project can use with `python -m rag.models`.
    generation_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    )
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768")))

    # "gemini" calls Vertex AI / Gemini API. "fake" is a deterministic local embedder for tests.
    embedder: str = field(default_factory=lambda: os.getenv("EMBEDDER", "gemini"))

    # Retrieval knobs.
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    candidates: int = field(default_factory=lambda: int(os.getenv("CANDIDATES", "40")))
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    # If the best chunk is less similar than this, we refuse instead of calling the LLM.
    min_similarity: float = field(
        default_factory=lambda: float(os.getenv("MIN_SIMILARITY", "0.55"))
    )

    # Chunking.
    chunk_tokens: int = field(default_factory=lambda: int(os.getenv("CHUNK_TOKENS", "350")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))

    # Set on AlloyDB to use the ScaNN index instead of pgvector HNSW (see sql/002_alloydb_scann.sql).
    alloydb: bool = field(default_factory=lambda: _bool("ALLOYDB", False))


def get_settings() -> Settings:
    return Settings()
