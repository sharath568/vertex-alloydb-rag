"""Embedders.

GeminiEmbedder calls gemini-embedding-001 through the google-genai SDK. The same code works
against Vertex AI (GOOGLE_GENAI_USE_VERTEXAI=true + project/location) or the Gemini API (API key).

FakeEmbedder is a deterministic hashing embedder. It is not semantic, but it is lexical enough
that tests and the local demo behave sensibly without any cloud credentials.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

from .config import Settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def l2_normalize(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class GeminiEmbedder:
    """gemini-embedding-001 with task types and a reduced output size.

    Two details that bite people in production:
    1. Use RETRIEVAL_DOCUMENT for chunks and RETRIEVAL_QUERY for questions. They are trained as a pair.
    2. When you ask for fewer than 3072 dimensions, the vectors are no longer unit length.
       Normalize them yourself, or cosine and dot-product rankings drift apart.
    """

    BATCH = 50  # stay well under per-request limits; tune for your quota

    def __init__(self, settings: Settings):
        from google import genai  # imported lazily so tests don't need the SDK configured

        self.client = genai.Client()
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = list(texts[i : i + self.BATCH])
            resp = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self.dim
                ),
            )
            out.extend(l2_normalize(e.values) for e in resp.embeddings)
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


_TOKEN = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Feature-hashing bag of words (plus bigrams). Deterministic, offline, good enough for tests."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        toks = _TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return l2_normalize(v)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "fake":
        return FakeEmbedder(settings.embedding_dim)
    return GeminiEmbedder(settings)
