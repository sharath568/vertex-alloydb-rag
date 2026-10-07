"""FastAPI service for Cloud Run.

Tenant comes from a header here to keep the demo small. In production, take it from a verified
identity (IAP / Identity Platform JWT claim), never from something the client can set freely.
"""

from __future__ import annotations

import json
import logging
import time
from functools import lru_cache

from fastapi import FastAPI, Header
from pydantic import BaseModel

from .answer import answer_question, gemini_generator
from .config import get_settings
from .db import pool
from .embed import get_embedder

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rag")

app = FastAPI(title="vertex-alloydb-rag")


@lru_cache
def _deps():
    s = get_settings()
    return s, get_embedder(s), gemini_generator(s)


class AskRequest(BaseModel):
    question: str


@app.get("/healthz")
def healthz() -> dict:
    s = get_settings()
    with pool(s).connection() as conn:
        conn.execute("SELECT 1")
    return {"ok": True}


@app.post("/ask")
def ask(req: AskRequest, x_tenant_id: str = Header(default="default")) -> dict:
    settings, embedder, generate = _deps()
    t0 = time.perf_counter()
    result = answer_question(
        req.question, tenant=x_tenant_id, settings=settings, embedder=embedder, generate=generate
    )
    # One structured log line per request. Cloud Logging parses JSON payloads automatically.
    log.info(json.dumps(
        {
            "event": "ask",
            "tenant": x_tenant_id,
            "grounded": result.grounded,
            "refused_reason": result.refused_reason,
            "n_sources": len(result.sources),
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        }
    ))
    return result.to_dict()
