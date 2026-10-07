"""Retrieval evaluation: hit rate@k, MRR, and a tenant-isolation check.

    python -m eval.run_eval            # uses EMBEDDER from env (gemini or fake)
    python -m eval.run_eval --k 3

Run it on every change to chunking, embeddings, index settings or the SQL. Retrieval quality is
the ceiling on answer quality, and it's cheap to measure: no LLM calls needed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rag.config import get_settings
from rag.embed import get_embedder
from rag.retrieve import retrieve

GOLDEN = Path(__file__).with_name("golden.jsonl")


def run(k: int = 5) -> dict:
    settings = get_settings()
    embedder = get_embedder(settings)
    cases = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]

    hits_at_k = 0
    rr_total = 0.0
    leaks = 0
    latencies = []
    misses = []
    for c in cases:
        t0 = time.perf_counter()
        hits = retrieve(c["q"], tenant=c["tenant"], settings=settings, embedder=embedder, top_k=k)
        latencies.append((time.perf_counter() - t0) * 1000)
        leaks += sum(1 for h in hits if not h.doc_id.startswith(c["tenant"] + "/"))
        rank = next(
            (i for i, h in enumerate(hits, 1) if h.doc_id == c["doc"] and h.heading.endswith(c["section"])),
            None,
        )
        if rank:
            hits_at_k += 1
            rr_total += 1 / rank
        else:
            misses.append(c["q"])

    latencies.sort()
    report = {
        "cases": len(cases),
        f"hit_rate@{k}": round(hits_at_k / len(cases), 3),
        "mrr": round(rr_total / len(cases), 3),
        "tenant_leaks": leaks,
        "p50_ms": round(latencies[len(latencies) // 2], 1),
        "p95_ms": round(latencies[int(len(latencies) * 0.95) - 1], 1),
        "misses": misses,
    }
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    print(json.dumps(run(ap.parse_args().k), indent=2))
