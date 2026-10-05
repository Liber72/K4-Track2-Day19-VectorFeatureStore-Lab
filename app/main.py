"""FastAPI service exposing /search?q=...&mode=keyword|semantic|hybrid.

Run: `make api`  (or `uvicorn app.main:app --reload --port 8000`)
Try: curl 'http://localhost:8000/search?q=cloud+computing&mode=hybrid'

The Searcher is built once at startup (lazy). On first request the embedding
model is loaded — subsequent requests reuse it. P99 should be < 50 ms after
warm-up; that's the rubric threshold.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.search import COLLECTION, Searcher

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "corpus_vn.jsonl"

# Module-level state intentionally — Searcher is heavy to build.
_searcher: Searcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Reuse NB1 vectors in Docker; the query embedding model loads on demand."""
    global _searcher
    if not CORPUS_PATH.exists():
        raise RuntimeError(
            f"{CORPUS_PATH} missing. Run `make seed` first."
        )
    _searcher = Searcher.from_corpus(
        CORPUS_PATH, require_existing=os.getenv("QDRANT_MODE", "memory") == "server",
    )
    try:
        yield
    finally:
        if _searcher.client is not None:
            _searcher.client.close()
        _searcher = None


app = FastAPI(
    title="Day 19 — Search API",
    description="Vector + BM25 + hybrid (RRF) retrieval. Lab 19, Track 2.",
    version="0.1.0",
    lifespan=lifespan,
)


class SearchHitOut(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: Literal["keyword", "semantic", "hybrid"]
    top_k: int
    latency_ms: float = Field(description="Server-side latency excluding network.")
    hits: list[SearchHitOut]


@app.get("/")
def root() -> dict:
    return {
        "name": "Day 19 Search API",
        "endpoints": ["/search", "/healthz", "/docs"],
    }


@app.get("/healthz")
def healthz() -> dict:
    return {
        "app": "lab19-search",
        "ready": _searcher is not None,
        "n_docs": _searcher.size if _searcher else 0,
        "collection": COLLECTION,
        "embedding_backend": _searcher.embedder.backend if _searcher else None,
        "reused_index": _searcher.reused_index if _searcher else False,
    }


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Query string (VN/EN OK)"),
    mode: Literal["keyword", "semantic", "hybrid"] = Query("hybrid"),
    top_k: int = Query(10, ge=1, le=100),
    rrf_k: int = Query(60, ge=1, le=200, description="Reciprocal Rank Fusion constant (hybrid only)"),
) -> SearchResponse:
    if _searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not yet ready")

    t0 = time.perf_counter()
    hits = _searcher.search(q, mode=mode, top_k=top_k, rrf_k=rrf_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    return SearchResponse(
        query=q,
        mode=mode,
        top_k=top_k,
        latency_ms=latency_ms,
        hits=[SearchHitOut(**h.dict()) for h in hits],
    )
