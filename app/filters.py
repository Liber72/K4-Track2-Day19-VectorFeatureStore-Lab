"""Filtered vector search: the three strategies, measured side by side (NB5).

Deck reference: "Filtered Search: Cai Bay Recall It Ai Noi Den".

    post-filter   ANN first, drop non-matching after  -> recall CLIFF
    pre-filter    exact scan over the matching subset -> always correct, loses the index
    filtered-ANN  the index itself is filter-aware    -> what you actually want

`FilteredIndex` deliberately REUSES the vectors already computed by
`app.search.Searcher` (pulled back out of Qdrant with `with_vectors=True`)
instead of re-embedding 1000 documents. Embedding is the slow part of this lab;
paying for it twice teaches nothing.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from qdrant_client import QdrantClient, models

from app.metadata import enrich
from app.search import COLLECTION as BASE_COLLECTION
from app.search import Searcher

FILTERED_COLLECTION = "lab19_filtered"


@dataclass
class Result:
    """One strategy's answer for one query."""
    strategy: str
    doc_ids: list[str]
    latency_ms: float
    # Candidates returned to Python (or scanned by our exact baseline).
    # This does NOT measure how many vectors Qdrant visited internally.
    fetched: int = 0

    def recall_against(self, truth: list[str]) -> float:
        if not truth:
            return 1.0
        return len(set(self.doc_ids) & set(truth)) / len(truth)


@dataclass
class FilteredIndex:
    client: QdrantClient
    docs: list[dict]
    vectors: np.ndarray                     # (N, dim) float32, row i <-> docs[i]
    embedder: object
    collection: str = FILTERED_COLLECTION
    _id_of: dict[str, int] = field(default_factory=dict)

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def from_searcher(cls, searcher: Searcher,
                      collection: str = FILTERED_COLLECTION) -> "FilteredIndex":
        """Clone the base collection into a filter-aware one with rich payloads."""
        client = searcher.client
        assert client is not None, "searcher was not built"

        if collection == BASE_COLLECTION:
            raise ValueError("The filtered demo must use a separate collection from NB1")
        points, offset = [], None
        while True:
            batch, offset = client.scroll(
                collection_name=BASE_COLLECTION, limit=128, offset=offset,
                with_vectors=True, with_payload=True,
            )
            points.extend(batch)
            if offset is None:
                break
        if not points or len(points) != len(searcher.docs):
            raise ValueError("The NB1 collection is empty or does not match the corpus")
        points = sorted(points, key=lambda p: p.id)

        # The base collection's payload only carries doc_id/title/text, so pull
        # the FULL corpus record (which has `topic`) from the searcher and key
        # it by doc_id. Without this, a topic filter matches zero documents and
        # every topic-filtered query silently returns nothing.
        by_id = {d["doc_id"]: d for d in searcher.docs}
        docs = [enrich({**by_id.get(p.payload["doc_id"], {}), **p.payload}) for p in points]
        vectors = np.asarray([p.vector for p in points], dtype=np.float32)

        if collection in {c.name for c in client.get_collections().collections}:
            client.delete_collection(collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE
            ),
        )
        # Payload indexes are what let a real Qdrant deployment filter *inside*
        # the HNSW walk instead of after it.
        for fname, ftype in (
            ("tenant", models.PayloadSchemaType.KEYWORD),
            ("access", models.PayloadSchemaType.KEYWORD),
            ("topic", models.PayloadSchemaType.KEYWORD),
            ("published_ts", models.PayloadSchemaType.INTEGER),
        ):
            # Local mode ignores indexes; server failures must remain visible.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Payload indexes have no effect.*")
                client.create_payload_index(collection, fname, field_schema=ftype, wait=True)

        # Small batches also work with 1,024-dimensional bge-m3 vectors over HTTP.
        for start in range(0, len(docs), 64):
            client.upsert(
                collection_name=collection,
                points=[models.PointStruct(id=i, vector=vectors[i].tolist(), payload=docs[i])
                        for i in range(start, min(start + 64, len(docs)))],
                wait=True,
            )

        idx = cls(client=client, docs=docs, vectors=vectors,
                  embedder=searcher.embedder, collection=collection)
        idx._id_of = {d["doc_id"]: i for i, d in enumerate(docs)}
        return idx

    def embed(self, query: str) -> np.ndarray:
        return np.asarray(next(self.embedder.embed([query])), dtype=np.float32)

    # ── ground truth ────────────────────────────────────────────────────
    def exact_top_k(self, qv: np.ndarray, predicate: Callable[[dict], bool], k: int) -> list[str]:
        """Brute-force cosine over the matching subset -- the correct answer."""
        keep = [i for i, d in enumerate(self.docs) if predicate(d)]
        if not keep:
            return []
        sub = self.vectors[keep]
        sims = (sub @ qv) / (np.linalg.norm(sub, axis=1) * np.linalg.norm(qv) + 1e-12)
        order = np.argsort(-sims)[:k]
        return [self.docs[keep[i]]["doc_id"] for i in order]

    # ── the three strategies ────────────────────────────────────────────
    def post_filter(self, query: str, predicate, k: int = 10, fetch_k: int | None = None) -> Result:
        """Ask the index for fetch_k, then throw away whatever does not match."""
        fetch_k = fetch_k or k
        qv = self.embed(query)
        t0 = time.perf_counter()
        hits = self.client.query_points(
            collection_name=self.collection, query=qv.tolist(), limit=fetch_k
        ).points
        kept = [h.payload["doc_id"] for h in hits if predicate(h.payload)][:k]
        return Result("post-filter", kept, (time.perf_counter() - t0) * 1000, fetched=len(hits))

    def pre_filter(self, query: str, predicate, k: int = 10) -> Result:
        """Filter first, then scan the survivors exactly. Correct, but no index."""
        qv = self.embed(query)
        t0 = time.perf_counter()
        ids = self.exact_top_k(qv, predicate, k)
        n = sum(1 for d in self.docs if predicate(d))
        return Result("pre-filter", ids, (time.perf_counter() - t0) * 1000, fetched=n)

    def filtered_ann(self, query: str, qfilter: models.Filter | None, k: int = 10) -> Result:
        """Hand the filter to the engine and let it stay inside the index."""
        qv = self.embed(query)
        t0 = time.perf_counter()
        hits = self.client.query_points(
            collection_name=self.collection,
            query=qv.tolist(),
            query_filter=qfilter,
            limit=k,
        ).points
        ids = [h.payload["doc_id"] for h in hits]
        return Result("filtered-ANN", ids, (time.perf_counter() - t0) * 1000, fetched=len(ids))


# ── ready-made filters of controlled selectivity (used by NB5) ──────────
def access_filter(level: str = "internal") -> tuple[Callable[[dict], bool], models.Filter]:
    pred = lambda d: d.get("access") == level          # noqa: E731
    qf = models.Filter(must=[models.FieldCondition(
        key="access", match=models.MatchValue(value=level))])
    return pred, qf


def tenant_filter(tenant: str) -> tuple[Callable[[dict], bool], models.Filter]:
    pred = lambda d: d.get("tenant") == tenant          # noqa: E731
    qf = models.Filter(must=[models.FieldCondition(
        key="tenant", match=models.MatchValue(value=tenant))])
    return pred, qf


def recent_filter(since_ts: int) -> tuple[Callable[[dict], bool], models.Filter]:
    pred = lambda d: d.get("published_ts", 0) >= since_ts   # noqa: E731
    qf = models.Filter(must=[models.FieldCondition(
        key="published_ts", range=models.Range(gte=since_ts))])
    return pred, qf


def combo_filter(tenant: str, since_ts: int) -> tuple[Callable[[dict], bool], models.Filter]:
    """Deliberately narrow: ~1/3 x recency. This is where post-filter dies."""
    pred = lambda d: d.get("tenant") == tenant and d.get("published_ts", 0) >= since_ts  # noqa: E731
    qf = models.Filter(must=[
        models.FieldCondition(key="tenant", match=models.MatchValue(value=tenant)),
        models.FieldCondition(key="published_ts", range=models.Range(gte=since_ts)),
    ])
    return pred, qf
