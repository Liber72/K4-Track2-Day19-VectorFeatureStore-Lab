"""Searcher — keyword (BM25) + semantic (vector) + hybrid (RRF) on the lab corpus.

Designed to work in both lite (Qdrant in-memory) and docker (Qdrant server) modes;
switch via env var QDRANT_MODE=memory|server (defaults to memory).

The hybrid mode uses weighted RRF with k=60: keyword=0.05, semantic=0.95.
NB2 reports keyword, semantic and weighted hybrid retrieval.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder
from app.fusion import fuse_rankings
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

Mode = Literal["keyword", "semantic", "hybrid"]
# Model + dimension now come from EMBEDDING_BACKEND (see app/embeddings.py).
# Defaults are unchanged: fastembed / BAAI/bge-small-en-v1.5 / 384-dim.
EMBED_MODEL = Embedder().model_name
EMBED_DIM = Embedder().dim
COLLECTION = "lab19"


@dataclass
class SearchHit:
    doc_id: str
    title: str
    text: str
    score: float

    def dict(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "text": self.text, "score": self.score}


class Searcher:
    """Holds the BM25 index, Qdrant client, and document metadata.

    Server mode reuses the collection built in NB1 after checking its corpus
    and vector configuration. Memory mode builds its own temporary index.
    """

    def __init__(self) -> None:
        self.docs: list[dict] = []
        self.doc_ids: list[str] = []
        self.bm25: BM25Okapi | None = None
        self.client: QdrantClient | None = None
        self.embedder: Embedder | None = None
        self.reused_index = False
        self.bm25_ignored_tokens: set[str] = set()

    @property
    def size(self) -> int:
        return len(self.docs)

    @classmethod
    def from_corpus(cls, corpus_path: Path, *, require_existing: bool = False) -> "Searcher":
        # A student who opens NB1 before running setup otherwise gets a bare
        # FileNotFoundError pointing at a relative path, with no hint that the
        # corpus is generated rather than committed.
        if not Path(corpus_path).exists():
            raise FileNotFoundError(
                f"Corpus not found at {corpus_path}.\n"
                "The corpus is generated, not committed. Run:\n"
                "    bash setup-lite.sh      # first time (venv + deps + data)\n"
                "    make seed               # if you only need to regenerate data"
            )
        s = cls()
        s._load_docs(corpus_path)
        s._build_bm25()
        s._build_vector_index(require_existing=require_existing)
        return s

    # ── ingestion ───────────────────────────────────────────────────────
    def _load_docs(self, corpus_path: Path) -> None:
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.docs.append(d)
                self.doc_ids.append(d["doc_id"])

    def _build_bm25(self) -> None:
        # Unicode tokens handle punctuation; Vietnamese compounds are still syllables.
        # Corpus-wide boilerplate is removed before building the BM25 index.
        tokenized = [self._tokenize(d["title"] + " " + d["text"]) for d in self.docs]
        # Very common corpus words do not discriminate between documents.
        # Derive this filter from document frequency, without relevance labels.
        document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
        self.bm25_ignored_tokens = {
            token for token, count in document_frequency.items()
            if count / len(tokenized) >= 0.8
        }
        filtered = [[token for token in tokens if token not in self.bm25_ignored_tokens]
                    for tokens in tokenized]
        self.bm25 = BM25Okapi(filtered)

    def _build_vector_index(self, *, require_existing: bool = False) -> None:
        self.embedder = Embedder()

        mode = os.getenv("QDRANT_MODE", "memory")
        if mode == "server":
            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            self.client = QdrantClient(url=url, timeout=60)
        elif mode == "memory":
            if require_existing:
                raise ValueError("An existing NB1 collection requires QDRANT_MODE=server")
            self.client = QdrantClient(":memory:")
        else:
            raise ValueError(f"Unknown QDRANT_MODE={mode!r}; use memory or server")

        if mode == "server" and self.client.collection_exists(COLLECTION):
            self._validate_existing_collection()
            self.reused_index = True
            return
        if require_existing:
            raise ValueError("Collection lab19 is missing. Run NB1 before NB2/NB3.")

        self.client.create_collection(
            collection_name=COLLECTION,
            # dimension must follow the chosen model, not a module constant --
            # switching EMBEDDING_BACKEND changes it (384 -> 1024 -> 1536).
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )

        BATCH = 32
        for start in range(0, len(self.docs), BATCH):
            batch = self.docs[start:start + BATCH]
            texts = [d["title"] + " " + d["text"] for d in batch]
            vectors = list(self.embedder.embed(texts))
            if len(vectors) != len(batch):
                raise ValueError("Expected one embedding per document")
            points: list[PointStruct] = []
            for i, (d, v) in enumerate(zip(batch, vectors)):
                points.append(PointStruct(
                    id=start + i,
                    vector=v.tolist(),
                    payload=d,
                ))
            self.client.upsert(collection_name=COLLECTION, points=points, wait=True)

    def _validate_existing_collection(self) -> None:
        """Check that persisted vectors belong to the corpus being searched."""
        assert self.client is not None and self.embedder is not None
        config = self.client.get_collection(COLLECTION).config.params.vectors
        if (not isinstance(config, VectorParams) or config.size != self.embedder.dim
                or config.distance != Distance.COSINE):
            raise ValueError("Collection lab19 has a different vector configuration. Check NB1/model.")
        expected = {d["doc_id"]: d for d in self.docs}
        if len(expected) != len(self.docs):
            raise ValueError("Corpus contains duplicate doc_id values")
        if self.client.count(COLLECTION, exact=True).count != len(expected):
            raise ValueError("Collection lab19 and corpus have different document counts. Check NB1.")
        seen: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                COLLECTION, limit=256, offset=offset, with_vectors=False,
                with_payload=["doc_id", "title", "text"],
            )
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                doc = expected.get(doc_id)
                if (doc is None or doc_id in seen
                        or any(payload.get(key) != doc[key] for key in ("title", "text"))):
                    raise ValueError("Collection lab19 does not match the corpus/payload. Check NB1.")
                seen.add(doc_id)
            if offset is None:
                break
        if seen != set(expected):
            raise ValueError("Collection lab19 is missing corpus documents")

    # ── retrieval ───────────────────────────────────────────────────────
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+(?:[-/]\w+)*", text.casefold())

    def search(
        self,
        query: str,
        mode: Mode = "hybrid",
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        if mode == "keyword":
            return self._search_keyword(query, top_k)
        if mode == "semantic":
            return self._search_semantic(query, top_k)
        if mode == "hybrid":
            return self._search_hybrid(query, top_k, rrf_k)
        raise ValueError(f"unknown mode {mode!r}")

    def _search_keyword(self, query: str, top_k: int) -> list[SearchHit]:
        assert self.bm25 is not None
        scores = self.bm25.get_scores([
            token for token in self._tokenize(query)
            if token not in self.bm25_ignored_tokens
        ])
        # A zero BM25 score provides no lexical evidence. Do not give such
        # documents an arbitrary rank (and therefore a positive RRF vote).
        candidates = [i for i, score in enumerate(scores) if score > 0]
        ranked = sorted(candidates, key=lambda i: -scores[i])[:top_k]
        return [
            SearchHit(
                doc_id=self.docs[i]["doc_id"],
                title=self.docs[i]["title"],
                text=self.docs[i]["text"],
                score=float(scores[i]),
            )
            for i in ranked
        ]

    def _search_semantic(self, query: str, top_k: int) -> list[SearchHit]:
        assert self.client is not None and self.embedder is not None
        q_vec = next(self.embedder.embed([query])).tolist()
        result = self.client.query_points(
            collection_name=COLLECTION,
            query=q_vec,
            limit=top_k,
        )
        return [
            SearchHit(
                doc_id=p.payload["doc_id"],
                title=p.payload["title"],
                text=p.payload["text"],
                score=float(p.score),
            )
            for p in result.points
        ]

    def _search_hybrid(self, query: str, top_k: int, rrf_k: int) -> list[SearchHit]:
        # Pull a deeper top-K from each retriever so RRF has signal beyond top-10.
        depth = max(top_k * 5, 50)
        kw_hits = self._search_keyword(query, depth)
        sem_hits = self._search_semantic(query, depth)

        # Weighted RRF uses one-based ranks and shared keyword/semantic weights.
        meta = {h.doc_id: h for h in (*sem_hits, *kw_hits)}
        ordered = fuse_rankings(
            [h.doc_id for h in kw_hits], [h.doc_id for h in sem_hits], k=rrf_k,
        )[:top_k]
        return [
            SearchHit(
                doc_id=doc_id,
                title=meta[doc_id].title,
                text=meta[doc_id].text,
                score=score,
            )
            for doc_id, score in ordered
        ]
