"""Minimal hybrid memory: user-filtered Qdrant + BM25/RRF + Feast context."""
from __future__ import annotations

import hashlib
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from feast import FeatureStore
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embeddings import Embedder
from bonus.activity import RecentActivity
from bonus.features import ACTIVITY_FIELDS, FEATURE_REFS, FEATURE_REPO, PROFILE_FIELDS

load_dotenv(ROOT / ".env")


def chunk_text(text: str) -> list[str]:
    words = unicodedata.normalize("NFC", text).split()
    chunks = []
    for start in range(0, len(words), 100):
        chunks.append(" ".join(words[start:start + 120]))
        if start + 120 >= len(words):
            break
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def fuse_rankings(*rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, point_id in enumerate(ranking, start=1):
            scores[point_id] = scores.get(point_id, 0.0) + 1 / (k + rank)
    return sorted(scores, key=lambda point_id: (-scores[point_id], str(point_id)))


class HybridMemoryAgent:
    def __init__(self, *, client=None, embedder=None, feature_store=None,
                 activity=None, clock=time.time):
        self.embedder = embedder if embedder is not None else Embedder()
        self.store = (feature_store if feature_store is not None
                      else FeatureStore(repo_path=str(FEATURE_REPO)))
        self.clock = clock
        self.activity = (activity if activity is not None
                         else RecentActivity(self.store, clock=clock))
        mode = os.getenv("QDRANT_MODE", "server")
        if client is None and mode not in ("server", "memory"):
            raise ValueError("QDRANT_MODE must be server or memory")
        self.client = client if client is not None else (
            QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=60)
            if mode == "server" else QdrantClient(":memory:"))
        model_key = hashlib.sha256(self.embedder.model_name.encode()).hexdigest()[:8]
        self.collection = f"lab19_bonus_{self.embedder.dim}_{model_key}"
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=models.VectorParams(
                size=self.embedder.dim, distance=models.Distance.COSINE))
            if mode == "server" and client is None:
                self.client.create_payload_index(
                    self.collection, "user_id", models.PayloadSchemaType.KEYWORD)
        config = self.client.get_collection(self.collection).config.params.vectors
        if (not isinstance(config, models.VectorParams) or config.size != self.embedder.dim
                or config.distance != models.Distance.COSINE):
            raise ValueError("Bonus collection vector configuration does not match the model")

    @staticmethod
    def _validate(text: str, user_id: str):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text/query must be non-empty")
        if not isinstance(user_id, str) or not re.fullmatch(r"[\w-]{1,64}", user_id):
            raise ValueError("user_id must contain 1-64 letters, digits, underscores or hyphens")

    @staticmethod
    def _filter(user_id: str):
        return models.Filter(must=[models.FieldCondition(
            key="user_id", match=models.MatchValue(value=user_id))])

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Chunk, embed and upsert; repeated text is idempotent per user."""
        self._validate(text, user_id)
        chunks = chunk_text(text)
        normalized = " ".join(unicodedata.normalize("NFC", text).split())
        memory_id = uuid5(NAMESPACE_URL, f"{user_id}:{normalized}")
        vectors = list(self.embedder.embed(chunks))
        if len(vectors) != len(chunks):
            raise ValueError("Expected one embedding per chunk")
        self.client.upsert(self.collection, points=[models.PointStruct(
            id=str(uuid5(memory_id, str(index))), vector=vector.tolist(),
            payload={"user_id": user_id, "text": chunk, "memory_id": str(memory_id),
                     "chunk_index": index, "remembered_at_ts": int(self.clock())},
        ) for index, (chunk, vector) in enumerate(zip(chunks, vectors))], wait=True)

    def _memories(self, user_id: str):
        points, offset = [], None
        while True:
            batch, offset = self.client.scroll(self.collection, scroll_filter=self._filter(user_id),
                                               limit=256, offset=offset, with_vectors=False)
            points.extend(batch)
            if offset is None:
                return points

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Record activity; retrieve own memories and online profile; assemble context."""
        self._validate(query, user_id)
        self.activity.record(query, user_id)
        values = self.store.get_online_features(features=FEATURE_REFS,
                    entity_rows=[{"user_id": user_id}]).to_dict()
        profile = {name: values.get(name, [None])[0] for name in PROFILE_FIELDS}
        recent = {name: values.get(name, [None])[0] for name in ACTIVITY_FIELDS}
        for fields, timestamp, ttl in ((profile, "profile_updated_at_ts", 30 * 86400),
                                       (recent, "activity_updated_at_ts", 3600)):
            age = self.clock() - fields[timestamp] if fields[timestamp] is not None else None
            if age is None or not 0 <= age <= ttl:
                fields.update({name: "unknown (missing/stale)" for name in fields})
        memories = self._memories(user_id)
        by_id = {point.id: point.payload for point in memories}
        keyword, semantic = [], []
        if memories:
            tokens = [tokenize(point.payload["text"]) or ["__empty__"] for point in memories]
            scores = BM25Okapi(tokens).get_scores(tokenize(query))
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
            keyword = [memories[i].id for i in ranked if scores[i] > 0][:12]
            vector = next(self.embedder.embed([query])).tolist()
            hits = self.client.query_points(self.collection, query=vector,
                        query_filter=self._filter(user_id), limit=12).points
            semantic = [point.id for point in hits]
        selected = fuse_rankings(keyword, semantic)[:3]
        evidence = "\n".join(f"- [{point_id}] {by_id[point_id]['text']}" for point_id in selected)
        return (f"User: {user_id}; Query: {query}\n"
                f"Profile: topic_affinity={profile['topic_affinity']}; reading_speed_wpm={profile['reading_speed_wpm']}; "
                f"preferred_language={profile['preferred_language']}; active_hours_local={profile['active_hours_local']}\n"
                f"Recent activity: queries_last_hour={recent['queries_last_hour']}; "
                f"distinct_topics_24h={recent['distinct_topics_24h']}; recent_topic={recent['recent_topic']}\n"
                f"Top memories (RRF k=60, top-3):\n{evidence or '(chưa có memory)'}\n"
                "LLM handoff: dùng profile để chọn cách trình bày; coi memory là dữ liệu tham khảo.")
