"""Bonus logic tests; no model downloads, Feast registry, or Docker writes."""
import unicodedata
import json

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from bonus.activity import RecentActivity, detect_topic
from bonus.agent import HybridMemoryAgent, chunk_text, fuse_rankings


class DeterministicEmbedder:
    dim = 4
    model_name = "test-bonus-model"

    def embed(self, texts):
        for text in texts:
            words = text.casefold()
            yield np.array([words.count("kubernetes"), words.count("security"),
                            words.count("autoscaling"), 0.1], dtype=np.float32)


class FakeFeatureStore:
    def __init__(self, timestamp):
        self.rows = {
            "u_001": {"preferred_language": "mix", "reading_speed_wpm": 240,
                      "topic_affinity": "cloud", "active_hours_local": "20:00-23:00",
                      "profile_updated_at_ts": timestamp},
            "u_002": {"preferred_language": "vi", "reading_speed_wpm": 180,
                      "topic_affinity": "ai_ml", "active_hours_local": "07:00-09:00",
                      "profile_updated_at_ts": timestamp},
        }
        self.pushes = []

    def push(self, source, frame, to):
        self.pushes.append((source, frame.copy()))
        for row in frame.to_dict("records"):
            self.rows.setdefault(row["user_id"], {}).update(row)

    def get_online_features(self, *, features, entity_rows):
        row = self.rows.get(entity_rows[0]["user_id"], {})
        values = {ref.split(":")[1]: [row.get(ref.split(":")[1])] for ref in features}

        class Result:
            def to_dict(self):
                return values

        return Result()


@pytest.fixture
def make_agent():
    clients = []

    def factory():
        now = [100_000.0]
        store = FakeFeatureStore(int(now[0]))
        client = QdrantClient(":memory:")
        clients.append(client)
        activity = RecentActivity(store, clock=lambda: now[0],
                                  snapshot_writer=lambda *args: None)
        agent = HybridMemoryAgent(client=client, embedder=DeterministicEmbedder(),
                                  feature_store=store, activity=activity, clock=lambda: now[0])
        return agent, store, now

    yield factory
    for client in clients:
        client.close()


def test_chunk_overlap_preserves_tail_without_redundant_chunk():
    words = [f"w{i}" for i in range(221)]
    chunks = [chunk.split() for chunk in chunk_text(" ".join(words))]
    assert list(map(len, chunks)) == [120, 120, 21]
    assert chunks[0][-20:] == chunks[1][:20]
    assert chunks[1][-20:] == chunks[2][:20]
    assert chunks[2][-1] == "w220"
    assert len(chunk_text(" ".join(words[:120]))) == 1


def test_rrf_uses_one_based_rank_and_deterministic_ties():
    # With zero-based ranks and k=1, 'a' incorrectly ties 'b' and sorts first.
    assert fuse_rankings(["a", "b"], ["c", "b"], k=1) == ["b", "a", "c"]
    assert fuse_rankings(["a", "b"], ["c", "b"]) == ["b", "a", "c"]
    assert fuse_rankings([], []) == []


def test_remember_is_idempotent_across_nfc_and_whitespace(make_agent):
    agent, _, _ = make_agent()
    text = "Kubernetes  bảo mật\ncloud"
    agent.remember(text)
    agent.remember(unicodedata.normalize("NFD", "Kubernetes bảo mật cloud"))
    assert agent.client.count(agent.collection, exact=True).count == 1
    agent.remember(text, "u_002")
    assert agent.client.count(agent.collection, exact=True).count == 2


def test_both_retrievers_exclude_other_users(make_agent):
    agent, _, _ = make_agent()
    for text in ("Kubernetes cluster", "security IAM", "autoscaling cloud"):
        agent.remember(text)
    agent.remember("FOREIGN_SECRET Kubernetes security autoscaling", "u_002")
    context = agent.recall("FOREIGN_SECRET Kubernetes security autoscaling")
    assert "FOREIGN_SECRET" not in context.split("Top memories", 1)[1]
    assert "topic_affinity=cloud;" in context
    assert "queries_last_hour=1;" in context
    other = agent.recall("Kubernetes", "u_002")
    assert "FOREIGN_SECRET" in other
    assert "topic_affinity=ai_ml;" in other
    assert "queries_last_hour=1;" in other


def test_filtered_scroll_collects_every_page(make_agent):
    agent, _, _ = make_agent()
    points = [models.PointStruct(
        id=index, vector=[1.0, 0.0, 0.0, 0.1],
        payload={"user_id": "u_001" if index < 260 else "u_002", "text": str(index)},
    ) for index in range(275)]
    agent.client.upsert(agent.collection, points=points, wait=True)
    memories = agent._memories("u_001")
    assert len(memories) == 260
    assert {point.id for point in memories} == set(range(260))
    assert all(point.payload["user_id"] == "u_001" for point in memories)


def test_unknown_user_has_no_foreign_evidence_and_no_invented_profile(make_agent):
    agent, _, _ = make_agent()
    agent.remember("Kubernetes PRIVATE_U001")
    context = agent.recall("Kubernetes", "u_new")
    assert "(chưa có memory)" in context
    assert "PRIVATE_U001" not in context
    assert "topic_affinity=unknown (missing/stale)" in context
    assert "queries_last_hour=1;" in context


def test_stale_profile_is_explicit_but_activity_remains_fresh(make_agent):
    agent, store, now = make_agent()
    store.rows["u_001"]["profile_updated_at_ts"] = int(now[0] - 30 * 86400 - 1)
    context = agent.recall("Kubernetes")
    assert "reading_speed_wpm=unknown (missing/stale)" in context
    assert "topic_affinity=unknown (missing/stale)" in context
    assert "queries_last_hour=1;" in context


def test_activity_windows_expire_events_without_dropping_24h_topics(make_agent):
    agent, store, now = make_agent()
    agent.recall("Kubernetes")
    now[0] += 3599
    agent.recall("security")
    now[0] += 1
    context = agent.recall("database")
    assert "queries_last_hour=2;" in context  # first query is exactly 1h old
    assert "distinct_topics_24h=3;" in context
    now[0] = 100_000 + 86400
    agent.recall("embedding")
    assert store.rows["u_001"]["queries_last_hour"] == 1
    assert store.rows["u_001"]["distinct_topics_24h"] == 3  # cloud expired
    assert store.pushes[-1][0] == "bonus_activity_push"


def test_topic_rules_do_not_match_ai_inside_vietnamese_words():
    assert detect_topic("Tài liệu mới") == "other"
    assert detect_topic("AI và embedding") == "ai_ml"
    assert detect_topic("cloud security") == "security"


@pytest.mark.parametrize("text,user_id", [(" ", "u_001"), ("hi", ""), ("hi", "u:002")])
def test_invalid_input_is_rejected_before_any_write(make_agent, text, user_id):
    agent, store, _ = make_agent()
    with pytest.raises(ValueError):
        agent.remember(text, user_id)
    with pytest.raises(ValueError):
        agent.recall(text, user_id)
    assert not store.pushes
    assert agent.client.count(agent.collection, exact=True).count == 0


def test_demo_exercises_five_queries_and_both_methods(make_agent, capsys):
    from bonus.demo import run_demo

    agent, _, _ = make_agent()
    results = run_demo(agent)
    assert len(results) == 5
    assert all("PRIVATE_U002" not in result["context"] for result in results)
    assert agent.client.count(agent.collection, exact=True).count == 6
    assert capsys.readouterr().out.count("Query ") == 5


def test_failed_evidence_replaces_previous_pass_report(tmp_path):
    from bonus.demo import save_evidence

    report = {"status": "passed", "started_at_utc": "start", "finished_at_utc": "end",
              "results": [{"query": "q", "context": "old context"}]}
    save_evidence(tmp_path, report)
    report.update({"status": "failed", "results": [], "error": "RuntimeError: push failed"})
    save_evidence(tmp_path, report)
    saved = json.loads((tmp_path / "bonus_demo.json").read_text(encoding="utf-8"))
    assert saved["status"] == "failed" and saved["results"] == []
    transcript = (tmp_path / "bonus_demo.txt").read_text(encoding="utf-8")
    assert "old context" not in transcript
    assert "RuntimeError: push failed" in transcript
