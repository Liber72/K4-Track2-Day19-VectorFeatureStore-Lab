import pytest
from qdrant_client import QdrantClient

from app.cache import SemanticCache


@pytest.fixture(scope="module")
def embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def make(embedder, **kw):
    return SemanticCache(client=QdrantClient(":memory:"), embedder=embedder, **kw)


def test_identical_question_hits(embedder):
    c = make(embedder, threshold=0.75, ttl_s=None)
    c.put("acme", "tối ưu chi phí cloud", "A")
    assert c.get("acme", "tối ưu chi phí cloud").answer == "A"


def test_unrelated_question_misses(embedder):
    c = make(embedder, threshold=0.85, ttl_s=None)
    c.put("acme", "tối ưu chi phí cloud", "A")
    assert c.get("acme", "cách nướng bánh mì") is None


def test_threshold_controls_hits(embedder):
    strict = make(embedder, threshold=0.999, ttl_s=None)
    strict.put("acme", "tối ưu chi phí cloud", "A")
    assert strict.get("acme", "giảm chi phí hạ tầng") is None


def test_ttl_expires_entry(embedder):
    c = make(embedder, threshold=0.75, ttl_s=1800)
    c.put("acme", "giá GPU hiện tại", "A")
    assert c.get("acme", "giá GPU hiện tại") is not None
    c.advance(3600)
    assert c.get("acme", "giá GPU hiện tại") is None
    assert c.stats.stale_evictions == 1


def test_namespacing_blocks_cross_tenant_read(embedder):
    """The security case: tenant B must never receive tenant A's answer."""
    safe = make(embedder, threshold=0.70, ttl_s=None, namespaced=True)
    safe.put("acme", "doanh thu quý 3", "SECRET-ACME")
    assert safe.get("globex", "doanh thu quý 3") is None


def test_missing_namespace_leaks_on_purpose(embedder):
    """Guards the teaching demo in NB7 §4 -- if this ever passes silently,
    the notebook stops demonstrating anything."""
    leaky = make(embedder, threshold=0.70, ttl_s=None, namespaced=False)
    leaky.put("acme", "doanh thu quý 3", "SECRET-ACME")
    stolen = leaky.get("globex", "doanh thu quý 3")
    assert stolen is not None and stolen.answer == "SECRET-ACME"
    assert stolen.tenant == "acme"


def test_stats_track_hits_and_misses(embedder):
    c = make(embedder, threshold=0.85, ttl_s=None)
    c.put("acme", "tối ưu chi phí cloud", "A")
    c.get("acme", "tối ưu chi phí cloud")
    c.get("acme", "cách nướng bánh mì")
    assert c.stats.hits == 1 and c.stats.misses == 1
    assert 0.0 < c.stats.hit_rate < 1.0


def test_new_cache_on_shared_client_keeps_previous_entries(embedder):
    client = QdrantClient(":memory:")
    first = SemanticCache(client=client, embedder=embedder)
    first.put("acme", "câu hỏi của cache A", "A")
    second = SemanticCache(client=client, embedder=embedder)
    second.put("acme", "câu hỏi của cache B", "B")
    assert first.collection != second.collection
    assert first.get("acme", "câu hỏi của cache A").answer == "A"
    assert second.get("acme", "câu hỏi của cache B").answer == "B"


def test_ttl_expires_at_exact_boundary(embedder):
    cache = make(embedder, ttl_s=1800)
    cache.put("acme", "câu hỏi hết hạn", "A")
    cache.advance(1800)
    assert cache.get("acme", "câu hỏi hết hạn") is None
    assert cache.stats.stale_evictions == 1
