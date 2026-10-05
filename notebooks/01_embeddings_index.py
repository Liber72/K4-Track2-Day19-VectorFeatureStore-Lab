# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python (Lab 19)
#     language: python
#     name: lab19
# ---

# %% [markdown]
# # NB1 — Embeddings & Vector Indexing
#
# **Stack:** `BAAI/bge-m3` (Sentence Transformers) + Qdrant server trong Docker.
# Maps to slide §1 (Embeddings) + §2 (Vector DB Landscape) + deliverable bullet 1.
#
# > Mục tiêu: hiểu cách 1 đoạn text được biến thành vector dày, và cách Qdrant
# > index + query vectors đó. Python chạy trong kernel Lab 19, Qdrant chạy
# > trong Docker. Cấu hình model và địa chỉ server được đọc từ `.env`.

# %%
import _setup  # noqa: F401  -- adds repo root to sys.path
import json
import os
from pathlib import Path

from app.embeddings import Embedder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

DATA = Path(_setup.__file__).resolve().parent.parent / "data"

# %% [markdown]
# ## 1. Load corpus
#
# Corpus được sinh bởi `scripts/seed_corpus.py` (chạy trong `make seed`):
# 1000 docs tiếng Việt, 10 chủ đề × 100 docs/chủ đề. Mỗi doc có `doc_id`,
# `topic`, `title`, `text`.

# %%
docs = []
with (DATA / "corpus_vn.jsonl").open(encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))

print(f"Corpus size: {len(docs)} docs")
assert len(docs) == 1000, f"expected 1000 docs, got {len(docs)}"
print(f"First doc:")
print(json.dumps(docs[0], ensure_ascii=False, indent=2))

# %% [markdown]
# ## 2. Embedding model: `BAAI/bge-m3`
#
# `Embedder` chọn model theo `EMBEDDING_BACKEND` trong `.env`.
# Với `bge-m3`, mỗi đoạn văn được biểu diễn bằng vector 1024 chiều.
#
# > Cell này tải model nếu chưa có trong cache, rồi embed một câu mẫu.
# > Lần đầu có thể mất vài phút hoặc lâu hơn, tùy mạng và máy.

# %%
embedder = Embedder()
print(f"Backend: {embedder.backend}")
print(f"Model: {embedder.model_name}")
assert embedder.backend == "bge-m3", "Kiểm tra EMBEDDING_BACKEND trong .env và restart kernel"

sample = next(embedder.embed(["cloud computing tiếng Việt"]))
print(f"Vector dim: {len(sample)}")
print(f"First 8 values: {sample[:8].tolist()}")
assert len(sample) == embedder.dim == 1024

# %% [markdown]
# ## 3. Kết nối Qdrant server và tạo collection
#
# Collection `lab19` chứa vector và payload của tài liệu. Số chiều của
# collection phải khớp với model; Cosine dùng để đo độ tương đồng.
# Nếu collection đã tồn tại, kiểm tra cấu hình và dùng lại để chạy lại notebook.

# %%
assert os.getenv("QDRANT_MODE") == "server", "Đặt QDRANT_MODE=server trong .env"
qdrant_url = os.environ["QDRANT_URL"]
client = QdrantClient(url=qdrant_url, timeout=60)

if not client.collection_exists("lab19"):
    client.create_collection(
        collection_name="lab19",
        vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
    )

info = client.get_collection("lab19")
vector_config = info.config.params.vectors
assert isinstance(vector_config, VectorParams), "Collection lab19 cần một vector không đặt tên"
assert vector_config.size == embedder.dim, (
    f"Collection có {vector_config.size} chiều, model cần {embedder.dim} chiều"
)
assert vector_config.distance == Distance.COSINE, "Collection lab19 cần khoảng cách Cosine"
print(f"Qdrant: {qdrant_url}")
print(f"Collection: lab19 — {embedder.dim} chiều, Cosine")

# %% [markdown]
# ## 4. Embed + upsert toàn bộ corpus
#
# Embed `title + " " + text` theo batch 32 tài liệu, rồi upsert từng batch
# vào collection `lab19`. `wait=True` đợi Qdrant ghi xong trước khi tiếp tục.
#
# ID dựa trên vị trí trong corpus, nên chạy lại cùng dữ liệu sẽ cập nhật
# các point đã có. Payload giữ doc_id, topic, title và text để đọc kết quả.

# %%
BATCH = 32
for start in range(0, len(docs), BATCH):
    batch = docs[start:start + BATCH]
    texts = [d["title"] + " " + d["text"] for d in batch]
    vectors = list(embedder.embed(texts))
    assert len(vectors) == len(batch), "Mỗi tài liệu phải có một vector"
    points = [
        PointStruct(
            id=start + i,
            vector=v.tolist(),
            payload={
                "doc_id": d["doc_id"], "topic": d["topic"],
                "title": d["title"], "text": d["text"],
            },
        )
        for i, (d, v) in enumerate(zip(batch, vectors))
    ]

    client.upsert(collection_name="lab19", points=points, wait=True)
    print(f"Indexed: {start + len(batch)}/{len(docs)}")

n_indexed = client.count(collection_name="lab19", exact=True).count
print(f"Indexed: {n_indexed} vectors")
assert n_indexed == 1000, f"expected 1000 indexed, got {n_indexed}"

# %%
assert client.count(collection_name="lab19").count == 1000
print("PASS- Đã đủ 1000 vectors")

# %% [markdown]
# ## 5. First similarity search
#
# Top-5 docs gần nhất với câu query. Chú ý: cùng 1 query có thể trả về docs
# từ nhiều topic — đó là dấu hiệu vector embedding tổng quát. Để filter theo
# topic, dùng Qdrant payload filter.

# %%
query = "cloud computing và tự động mở rộng"
q_vec = next(embedder.embed([query])).tolist()
hits = client.query_points(collection_name="lab19", query=q_vec, limit=5).points

print(f"Query: {query!r}")
print(f"Top-5:")
for i, h in enumerate(hits, 1):
    print(f"  {i}. [{h.payload['topic']:>9}] score={h.score:.3f}  {h.payload['title']}")

# %% [markdown]
# ## 6. Quick sanity — top-5 should be mostly `cloud` topic
#
# Vector embedding cluster theo semantic, không cần keyword "cloud" xuất hiện
# trong doc. Chạy lại với query 100% paraphrase, không có chữ "cloud":

# %%
query2 = "phương pháp tự động mở rộng hạ tầng theo lưu lượng người dùng"
q_vec2 = next(embedder.embed([query2])).tolist()
hits2 = client.query_points(collection_name="lab19", query=q_vec2, limit=5).points

print(f"Query (paraphrase): {query2!r}")
for h in hits2:
    print(f"  [{h.payload['topic']:>9}] score={h.score:.3f}  {h.payload['title']}")

cloud_count = sum(h.payload["topic"] == "cloud" for h in hits2)
print(f"Cloud trong top-5: {cloud_count}/{len(hits2)}")
if len(hits2) == 5 and cloud_count >= 3:
    print("PASS — phần lớn kết quả thuộc topic cloud")
else:
    print("Chưa đạt mục tiêu paraphrase; giữ output để phân tích kết quả")

# %% [markdown]
# ## Deliverable evidence (chụp màn hình)
#
# 1. Mục 2: model `BAAI/bge-m3`, `Vector dim: 1024`.
# 2. Mục 3 và 4: kết nối Qdrant server, `Indexed: 1000 vectors`.
# 3. Mục 5 và 6: top-5 của hai query, kèm số kết quả thuộc topic `cloud`.
# Lưu notebook bằng Ctrl+S và ảnh vào `submission/screenshots/`.
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** the `for start in range(0, ..., BATCH)` upsert loop —
# pattern is mechanical, AI generates it perfectly. Just give it the spec
# (batch size, payload schema) and review the diff.
#
# **Think hard yourself:** vì sao chọn `BAAI/bge-m3` cho corpus tiếng Việt?
# Model hỗ trợ nhiều ngôn ngữ, nhưng cần nhiều tài nguyên hơn model nhỏ.
# Đổi model có thể thay đổi số chiều và cần index lại. **Don't ask AI to
# pick the embedding model without first telling it: language(s), corpus
# size, latency budget, and re-index cost.** Đây là 1 quyết định kiến trúc,
# không phải boilerplate.

# %%
