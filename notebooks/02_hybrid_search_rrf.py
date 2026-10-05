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
# # NB2 — Hybrid Search: BM25 + Vector + RRF
#
# **Stack:** BM25 + bge-m3 + Qdrant Docker; dùng collection `lab19` của NB1.
# Mục tiêu: so sánh Precision@10 trên cùng 50 golden queries.
# BM25 tìm theo từ khóa, vector tìm theo ý nghĩa, RRF kết hợp thứ hạng.
# Kết quả phụ thuộc corpus và model; chỉ kết luận sau khi đo.
#
# Trước khi chạy: lưu NB1 và shutdown kernel NB1 để giải phóng model trong RAM.
# Chọn kernel **Python (Lab 19)** cho notebook này.

# %%
import _setup  # noqa: F401
import json
import os
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from app.evaluation import precision_at_k, save_result
from app.search import COLLECTION, Searcher

DATA = Path(_setup.__file__).resolve().parent.parent / "data"

# %% [markdown]
# ## 1. Đọc corpus, dựng BM25 và dùng lại vector NB1
#
# BM25 chỉ cần token hóa văn bản. Vector đã lưu trong Qdrant được kiểm tra
# số chiều, số tài liệu và payload trước khi dùng; cell này không embed lại corpus.

# %%
assert os.getenv("QDRANT_MODE") == "server", "Kiểm tra .env và restart kernel"
searcher = Searcher.from_corpus(DATA / "corpus_vn.jsonl", require_existing=True)
docs = searcher.docs
bm25, embedder, client = searcher.bm25, searcher.embedder, searcher.client
assert len(docs) == 1000
assert embedder.backend == "bge-m3"
assert searcher.reused_index
print(f"Corpus: {len(docs)} docs")
print(f"Model: {embedder.model_name} ({embedder.dim} chiều)")
print(f"Collection: {COLLECTION} — dùng lại vector từ NB1")
print("BM25 + vector indices ready")
print("Ignored common corpus tokens:", sorted(searcher.bm25_ignored_tokens))

# %% [markdown]
# ## 2. Hai hàm tìm kiếm cơ sở
#
# Mỗi hàm trả danh sách doc_id theo thứ hạng. Khi kết hợp RRF, lấy top-50
# từ mỗi bộ tìm kiếm trước khi chọn top-10.
# Cache vector câu hỏi giúp tránh embed cùng câu hai lần trong phép đo chất lượng.
# API ở NB3 vẫn embed từng request; cache này chỉ dùng trong notebook NB2.

# %%
TOP_K = 10
RRF_K = 60

def search_keyword(query: str, top_k: int = TOP_K) -> list[str]:
    tokens = [token for token in searcher._tokenize(query)
              if token not in searcher.bm25_ignored_tokens]
    scores = bm25.get_scores(tokens)
    # Only documents with positive lexical evidence participate in ranking.
    candidates = [i for i, score in enumerate(scores) if score > 0]
    ranked = sorted(candidates, key=lambda i: -scores[i])[:top_k]
    return [docs[i]["doc_id"] for i in ranked]

@lru_cache(maxsize=128)
def query_vector(query: str) -> tuple[float, ...]:
    return tuple(next(embedder.embed([query])).tolist())

def search_semantic(query: str, top_k: int = TOP_K) -> list[str]:
    result = client.query_points(
        collection_name=COLLECTION,
        query=list(query_vector(query)),
        limit=top_k,
    )
    return [point.payload["doc_id"] for point in result.points]

# %% [markdown]
# ## 3. Reciprocal Rank Fusion
#
# Công thức: score(d) = tổng `1 / (k + rank)` từ hai bộ tìm kiếm.
# Rank bắt đầu từ **1**, k = **60**. Không cộng trực tiếp BM25 score với cosine,
# vì hai thang điểm có ý nghĩa khác nhau.

# %%
def search_hybrid(query: str, top_k: int = TOP_K, rrf_k: int = RRF_K) -> list[str]:
    depth = max(top_k * 5, 50)
    rankings = (search_keyword(query, depth), search_semantic(query, depth))
    scores: dict[str, float] = {}
    for ranked_ids in rankings:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: -item[1])[:top_k]]

test_q = "co giãn linh hoạt theo nhu cầu sử dụng"
print(f"Query: {test_q}")
for mode, fn in (("keyword", search_keyword), ("semantic", search_semantic), ("hybrid", search_hybrid)):
    print(f"  {mode:8} top-3: {fn(test_q)[:3]}")

# %% [markdown]
# ## 4. Đánh giá trên 50 golden queries
#
# Precision@10 = số tài liệu relevant trong top-10 / 10.
# Dùng relevant_doc_ids của golden set; cùng query và cùng corpus cho ba mode.
# Cell in tiến độ, bảng trung bình và kiểm tra mục tiêu hybrid thắng cả hai mode.

# %%
with (DATA / "golden_set.jsonl").open(encoding="utf-8") as f:
    golden = [json.loads(line) for line in f if line.strip()]
assert len(golden) == 50

p_kw, p_sem, p_hyb, query_results = [], [], [], []
for i, q in enumerate(golden, start=1):
    relevant = set(q["relevant_doc_ids"])
    scores = {
        "keyword": precision_at_k(search_keyword(q["query"]), relevant, TOP_K),
        "semantic": precision_at_k(search_semantic(q["query"]), relevant, TOP_K),
        "hybrid": precision_at_k(search_hybrid(q["query"]), relevant, TOP_K),
    }
    p_kw.append(scores["keyword"])
    p_sem.append(scores["semantic"])
    p_hyb.append(scores["hybrid"])
    query_results.append({
        "query_id": q["query_id"], "query": q["query"],
        "type": q["mode_hint"], "topic": q["topic"], **scores,
    })
    if i % 10 == 0:
        print(f"Evaluated: {i}/{len(golden)} queries")

averages = {
    "keyword": statistics.mean(p_kw),
    "semantic": statistics.mean(p_sem),
    "hybrid": statistics.mean(p_hyb),
}
print("\nPrecision@10 — trung bình 50 queries")
print(f"  {'mode':10} {'Precision@10':>14}")
for mode, value in averages.items():
    print(f"  {mode:10} {value:>13.1%}")
hybrid_beats_both = averages["hybrid"] > max(averages["keyword"], averages["semantic"])
print("PASS — hybrid thắng cả keyword và semantic" if hybrid_beats_both
      else "Chưa đạt mục tiêu hybrid thắng cả hai; giữ số đo để phân tích.")

# %% [markdown]
# ## 5. So sánh theo loại query và lưu kết quả
#
# exact: có từ khóa rõ; paraphrase: diễn đạt cùng ý bằng từ khác;
# mixed: kết hợp từ khóa và diễn đạt lại. Quan sát mode thắng ở từng nhóm.
# File JSON lưu số đo thật để tạo reflection sau khi hoàn thành NB4.

# %%
by_type = defaultdict(lambda: defaultdict(list))
for q, scores in zip(golden, query_results):
    for mode in averages:
        by_type[q["mode_hint"]][mode].append(scores[mode])

slice_results = {}
print(f"  {'type':12} {'n':>3} {'keyword':>10} {'semantic':>10} {'hybrid':>10}")
for query_type in ("exact", "paraphrase", "mixed"):
    group = by_type[query_type]
    values = {mode: statistics.mean(group[mode]) for mode in averages}
    slice_results[query_type] = {"n": len(group["keyword"]), **values}
    print(f"  {query_type:12} {len(group['keyword']):>3} "
          f"{values['keyword']:>9.1%} {values['semantic']:>9.1%} {values['hybrid']:>9.1%}")

result_path = save_result("nb2_quality.json", {
    "n_queries": len(golden), "top_k": TOP_K, "rrf_k": RRF_K,
    "embedding_backend": embedder.backend, "collection": COLLECTION,
    "keyword_min_score_exclusive": 0.0,
    "keyword_preprocessing": "unicode_tokens_corpus_df_below_80_percent",
    "keyword_ignored_tokens": sorted(searcher.bm25_ignored_tokens),
    "average": averages, "by_type": slice_results,
    "hybrid_beats_both": hybrid_beats_both, "queries": query_results,
})
print(f"Saved: {result_path}")

# %% [markdown]
# ## Bằng chứng cần lưu
#
# - Mục 4: bảng Precision@10 và kết luận từ số đo.
# - Mục 5: bảng exact/paraphrase/mixed.
# - Ctrl+S để giữ output notebook.
# - Ảnh: `submission/screenshots/nb2_precision.png` và `nb2_slices.png`.
#
# Khi hybrid chưa thắng: đối chiếu từng query trong nb2_quality.json và các
# danh sách top-10, kiểm tra RRF, chất lượng BM25 và model. Giữ nguyên số đo.

# %%
