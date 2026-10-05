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
# # NB5 — Filtered Search
# Chạy từng cell từ trên xuống. Docker dùng lại vector NB1 với model trong `.env`.
# Kết quả thực tế tự lưu ở `submission/results/nb5_filtered_search.json`.
# Ground truth là top-10 cosine chính xác **trong subset khớp filter**.
# `post-filter` lấy ứng viên rồi bỏ doc không khớp; `pre-filter` quét subset;
# `filtered-ANN` truyền điều kiện vào Qdrant. Không gán sẵn recall cho chiến lược nào.

# %%
import _setup  # noqa: F401
import os
from pathlib import Path

import pandas as pd

from _advanced import Evidence
from app.filters import FilteredIndex, access_filter, combo_filter, recent_filter, tenant_filter
from app.metadata import selectivity
from app.search import Searcher

ROOT = Path(_setup.__file__).resolve().parent.parent
evidence = Evidence("nb5_filtered_search.json", notebook="NB5")

# %% [markdown]
# ## 1. Dùng lại index NB1
# `lab19_filtered_nb5` là bản sao có metadata, được dựng lại khi chạy cell này.
# Collection `lab19` giữ nguyên. Chế độ memory dựng index tạm trong process.

# %%
searcher = Searcher.from_corpus(
    ROOT / "data/corpus_vn.jsonl", require_existing=os.getenv("QDRANT_MODE") == "server",
)
index = FilteredIndex.from_searcher(searcher, collection="lab19_filtered_nb5")
model = {"model": searcher.embedder.model_name, "dim": searcher.embedder.dim,
         "qdrant_mode": os.getenv("QDRANT_MODE", "memory"),
         "reused_nb1": searcher.reused_index, "n_docs": len(index.docs)}
print(model)
print("Payload mẫu:", {key: index.docs[0][key]
                        for key in ("doc_id", "topic", "tenant", "access", "published")})

# %% [markdown]
# ## 2. Recall theo độ chọn lọc
# Latency bên dưới đo retrieval, không gồm embedding query. `pre_ms` gồm lọc
# metadata và quét cosine trong Python; hai cột kia gồm round-trip tới Qdrant.

# %%
QUERY = "tự động mở rộng hệ thống theo lưu lượng"
cases = [
    ("không filter", lambda doc: True, None),
    ("access=internal", *access_filter("internal")),
    ("tenant=acme", *tenant_filter("acme")),
    ("published >=2026", *recent_filter(20260101)),
    ("acme AND >=2026", *combo_filter("acme", 20260101)),
]
rows, retrievals = [], []
for name, predicate, qfilter in cases:
    truth = index.pre_filter(QUERY, predicate, k=10)
    post = index.post_filter(QUERY, predicate, k=10, fetch_k=10)
    filtered = index.filtered_ann(QUERY, qfilter, k=10)
    rows.append({
        "filter": name, "selectivity_pct": 100 * selectivity(index.docs, predicate),
        "post_recall": post.recall_against(truth.doc_ids),
        "filtered_recall": filtered.recall_against(truth.doc_ids),
        "pre_ms": truth.latency_ms, "post_ms": post.latency_ms,
        "filtered_ms": filtered.latency_ms,
    })
    retrievals.append({"filter": name, "truth": truth.doc_ids,
                       "post": post.doc_ids, "filtered": filtered.doc_ids})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# %% [markdown]
# Mục tiêu rubric: post-filter giảm rõ khi filter chặt, filtered search đạt 1.00.
# Đây là điều cần kiểm chứng trên model đang chạy, không phải bảo đảm của ANN.
# Với collection nhỏ, Qdrant có thể chọn exact scan thay vì HNSW; notebook không
# đo số vector engine thực sự duyệt. Xem [tài liệu Qdrant](https://qdrant.tech/documentation/search-patterns/vector-search-filtering/).

# %% [markdown]
# ## 3. Over-fetch ladder
# `% corpus trả về` là số ứng viên yêu cầu / kích thước corpus, **không phải**
# phần trăm vector đã được Qdrant quét. Đo trên ba query với cùng filter chặt.

# %%
predicate, qfilter = combo_filter("acme", 20260101)
queries = [QUERY, "bảo mật xác thực người dùng", "mô hình ngôn ngữ lớn"]
truths = {q: index.pre_filter(q, predicate, k=10).doc_ids for q in queries}
ladder = []
for fetch_k in sorted({min(n, len(index.docs)) for n in (10, 50, 200, 500, len(index.docs))}):
    recalls = [index.post_filter(q, predicate, k=10, fetch_k=fetch_k)
               .recall_against(truths[q]) for q in queries]
    ladder.append({"fetch_k": fetch_k, "recall": sum(recalls) / len(recalls),
                   "candidate_pct": 100 * fetch_k / len(index.docs),
                   "per_query_recall": recalls})
print(pd.DataFrame(ladder).drop(columns="per_query_recall").round(3).to_string(index=False))
filtered_recalls = [index.filtered_ann(q, qfilter, k=10).recall_against(truths[q])
                    for q in queries]
full = next((row for row in ladder if row["recall"] >= 1 - 1e-9), None)
print("Filtered-search recall:", filtered_recalls)
print("Mốc đầu trong ladder đạt recall=1:", full)
print("Rubric kỳ vọng khoảng 50%; giữ số đo thực tế nếu bge-m3 cho mốc khác.")

# %% [markdown]
# ## 4. Kiểm tra từng tenant

# %%
tenant_rows = []
for tenant in ("acme", "globex", "initech"):
    pred_t, qf_t = tenant_filter(tenant)
    truth = index.pre_filter(QUERY, pred_t, k=10).doc_ids
    post = index.post_filter(QUERY, pred_t, k=10, fetch_k=10)
    filtered = index.filtered_ann(QUERY, qf_t, k=10)
    tenant_rows.append({"tenant": tenant, "post_recall": post.recall_against(truth),
                        "filtered_recall": filtered.recall_against(truth)})
print(pd.DataFrame(tenant_rows).to_string(index=False))

# %% [markdown]
# ## 5. Lưu bằng chứng
# `REVIEW` nghĩa là đã đo nhưng chưa khớp mục tiêu; không sửa số liệu để đạt rubric.
# Chụp bảng mục 2 và 3, lưu notebook bằng Ctrl+S.

# %%
evidence.finish(
    checks={
        "post_filter_drops_on_tight_filter": rows[-1]["post_recall"] < rows[0]["post_recall"],
        "filtered_recall_is_one": all(row["filtered_recall"] >= 1 - 1e-9 for row in rows)
                                  and all(r >= 1 - 1e-9 for r in filtered_recalls),
        "overfetch_recovers_recall": full is not None,
        "overfetch_near_half_corpus": full is not None and 40 <= full["candidate_pct"] <= 60,
    },
    environment=model, query=QUERY, selectivity=rows, retrievals=retrievals,
    overfetch=ladder, first_full_recall=full, tenants=tenant_rows,
    filtered_ladder_recalls=filtered_recalls,
)

# %% [markdown]
# Khi xong, chọn **Kernel → Shut Down Kernel** để giải phóng model trước NB6.
