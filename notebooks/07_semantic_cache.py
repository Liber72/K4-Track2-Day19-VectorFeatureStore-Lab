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
# # NB7 — Semantic Cache
# Đo ba rủi ro: hit sai, dữ liệu hết hạn và rò chéo tenant. Dùng model từ `.env`
# (máy này: bge-m3), cache Qdrant in-memory riêng cho từng thí nghiệm.
# Không cần LLM. Kết quả lưu ở `submission/results/nb7_semantic_cache.json`.

# %%
import _setup  # noqa: F401
import json
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient

from _advanced import Evidence
from app.cache import SemanticCache
from app.embeddings import Embedder

ROOT = Path(_setup.__file__).resolve().parent.parent
evidence = Evidence("nb7_semantic_cache.json", notebook="NB7")
embedder = Embedder()
client = QdrantClient(":memory:")


def make_cache(**kwargs):
    return SemanticCache(client=client, embedder=embedder, dim=embedder.dim, **kwargs)


print(f"Model: {embedder.model_name}; dim={embedder.dim}")

# %% [markdown]
# ## 1. Hit / miss cơ bản

# %%
cache = make_cache(threshold=0.75, ttl_s=3600)
cache.put("acme", "làm sao tối ưu chi phí cloud", "Dùng spot instance và autoscaling.")
basic = []
for probe in ("làm sao tối ưu chi phí cloud", "cách giảm chi phí hạ tầng đám mây",
              "cách kiểm thử unit test"):
    hit = cache.get("acme", probe)
    basic.append({"query": probe, "hit": hit is not None, "score": hit.score if hit else None})
    print(probe, "->", f"HIT {hit.score:.3f}" if hit else "MISS")

# %% [markdown]
# ## 2. Sweep ngưỡng: tính cả tiết kiệm và trả lời sai
# Warm/cold chia xen kẽ 50 câu golden. Mỗi câu warm có một answer ID riêng.
# Probe chỉ thêm lời dẫn/kết, giữ nguyên ý câu hỏi. Positive đúng khi lấy đúng
# answer ID; hit nhầm câu warm khác cũng tính sai. Mọi hit trên cold tính sai.
# Đây là proxy theo ID câu hỏi, chưa phải đánh giá tương đương câu trả lời bởi người.
# Hai câu khác ID đôi khi có thể dùng chung đáp án; cần nhãn thật khi triển khai.

# %%
with (ROOT / "data/golden_set.jsonl").open(encoding="utf-8") as stream:
    golden = [json.loads(line) for line in stream if line.strip()]
warm, cold = golden[::2], golden[1::2]
sweep = make_cache(threshold=0.0, ttl_s=None)
for query in warm:
    sweep.put("acme", query["query"], f"ANSWER::{query['query_id']}")


def variants(query):
    return [f"cho tôi hỏi {query}", f"{query}, giải thích giúp tôi",
            f"vui lòng giải thích: {query}"]


probes = []
for group, source in (("warm", warm), ("cold", cold)):
    for query in source:
        for variant in variants(query["query"]):
            nearest = sweep.peek("acme", variant)
            assert nearest is not None, "Cache warm không có ứng viên"
            score, payload = nearest
            probes.append({"group": group, "query_id": query["query_id"], "query": variant,
                           "score": score, "matched_answer": payload["answer"],
                           "correct": group == "warm"
                                      and payload["answer"] == f"ANSWER::{query['query_id']}"})

thresholds = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 1.0)
sweep_rows = []
n_positive = sum(p["group"] == "warm" for p in probes)
n_negative = len(probes) - n_positive
for threshold in thresholds:
    accepted = [p for p in probes if p["score"] >= threshold]
    good = sum(p["correct"] for p in accepted)
    wrong = len(accepted) - good
    wrong_positive = sum(p["group"] == "warm" and not p["correct"] for p in accepted)
    wrong_negative = sum(p["group"] == "cold" for p in accepted)
    sweep_rows.append({
        "threshold": threshold,
        "correct_savings": good / len(probes),
        "wrong_answer_rate": wrong / len(probes),
        "hit_rate": len(accepted) / len(probes),
        "wrong_given_hit": wrong / len(accepted) if accepted else 0.0,
        "positive_hit_rate": good / n_positive,
        "negative_false_hit_rate": wrong_negative / n_negative,
        "wrong_positive_count": wrong_positive, "accepted": len(accepted),
    })
table = pd.DataFrame(sweep_rows)
print(f"{len(warm)} câu cache; {n_positive} warm probes; {n_negative} cold probes")
print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print("correct_savings: phần tổng request tránh gọi LLM bằng đáp án đúng theo proxy.")
print("wrong_answer_rate: phần tổng request nhận đáp án sai; hit_rate gồm cả hit sai.")

# %% [markdown]
# ## 3. Chọn ngưỡng từ số đo
# Chính sách demo: tối đa 5% câu trả lời từ cache là sai **trên probe này**.
# Trong các ngưỡng đáp ứng, chọn ngưỡng tiết kiệm đúng nhiều nhất; hòa thì chọn
# ngưỡng cao hơn. Nếu không có hit hữu ích đạt chính sách, tắt semantic reuse.
# Việc chọn và đo cùng probe là tuning, chưa phải kiểm chứng trên held-out set.

# %%
ERROR_BUDGET = 0.05
eligible = [row for row in sweep_rows
            if row["correct_savings"] > 0 and row["wrong_given_hit"] <= ERROR_BUDGET]
selected = max(eligible, key=lambda row: (row["correct_savings"], row["threshold"])) if eligible else None
at_075 = next(row for row in sweep_rows if row["threshold"] == 0.75)
print(f"0.75: wrong/request={at_075['wrong_answer_rate']:.1%}; "
      f"wrong/hit={at_075['wrong_given_hit']:.1%}")
if at_075["wrong_given_hit"] > ERROR_BUDGET:
    print("0.75 chưa đủ: vượt mức lỗi 5% đã chọn cho demo.")
else:
    print("0.75 đạt mức lỗi trên mẫu này; vẫn cần held-out queries và nhãn answer thật.")
print("Ngưỡng chọn:", selected if selected else "Tắt semantic reuse: chưa có ngưỡng phù hợp")

# %% [markdown]
# ## 4. TTL bằng đồng hồ ảo
# `advance()` nhận delta thời gian. In đúng các mốc 0, 600, 1800, 3600 giây;
# tại `age >= ttl` entry hết hạn. Dữ liệu dưới đây là câu trả lời giả cho demo.

# %%
ttl_cache = make_cache(threshold=0.75, ttl_s=1800)
ttl_cache.put("acme", "bảng giá dịch vụ demo hiện tại", "Bảng giá demo phiên bản 1.")
ttl_rows, previous = [], 0
for elapsed in (0, 600, 1800, 3600):
    ttl_cache.advance(elapsed - previous)
    previous = elapsed
    hit = ttl_cache.get("acme", "bảng giá dịch vụ demo hiện tại")
    ttl_rows.append({"elapsed_s": elapsed, "hit": hit is not None})
    print(f"t={elapsed:4d}s -> {'HIT' if hit else 'MISS'}")
print("Stale evictions:", ttl_cache.stats.stale_evictions)

# %% [markdown]
# ## 5. Rò chéo tenant và chặn bằng namespace
# Dùng dữ liệu giả. Hai cache tồn tại đồng thời và giữ collection riêng.
# Tenant identity trong ứng dụng thật phải lấy từ phiên đăng nhập đã xác thực.

# %%
leaky = make_cache(threshold=0.70, ttl_s=None, namespaced=False)
safe = make_cache(threshold=0.70, ttl_s=None, namespaced=True)
for instance in (leaky, safe):
    instance.put("acme", "doanh thu quý 3 của chúng tôi", "ACME_DEMO_ONLY: 4,2 tỷ VND.")
stolen = leaky.get("globex", "doanh thu quý 3 của chúng tôi")
blocked = safe.get("globex", "doanh thu quý 3 của chúng tôi")
print("namespaced=False ->", stolen.answer if stolen else "MISS")
print("Chủ sở hữu:", stolen.tenant if stolen else None)
print("namespaced=True ->", blocked.answer if blocked else "MISS")

# %% [markdown]
# ## 6. Lưu bằng chứng
# Chụp bảng sweep + ngưỡng chọn, TTL và output tenant. Khi đổi embedding model,
# dùng cache mới cho model đó. Sau khi lưu notebook, tắt kernel trước NB8.

# %%
evidence.finish(
    checks={
        "identical_query_hits": basic[0]["hit"],
        "sweep_has_savings_and_errors": len(sweep_rows) == len(thresholds),
        "useful_threshold_within_error_budget": selected is not None,
        "ttl_hit_then_expired": [r["hit"] for r in ttl_rows] == [True, True, False, False],
        "stale_entry_evicted": ttl_cache.stats.stale_evictions >= 1,
        "cross_tenant_leak_demonstrated": stolen is not None and stolen.tenant == "acme",
        "namespace_blocks_leak": blocked is None,
    },
    model=embedder.model_name, dim=embedder.dim, basic=basic,
    sweep=sweep_rows, probes=probes, selected_threshold=selected,
    error_budget=ERROR_BUDGET, ttl=ttl_rows,
    tenant_demo={"leaked_answer": stolen.answer if stolen else None,
                 "safe_miss": blocked is None},
)

# %%
