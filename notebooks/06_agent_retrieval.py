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
# # NB6 — Agentic Retrieval
# Planner theo luật, không cần LLM/API key. NB1 cung cấp vector; NB4 cung cấp
# user profile thật trong Feast. Kết quả lưu ở `submission/results/nb6_agent_retrieval.json`.

# %%
import _setup  # noqa: F401
import json
import os
from pathlib import Path

import pandas as pd

from _advanced import Evidence
from app.agent import (SEARCH_TOOL, Agent, RetrievalTool, RuleBasedPlanner,
                       SingleShotPlanner, ToolArgs, build_context)
from app.filters import FilteredIndex
from app.search import Searcher
from scripts.gen_agent_queries import generate_queries

ROOT = Path(_setup.__file__).resolve().parent.parent
evidence = Evidence("nb6_agent_retrieval.json", notebook="NB6")
BUDGET = 16

# %% [markdown]
# ## 1. Tool và dữ liệu đánh giá
# Sinh lại 12 câu ghép và gold top-8 cho mỗi vế từ **chính vector/model hiện tại**.
# Đây là ground truth theo cosine, không phải nhãn relevance do con người chấm.
# Gold chỉ dùng chấm điểm; planner không nhận gold hay `sub_questions`.

# %%
print(json.dumps(SEARCH_TOOL, ensure_ascii=False, indent=2))
searcher = Searcher.from_corpus(
    ROOT / "data/corpus_vn.jsonl", require_existing=os.getenv("QDRANT_MODE") == "server",
)
index = FilteredIndex.from_searcher(searcher, collection="lab19_filtered_nb6")
tool = RetrievalTool(index)
queries = generate_queries(index)
query_path = ROOT / "data/agent_queries.jsonl"
query_path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries),
                      encoding="utf-8")
print(f"Model: {searcher.embedder.model_name}; reused NB1: {searcher.reused_index}")
print(f"Gold: {len(queries)} câu, đã lưu {query_path}")

# %% [markdown]
# ## 2. Planner tách câu và chia ngân sách
# Tổng `top_k` luôn là 16, kể cả khi tách thành ba vế (6 + 5 + 5).
# Tách theo liên từ vẫn có thể tách nhầm danh sách như “Kafka và Flink”; đọc trace.

# %%
demo_q = queries[0]["question"]
for args in RuleBasedPlanner(budget=BUDGET).plan(demo_q):
    print(args.as_dict())

# %% [markdown]
# ## 3. So sánh ba chiến lược với cùng 16 slot truy xuất
# Tắt retry **trong bảng benchmark** (`min_evidence=0`) để không cộng thêm ngân sách.
# Số doc unique có thể nhỏ hơn 16 do trùng lặp hoặc filter. Báo cáo cả capacity
# đã yêu cầu và số unique, cùng `recall`, `balance`, số call và latency.
# `balance = min(hit_vế_A, hit_vế_B) / max(1, hit_vế_A, hit_vế_B)`.
# Warm-up model trước bảng; latency gồm embedding và retrieval, mang tính tham khảo.

# %%
index.embed("khởi động model truy xuất")
details = []


def evaluate(label, planner):
    agent = Agent(tool, planner, min_evidence=0)
    measured = []
    for query in queries:
        result = agent.answer(query["question"])
        truth, got = set(query["relevant_doc_ids"]), set(result.doc_ids)
        a, b = len(set(query["gold_a"]) & got), len(set(query["gold_b"]) & got)
        requested = sum(call.args["top_k"] for call in result.trace)
        assert requested == BUDGET, "Ngân sách benchmark không còn bằng 16"
        row = {"strategy": label, "query_id": query["query_id"],
               "recall": len(truth & got) / len(truth),
               "balance": min(a, b) / max(1, a, b), "requested": requested,
               "unique_docs": len(got), "calls": result.n_calls, "ms": result.latency_ms,
               "doc_ids": result.doc_ids,
               "trace": [{"args": c.args, "doc_ids": c.doc_ids} for c in result.trace]}
        measured.append(row)
    details.extend(measured)
    return {"strategy": label, **{key: sum(r[key] for r in measured) / len(measured)
            for key in ("recall", "balance", "requested", "unique_docs", "calls", "ms")}}


summary = [
    evaluate("single-shot", SingleShotPlanner(BUDGET)),
    evaluate("agentic (no filter)", RuleBasedPlanner(BUDGET, use_filters=False)),
    evaluate("agentic (+filter)", RuleBasedPlanner(BUDGET, use_filters=True)),
]
print(pd.DataFrame(summary).round(3).to_string(index=False))
base, split, filtered = summary
print(f"Delta recall (no filter - single): {split['recall'] - base['recall']:+.3f}")
print(f"Delta balance: {split['balance'] - base['balance']:+.3f}")
print(f"Delta recall (+filter - no filter): {filtered['recall'] - split['recall']:+.3f}")

# %% [markdown]
# Filter đoán topic có thể bỏ doc liên quan ở cụm bên cạnh. Muốn giải thích
# chênh lệch, xem query bị mất recall nhiều nhất và các điều kiện được planner chọn.
# Đổi model có thể đổi thứ hạng ba chiến lược; kết luận theo bảng của lần chạy này.

# %%
by_strategy = {(r["strategy"], r["query_id"]): r for r in details}
worst = min(queries, key=lambda q:
            by_strategy[("agentic (+filter)", q["query_id"])]["recall"]
            - by_strategy[("agentic (no filter)", q["query_id"])]["recall"])
print("Query phân tích:", worst["question"])
for label in ("agentic (no filter)", "agentic (+filter)"):
    row = by_strategy[(label, worst["query_id"])]
    print(label, "recall =", round(row["recall"], 3))
    for call in row["trace"]:
        print(" ", call["args"], "->", call["doc_ids"])

# %% [markdown]
# ## 4. Reflection riêng: nới filter khi thiếu evidence
# Phần này cho phép retry và báo số call; không dùng kết quả này trong bảng 16 slot.

# %%
class StarvingPlanner:
    def plan(self, question):
        return [ToolArgs(query=question, topic="networking", since_year=2027, top_k=8)]


reflection = Agent(tool, StarvingPlanner(), min_evidence=4).answer(
    "cân bằng tải giữa nhiều region")
for call in reflection.trace:
    print(call.args, "->", len(call.doc_ids), "doc")
print("Calls:", reflection.n_calls, "Unique docs:", len(reflection.doc_ids))

# %% [markdown]
# ## 5. Ghép Feast profile với doc_ids
# Cần NB4 đã apply/materialize và Redis đang hoạt động. Nếu thiếu feature,
# cell báo lỗi rõ để bạn chạy lại NB4; context rỗng không được coi là đã đạt rubric.

# %%
from feast import FeatureStore

store = FeatureStore(repo_path=str(ROOT / "app/feast_repo"))
ctx = build_context("u_001", "làm sao tối ưu chi phí hạ tầng", tool, feature_store=store)
print(json.dumps(ctx, ensure_ascii=False, indent=2, default=str))
assert "_error" not in ctx["features"], ctx["features"]
assert all(ctx["features"].get(key) and ctx["features"][key][0] is not None
           for key in ("topic_affinity", "preferred_language")), "Chạy lại NB4 để nạp profile"
assert ctx["doc_ids"], "Không có tài liệu trong context"

# %% [markdown]
# ## 6. Lưu bằng chứng
# Chụp bảng mục 3, trace mục 4 và context mục 5. `REVIEW` cần đọc số đo/trace,
# không khẳng định agentic thắng nếu bảng chưa chứng minh điều đó.

# %%
evidence.finish(
    checks={
        "all_strategies_request_16": all(r["requested"] == BUDGET for r in details),
        "agentic_beats_single_recall": split["recall"] > base["recall"],
        "agentic_beats_single_balance": split["balance"] > base["balance"],
        "inferred_filter_reduces_recall": filtered["recall"] < split["recall"],
        "reflection_recovers": reflection.n_calls == 2 and bool(reflection.doc_ids),
        "feast_and_docs_present": bool(ctx["features"]) and bool(ctx["doc_ids"]),
    },
    model=searcher.embedder.model_name, dim=searcher.embedder.dim, budget=BUDGET,
    summary=summary, per_query=details, ground_truth=queries, context=ctx,
    reflection=[{"args": c.args, "doc_ids": c.doc_ids} for c in reflection.trace],
)
