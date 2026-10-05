"""Retrieval as a tool + a zero-key planning agent (NB6).

Deck reference: "Retrieval Nhu Mot Tool" and "Ghep Ngu Canh".

The planner here is deliberately RULE-BASED, not an LLM call. Two reasons:
  1. the lab must run with no API key and no GPU;
  2. the lesson is about *retrieval strategy* -- decompose, filter, reflect,
     retry -- and that lesson is clearer when the planner is inspectable.

Swap `RuleBasedPlanner` for an LLM planner in the bonus challenge; the Agent
below only requires `.plan(question) -> list[ToolArgs]`.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from qdrant_client import models

from app.filters import FilteredIndex

# The tool definition the agent sees. Provider-neutral JSON Schema: the same
# shape drops into OpenAI `tools`, Anthropic `input_schema`, or an MCP server.
SEARCH_TOOL: dict[str, Any] = {
    "name": "search_docs",
    # This description IS the retrieval prompt. It is the only text a planner
    # gets when deciding whether to call the tool at all.
    "description": (
        "Search the internal Vietnamese technical documentation corpus. "
        "Covers cloud, ai_ml, security, database, networking, devops, "
        "mobile, frontend, backend and data_eng. "
        "Ask ONE question per call. "
        "Returns ranked document chunks with doc_id, title and score."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "One single-intent question."},
            "topic": {
                "type": "string",
                "description": "Restrict to one topic cluster.",
                # enum, not free text: a planner cannot invent a topic that
                # does not exist, so the filter can never silently match zero.
                "enum": ["cloud", "ai_ml", "security", "database", "networking",
                         "devops", "mobile", "frontend", "backend", "data_eng"],
            },
            "since_year": {"type": "integer", "description": "Only docs published on/after."},
            "top_k": {"type": "integer", "default": 8, "maximum": 25},
        },
        "required": ["query"],
    },
}

TOPIC_HINTS = {
    "cloud":      ["cloud", "đám mây", "kubernetes", "serverless", "container", "aws", "mở rộng"],
    "ai_ml":      ["ai", "mô hình", "embedding", "llm", "học sâu", "huấn luyện", "tinh chỉnh"],
    "security":   ["bảo mật", "security", "oauth", "jwt", "mã hoá", "xác thực", "nhạy cảm"],
    "database":   ["cơ sở dữ liệu", "database", "sql", "truy vấn", "chỉ mục", "b-tree", "bản sao"],
    "networking": ["mạng", "network", "tcp", "dns", "cân bằng tải", "độ trễ", "region"],
    "devops":     ["devops", "ci/cd", "pipeline", "triển khai", "docker", "blue-green", "rollback"],
    "mobile":     ["mobile", "android", "ios", "di động", "apk", "ipa", "mất kết nối"],
    "frontend":   ["frontend", "giao diện", "react", "css", "ui", "lcp", "web vitals", "component"],
    "backend":    ["backend", "api", "microservice", "server", "idempotent", "circuit breaker"],
    "data_eng":   ["kafka", "flink", "streaming", "etl", "schema evolution", "downstream", "sự kiện"],
}

# Connectors that signal "this question has more than one intent".
SPLIT_RE = re.compile(r"\s+(?:và|hoặc|so với|vs\.?|cũng như)\s+|[;?]\s*", re.IGNORECASE)


@dataclass
class ToolArgs:
    query: str
    topic: str | None = None
    since_year: int | None = None
    top_k: int = 8

    def as_dict(self) -> dict:
        d = {"query": self.query, "top_k": self.top_k}
        if self.topic:
            d["topic"] = self.topic
        if self.since_year:
            d["since_year"] = self.since_year
        return d


@dataclass
class ToolCall:
    args: dict
    doc_ids: list[str]
    latency_ms: float


@dataclass
class AgentResult:
    question: str
    doc_ids: list[str]
    trace: list[ToolCall] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return len(self.trace)

    @property
    def latency_ms(self) -> float:
        return sum(c.latency_ms for c in self.trace)


class RetrievalTool:
    """The executable behind SEARCH_TOOL."""

    def __init__(self, index: FilteredIndex) -> None:
        self.index = index

    def __call__(self, args: ToolArgs) -> ToolCall:
        must = []
        if args.topic:
            must.append(models.FieldCondition(
                key="topic", match=models.MatchValue(value=args.topic)))
        if args.since_year:
            must.append(models.FieldCondition(
                key="published_ts", range=models.Range(gte=args.since_year * 10000)))
        qf = models.Filter(must=must) if must else None

        t0 = time.perf_counter()
        if qf is None:
            qv = self.index.embed(args.query)
            pts = self.index.client.query_points(
                collection_name=self.index.collection, query=qv.tolist(), limit=args.top_k
            ).points
            ids = [p.payload["doc_id"] for p in pts]
        else:
            ids = self.index.filtered_ann(args.query, qf, k=args.top_k).doc_ids
        return ToolCall(args.as_dict(), ids, (time.perf_counter() - t0) * 1000)


class Planner(Protocol):
    def plan(self, question: str) -> list[ToolArgs]: ...


class SingleShotPlanner:
    """The baseline: embed the whole question once, retrieve once."""

    def __init__(self, budget: int = 16) -> None:
        if budget < 1:
            raise ValueError("budget must be positive")
        self.budget = budget

    def plan(self, question: str) -> list[ToolArgs]:
        return [ToolArgs(query=question, top_k=self.budget)]


class RuleBasedPlanner:
    """Decompose -> (optionally) infer filters -> same budget, split per part.

    `use_filters` is exposed because inferring a filter is NOT free: a topic
    guessed from keywords narrows the search and can exclude relevant documents
    that live in a neighbouring cluster. NB6 measures both settings so the
    trade-off is visible rather than assumed.
    """

    def __init__(self, budget: int = 16, use_filters: bool = True) -> None:
        if budget < 1:
            raise ValueError("budget must be positive")
        self.budget = budget
        self.use_filters = use_filters

    @staticmethod
    def detect_topic(text: str) -> str | None:
        low = text.lower()
        best, best_n = None, 0
        for topic, hints in TOPIC_HINTS.items():
            n = sum(1 for h in hints if h in low)
            if n > best_n:
                best, best_n = topic, n
        return best

    @staticmethod
    def detect_year(text: str) -> int | None:
        m = re.search(r"\b(20\d{2})\b", text)
        return int(m.group(1)) if m else None

    def plan(self, question: str) -> list[ToolArgs]:
        parts = [p.strip() for p in SPLIT_RE.split(question) if p and len(p.strip()) > 3]
        if len(parts) < 2:
            parts = [question]
        # Preserve the whole question when there are more parts than slots.
        if len(parts) > self.budget:
            parts = parts[:self.budget - 1] + [" và ".join(parts[self.budget - 1:])]
        # Allocate the remainder too: a 3-part question gets 6 + 5 + 5, not 15.
        # This is requested capacity; duplicates/filters can yield fewer unique docs.
        per, remainder = divmod(self.budget, len(parts))
        return [
            ToolArgs(
                query=p,
                topic=self.detect_topic(p) if self.use_filters else None,
                since_year=self.detect_year(p) if self.use_filters else None,
                top_k=per + int(i < remainder),
            )
            for i, p in enumerate(parts)
        ]


class Agent:
    """plan -> call -> reflect -> (maybe) retry with the filters relaxed."""

    def __init__(self, tool: RetrievalTool, planner: Planner, min_evidence: int = 4) -> None:
        self.tool, self.planner, self.min_evidence = tool, planner, min_evidence

    def answer(self, question: str) -> AgentResult:
        res = AgentResult(question=question, doc_ids=[])
        seen: list[str] = []
        for args in self.planner.plan(question):
            call = self.tool(args)
            res.trace.append(call)
            # Reflection: a filter that returns almost nothing is a bad filter,
            # not an empty corpus. Retry once without it.
            if len(call.doc_ids) < self.min_evidence and (args.topic or args.since_year):
                relaxed = ToolArgs(query=args.query, top_k=args.top_k)
                call = self.tool(relaxed)
                res.trace.append(call)
            for d in call.doc_ids:
                if d not in seen:
                    seen.append(d)
        res.doc_ids = seen
        return res


def build_context(user_id: str, question: str, tool: RetrievalTool,
                  feature_store: Any | None = None, top_k: int = 8) -> dict:
    """Where the two halves of Day 19 meet.

    Feature store answers "who is this user" (personalisation).
    Vector store answers "what is relevant" (grounding).
    Degrades gracefully when Feast has not been applied yet, so NB6 runs
    before NB4 if a student jumps around.
    """
    features: dict[str, Any] = {}
    if feature_store is not None:
        try:
            features = feature_store.get_online_features(
                features=["user_profile_features:topic_affinity",
                          "user_profile_features:preferred_language"],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
        except Exception as exc:                      # noqa: BLE001
            features = {"_error": str(exc)}

    affinity = (features.get("topic_affinity") or [None])[0]
    args = ToolArgs(query=question, topic=affinity if affinity in TOPIC_HINTS else None,
                    top_k=top_k)
    call = tool(args)
    return {
        "user_id": user_id,
        "question": question,
        "features": features,
        "affinity_used": args.topic,
        "doc_ids": call.doc_ids,
        "tool_args": call.args,
    }
