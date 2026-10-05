"""Compare RRF configurations using saved rankings; never change search defaults."""
from __future__ import annotations

from statistics import mean

from app.evaluation import precision_at_k
from app.fusion import fuse_rankings

DEPTHS = (10, 20, 50, 100, 200)
KEYWORD_WEIGHTS = (0.05, 0.1, 0.2, 0.35, 0.5)


def analyze_rankings(records: list[dict], *, k: int = 60, top_k: int = 10) -> dict:
    """Descriptive lab sweep, not held-out validation or an automatic deployment."""
    if not records:
        raise ValueError("Need at least one query with measured rankings")
    baseline = {
        mode: mean(precision_at_k(row[mode], set(row["relevant_doc_ids"]), top_k)
                   for row in records)
        for mode in ("keyword", "semantic")
    }
    configurations = []
    candidate_ceiling = {}
    for depth in DEPTHS:
        # Label-based upper bound for diagnosis only, never used by the retriever.
        candidate_ceiling[str(depth)] = mean(
            min(top_k, len(
                (set(row["rankings"][str(depth)]["keyword"])
                 | set(row["rankings"][str(depth)]["semantic"]))
                & set(row["relevant_doc_ids"])
            )) / top_k for row in records
        )
        for keyword_weight in KEYWORD_WEIGHTS:
            semantic_weight = 1.0 - keyword_weight
            query_results = []
            for row in records:
                relevant = set(row["relevant_doc_ids"])
                rankings = row["rankings"][str(depth)]
                fused = fuse_rankings(
                    rankings["keyword"], rankings["semantic"], k=k,
                    keyword_weight=keyword_weight, semantic_weight=semantic_weight,
                )
                hybrid_ids = [doc_id for doc_id, _ in fused[:top_k]]
                semantic_ids = row["semantic"][:top_k]
                hybrid_correct = sum(doc_id in relevant for doc_id in hybrid_ids)
                semantic_correct = sum(doc_id in relevant for doc_id in semantic_ids)
                added = [doc_id for doc_id in hybrid_ids if doc_id not in semantic_ids]
                removed = [doc_id for doc_id in semantic_ids if doc_id not in hybrid_ids]
                query_results.append({
                    "query_id": row["query_id"], "query": row["query"], "type": row["type"],
                    "semantic": semantic_correct / top_k, "hybrid": hybrid_correct / top_k,
                    "correct_delta": hybrid_correct - semantic_correct,
                    "hybrid_ids": hybrid_ids,
                    "added": [{"doc_id": doc_id, "relevant": doc_id in relevant} for doc_id in added],
                    "removed": [{"doc_id": doc_id, "relevant": doc_id in relevant} for doc_id in removed],
                })
            average = mean(row["hybrid"] for row in query_results)
            groups = sorted({row["type"] for row in query_results})
            configurations.append({
                "depth": depth, "keyword_weight": keyword_weight,
                "semantic_weight": semantic_weight, "rrf_k": k,
                "average": average,
                "by_type": {group: mean(row["hybrid"] for row in query_results if row["type"] == group)
                            for group in groups},
                "correct_delta": sum(row["correct_delta"] for row in query_results),
                "wins": sum(row["correct_delta"] > 0 for row in query_results),
                "losses": sum(row["correct_delta"] < 0 for row in query_results),
                "ties": sum(row["correct_delta"] == 0 for row in query_results),
                "beats_both": average > max(baseline.values()) + 1e-12,
                "queries": query_results,
            })
    # Stable tie-break: prefer shallower, cheaper retrieval when lab quality ties.
    configurations.sort(key=lambda item: (-item["average"], item["depth"], item["keyword_weight"]))
    return {
        "evaluation_note": "Exploratory sweep on the already-inspected lab golden set; no independent validation",
        "n_queries": len(records), "top_k": top_k, "baseline": baseline,
        "candidate_ceiling": candidate_ceiling,
        "configurations": configurations,
        "winning_configurations": sum(item["beats_both"] for item in configurations),
    }
