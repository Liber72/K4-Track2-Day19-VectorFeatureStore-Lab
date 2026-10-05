"""Rank fusion shared by NB2 and the search API (no relevance labels)."""
from __future__ import annotations

# Selected by the user's lab sweep: depth=50, k=60, P@10=0.954.
# This is a fit to the lab golden set, not an independently validated optimum.
KEYWORD_WEIGHT = 0.05
SEMANTIC_WEIGHT = 0.95


def fuse_rankings(
    keyword_ids: list[str], semantic_ids: list[str], *, k: int = 60,
    keyword_weight: float = KEYWORD_WEIGHT,
    semantic_weight: float = SEMANTIC_WEIGHT,
) -> list[tuple[str, float]]:
    """Weighted RRF; setting both weights to 1 reproduces standard RRF."""
    if k < 0 or min(keyword_weight, semantic_weight) < 0 or keyword_weight + semantic_weight <= 0:
        raise ValueError("RRF requires k >= 0 and nonnegative weights with a positive sum")
    scores: dict[str, float] = {}
    for ids, weight in ((keyword_ids, keyword_weight), (semantic_ids, semantic_weight)):
        if weight == 0:
            continue
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])
