"""Metrics and saved evidence shared by the core lab notebooks."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "submission" / "results"


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Relevant documents in the first k positions, divided by k."""
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(doc_id in relevant_ids for doc_id in retrieved_ids[:k]) / k


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile; p is between 0 and 1."""
    if not values or not 0 <= p <= 1:
        raise ValueError("Need nonempty values and 0 <= p <= 1")
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(len(ordered) * p) - 1)])


def save_result(filename: str, result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    data = {"measured_at": datetime.now(timezone.utc).isoformat(), **result}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
