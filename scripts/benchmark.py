"""CLI quality/latency benchmark; notebook NB3 measures the actual HTTP API."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.evaluation import percentile, precision_at_k, save_result
from app.search import Searcher


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=2, help="Lượt đo latency trên 50 queries (mặc định 2)")
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps phải >= 1")
    with (ROOT / "data/golden_set.jsonl").open(encoding="utf-8") as f:
        golden = [json.loads(line) for line in f if line.strip()]
    assert len(golden) == 50
    searcher = Searcher.from_corpus(ROOT / "data/corpus_vn.jsonl")
    print(f"Model: {searcher.embedder.model_name}; docs: {searcher.size}; reused: {searcher.reused_index}")
    modes = ("keyword", "semantic", "hybrid")
    quality = {mode: [] for mode in modes}
    slices = {}
    for i, q in enumerate(golden, start=1):
        relevant = set(q["relevant_doc_ids"])
        group = slices.setdefault(q["mode_hint"], {mode: [] for mode in modes})
        for mode in modes:
            ids = [hit.doc_id for hit in searcher.search(q["query"], mode=mode, top_k=10, rrf_k=60)]
            score = precision_at_k(ids, relevant)
            quality[mode].append(score)
            group[mode].append(score)
        if i % 10 == 0:
            print(f"Quality: {i}/{len(golden)}")
    averages = {mode: statistics.mean(scores) for mode, scores in quality.items()}
    print("\nPrecision@10")
    for mode, score in averages.items():
        print(f"  {mode:10} {score:.1%}")
    print(f"\n{'type':12} {'keyword':>10} {'semantic':>10} {'hybrid':>10}")
    slice_summary = {}
    for query_type in ("exact", "paraphrase", "mixed"):
        values = {mode: statistics.mean(slices[query_type][mode]) for mode in modes}
        slice_summary[query_type] = values
        print(f"{query_type:12} {values['keyword']:>9.1%} {values['semantic']:>9.1%} {values['hybrid']:>9.1%}")

    latency = {}
    for mode in modes:
        for q in golden[:10]:
            searcher.search(q["query"], mode=mode)
        values = []
        for _ in range(args.reps):
            for q in golden:
                start = time.perf_counter()
                searcher.search(q["query"], mode=mode)
                values.append((time.perf_counter() - start) * 1000)
        latency[mode] = {"n_calls": len(values), "p50": percentile(values, 0.50),
                         "p95": percentile(values, 0.95), "p99": percentile(values, 0.99)}
    print("\nDirect Searcher latency (ms), sau warm-up — không gồm HTTP")
    print(f"{'mode':10} {'P50':>10} {'P95':>10} {'P99':>10}")
    for mode, values in latency.items():
        print(f"{mode:10} {values['p50']:>10.2f} {values['p95']:>10.2f} {values['p99']:>10.2f}")
    passed = averages["hybrid"] > max(averages["keyword"], averages["semantic"])
    print("\nPASS — hybrid thắng cả hai mode" if passed
          else "\nChưa đạt mục tiêu hybrid thắng cả hai; đối chiếu model, corpus và thứ hạng từng query.")
    print("Saved:", save_result("benchmark_cli.json", {
        "n_queries": len(golden), "average": averages, "by_type": slice_summary,
        "direct_latency": latency, "hybrid_beats_both": passed,
    }))
    searcher.client.close()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
