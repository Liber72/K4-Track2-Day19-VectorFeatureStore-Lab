"""Generate reflection from measured results and check the core submission."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = "<!-- Generated from notebook results by prepare_submission.py -->"
MODES = ("keyword", "semantic", "hybrid")


def read_result(root: Path, filename: str) -> dict | None:
    path = root / "submission/results" / filename
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_reflection(root: Path = ROOT, name: str | None = None) -> Path | None:
    quality = read_result(root, "nb2_quality.json")
    latency = read_result(root, "nb3_latency.json")
    features = read_result(root, "nb4_features.json")
    if any(result is None for result in (quality, latency, features)):
        print("Reflection đang chờ nb2_quality.json, nb3_latency.json và nb4_features.json.")
        return None
    if quality["n_queries"] != 50:
        raise ValueError("Reflection cần kết quả đủ 50 golden queries")
    name = name or os.getenv("LAB_STUDENT_NAME", "HoangThaiDat")
    averages = quality["average"]
    statements = []
    for query_type in ("exact", "paraphrase", "mixed"):
        scores = quality["by_type"][query_type]
        best = max(scores[mode] for mode in MODES)
        winners = "/".join(mode for mode in MODES if math.isclose(scores[mode], best, abs_tol=1e-12))
        statements.append(f"Nhóm {query_type}: {winners} cao nhất ({best:.1%}).")
    conclusion = ("Hybrid thắng cả hai mode về trung bình." if
                  averages["hybrid"] > max(averages["keyword"], averages["semantic"])
                  else "Hybrid chưa thắng cả hai mode về trung bình.")
    answer = (
        f"Trên 50 golden queries, Precision@10 trung bình: keyword {averages['keyword']:.1%}, "
        f"semantic {averages['semantic']:.1%}, hybrid {averages['hybrid']:.1%}. "
        + " ".join(statements) + " " + conclusion + "\n\n"
        "BM25 dựa vào từ khóa, vector dựa vào ý nghĩa; RRF cộng điểm theo thứ hạng "
        "của hai retriever. Chất lượng kết hợp phụ thuộc chất lượng từng danh sách ứng viên. "
        "Tôi dùng riêng BM25 khi cần khớp thuật ngữ hoặc mã chính xác; dùng riêng vector "
        "khi truy vấn diễn đạt lại và semantic đã đủ tốt. Hybrid phù hợp khi cần cả "
        "hai tín hiệu, nhưng có thêm chi phí xử lý và cần đánh giá trên dữ liệu thật."
    )
    body = (
        "# Reflection — Lab 19\n\n"
        f"**Tên:** {name}\n"
        "**Cohort:** A20-K4\n"
        "**Path đã chạy:** Docker\n\n"
        f"{MARKER}\n\n"
        "## Câu trả lời\n\n"
        f"{answer}\n"
    )
    assert len(answer.split()) <= 200
    path = root / "submission/REFLECTION.md"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if MARKER not in existing and "_Answer here._" not in existing:
            print(f"Giữ reflection bạn đã tự viết: {path}")
            return path
        if "## Bonus challenge" in existing:
            body += "\n" + existing[existing.index("## Bonus challenge"):]
    path.write_text(body, encoding="utf-8")
    print(f"Reflection đã tạo từ số đo thật ({len(answer.split())} từ): {path}")
    return path


def notebook_complete(path: Path) -> bool:
    if not path.exists():
        return False
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"]
             if cell["cell_type"] == "code" and "".join(cell.get("source", [])).strip()]
    return bool(cells) and all(
        cell.get("execution_count") is not None
        and not any(output.get("output_type") == "error" for output in cell.get("outputs", []))
        for cell in cells
    )


def check_submission(root: Path = ROOT) -> bool:
    checks = {}
    notebooks = [
        "01_embeddings_index", "02_hybrid_search_rrf",
        "03_search_api_benchmark", "04_feast_feature_store",
    ]
    for notebook in notebooks:
        checks[f"Notebook {notebook} đã chạy và lưu output"] = notebook_complete(
            root / "notebooks" / (notebook + ".ipynb"),
        )
    quality = read_result(root, "nb2_quality.json")
    latency = read_result(root, "nb3_latency.json")
    features = read_result(root, "nb4_features.json")
    checks["NB2 đủ 50 queries và hai bảng chất lượng"] = bool(
        quality and quality.get("n_queries") == 50
        and len(quality.get("queries", [])) == 50
        and set(quality.get("by_type", {})) == {"exact", "paraphrase", "mixed"}
    )
    checks["NB2 hybrid trung bình > keyword và semantic"] = bool(
        quality and quality["average"]["hybrid"] >
        max(quality["average"]["keyword"], quality["average"]["semantic"])
    )
    checks["NB3 đo đủ 100 calls mỗi mode"] = bool(
        latency and set(latency.get("latencies", {})) == set(MODES)
        and all(latency["latencies"][mode]["n_calls"] == 100 for mode in MODES)
    )
    checks["NB3 hybrid P99 server-side < 50ms"] = bool(
        latency and latency["latencies"]["hybrid"]["p99_server"] < 50
    )
    checks["NB4 ba views, materialize và PIT 3 dòng"] = bool(
        features and len(features.get("registered_views", [])) == 3
        and features.get("materialize_passed") and features.get("pit_passed")
        and features.get("pit_rows") == 3
    )
    checks["NB4 online lookup P99 < 10ms"] = bool(
        features and features["lookup_latency"]["p99"] < 10
    )
    reflection = root / "submission/REFLECTION.md"
    checks["Reflection đã điền"] = bool(
        reflection.exists() and "_Answer here._" not in reflection.read_text(encoding="utf-8")
    )
    screenshots = (
        "nb1_indexed_1000", "nb1_top5", "nb2_precision", "nb2_slices",
        "nb3_response", "nb3_latency", "nb4_views", "nb4_materialize",
        "nb4_online", "nb4_pit",
    )
    for stem in screenshots:
        checks[f"Ảnh {stem}"] = any(
            (root / "submission/screenshots" / (stem + extension)).is_file()
            and (root / "submission/screenshots" / (stem + extension)).stat().st_size > 0
            for extension in (".png", ".jpg", ".jpeg", ".webp")
        )
    print("\nCore submission checklist")
    for label, passed in checks.items():
        print(f"[{'OK' if passed else 'CÒN THIẾU'}] {label}")
    complete = all(checks.values())
    print("\nBộ bài core đủ tiêu chí kiểm tra." if complete
          else "\nBộ bài còn thiếu bằng chứng hoặc còn số đo chưa đạt; xem các dòng trên.")
    print("Sau khi hoàn tất: push repo public và nộp URL vào VinUni LMS Day 19.")
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="Tên học viên dùng trong reflection")
    parser.add_argument("--check-only", action="store_true", help="Chỉ kiểm tra, giữ nguyên reflection")
    args = parser.parse_args()
    if not args.check_only:
        write_reflection(name=args.name)
    return 0 if check_submission() else 1


if __name__ == "__main__":
    sys.exit(main())
