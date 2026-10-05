"""Run from the repo root: python bonus/demo.py. Prints exactly five contexts."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bonus.agent import HybridMemoryAgent
from bonus.features import prepare_demo_features

MEMORIES = [
    "Tôi đã đọc hướng dẫn Kubernetes: Pod, Deployment, Service và vận hành cluster. "
    "Ghi chú: Kubernetes quản lý các container và triển khai ứng dụng trên hạ tầng cloud.",
    "Tài liệu autoscaling mô tả tự động mở rộng hạ tầng theo lưu lượng người dùng. "
    "Kubernetes Horizontal Pod Autoscaler tăng giảm số replica theo CPU; cluster autoscaler bổ sung node.",
    "Tôi đã đọc cloud security: bảo mật đám mây, IAM least privilege, MFA, "
    "mã hóa dữ liệu và audit log. Tôi muốn ví dụ tiếng Việt kèm thuật ngữ English.",
    "Ghi chú về tối ưu chi phí cloud: so sánh compute, lưu trữ và networking; "
    "đo tải thực tế trước khi chọn cấu hình. Tôi thích bài thực hành ngắn có ví dụ.",
    "Tôi đã đọc về vector embedding, BM25, Reciprocal Rank Fusion và Feast Feature Store "
    "để kết hợp truy hồi tài liệu với profile trong một trợ lý AI cá nhân.",
]
QUERIES = [
    "Tôi đã đọc gì về Kubernetes?",
    "Recommend đọc gì tiếp",
    "Tôi đang quan tâm gì gần đây?",
    "Tài liệu về tự động mở rộng hạ tầng?",
    "Cho tôi summary cloud security",
]


def run_demo(agent) -> list[dict]:
    """Exercise both public methods; fail if any context loses required data."""
    for text in MEMORIES:
        agent.remember(text)
    agent.remember("PRIVATE_U002: Ghi chú Kubernetes riêng của người dùng thứ hai.", "u_002")
    results = []
    for index, query in enumerate(QUERIES, start=1):
        context = agent.recall(query)
        assert "PRIVATE_U002" not in context, "Cross-user memory leak"
        assert f"queries_last_hour={index};" in context, "Activity push/lookup mismatch"
        assert "topic_affinity=cloud;" in context, "Profile push/lookup mismatch"
        assert "reading_speed_wpm=240;" in context, "Reading speed missing"
        assert "preferred_language=mix;" in context, "Language missing"
        assert "(chưa có memory)" not in context, "Episodic memory missing"
        results.append({"query": query, "context": context})
        print(f"\n{'=' * 24} Query {index}/5 {'=' * 24}\n{context}")
    assert len(results) == 5
    return results


def save_evidence(output_dir: Path, report: dict) -> None:
    """Persist real contexts; a failed run replaces an earlier success report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bonus_demo.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"Status: {report['status']}", f"Started UTC: {report['started_at_utc']}",
             f"Finished UTC: {report['finished_at_utc']}"]
    for index, result in enumerate(report["results"], start=1):
        lines.append(f"\nQuery {index}/5\n{result['context']}")
    if "error" in report:
        lines.append(f"\nERROR: {report['error']}")
    (output_dir / "bonus_demo.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(output_dir: Path | None = None) -> None:
    output_dir = output_dir or ROOT / "submission/results/bonus"
    report = {"status": "failed", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "results": []}
    agent = None
    try:
        store = prepare_demo_features()
        agent = HybridMemoryAgent(feature_store=store)
        report.update({"feast_project": store.project, "collection": agent.collection,
                       "embedding_model": agent.embedder.model_name,
                       "embedding_dim": agent.embedder.dim})
        print(f"Feast project: {store.project}; bonus collection: {agent.collection}")
        report["results"] = run_demo(agent)
        report["checks"] = {"five_contexts": True, "remember_and_recall": True,
                            "profile_present": True, "fresh_activity": True,
                            "foreign_marker_absent": True}
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            save_evidence(output_dir, report)
        finally:
            if agent is not None:
                agent.client.close()
    print("\nPASS: 5 contexts; profile + fresh activity; u_002 marker absent from u_001.")
    print(f"Evidence saved: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="Evidence directory (default: submission/results/bonus)")
    main(parser.parse_args().output_dir)
