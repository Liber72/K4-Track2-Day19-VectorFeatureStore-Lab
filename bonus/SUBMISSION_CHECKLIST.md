# Checklist bonus — đối chiếu 20 điểm

Checklist mô tả bằng chứng cần có; không phải kết quả chấm điểm hay xác nhận
runtime. Các bước chạy và screenshot do học viên thực hiện.

| Tiêu chí theo `rubric.md` | Điểm | Bằng chứng đã chuẩn bị | Việc cần xác nhận |
|---|---:|---|---|
| Architecture ≥ 600 từ, có sơ đồ | 3 | `ARCHITECTURE.md`, sơ đồ Mermaid | Review sơ đồ trên GitHub |
| Ba quyết định có tradeoff explicit | 6 | Mục chunking, feature schema, freshness | Đọc và giải thích được lựa chọn |
| Vietnamese-context awareness | 2 | NFC, dấu tiếng Việt, tokenizer, code-switching | Review phù hợp dữ liệu VN |
| Loại bỏ phương án với lý do | 2 | Episodic trong feature store; BM25 global rồi post-filter | Review lý do loại bỏ |
| `HybridMemoryAgent.remember()` và `.recall()` chạy | 4 | `agent.py`; demo gọi cả hai với Qdrant/Feast thật | Chạy demo thành công |
| Demo exits 0, in 5 queries | 3 | `demo.py`; JSON/text lưu context thật | Đủ 5 context, PASS, exit code 0 |

## Thứ tự hoàn tất bằng chứng

1. Chạy tests: `.\.venv\Scripts\python.exe -m pytest tests\test_bonus.py -q`.
2. Giữ ba dịch vụ Docker sẵn sàng, chạy:
   `.\.venv\Scripts\python.exe bonus\demo.py`.
3. Kiểm tra `$LASTEXITCODE` ngay sau demo: cần `0`.
4. Mở `submission/results/bonus/bonus_demo.json`: `status` cần là `passed`,
   `results` có đúng 5 mục, mỗi mục có query/context. Kiểm tra timestamp là
   lần chạy vừa hoàn tất. File này chỉ có sau khi bạn chạy, không sinh sẵn.
5. Quan sát memory Kubernetes/autoscaling/security và profile/activity trong
   output; nếu chất lượng truy hồi chưa phù hợp, giữ output để phân tích.
6. Chụp screenshot theo [README.md](README.md), nếu muốn thêm bằng chứng.
7. Đánh dấu mục tự chạy bonus trong `submission/REFLECTION.md` sau khi đạt.
8. Commit `bonus/`, `tests/test_bonus.py`, kết quả bonus và screenshot vào
   cùng repo public; nộp URL theo quy định chung. Registry Feast không commit.

Test dùng embedding/Feast giả kiểm tra logic; demo dùng model, Postgres,
Redis và Qdrant thật mới là bằng chứng tích hợp. PASS chứng minh các điều
kiện demo kiểm tra, không chứng minh hybrid luôn thắng hoặc latency dưới 1 giây.

Nếu demo lỗi, JSON/text có `status: failed` và lỗi; không dùng chúng làm bằng
chứng thành công. Nếu bị dừng cưỡng bức trước khi ghi report, kiểm tra timestamp
để tránh dùng nhầm bằng chứng cũ.
