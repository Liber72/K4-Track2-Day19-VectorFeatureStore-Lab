# Reflection — Lab 19

**Tên:** HoangThaiDat
**Cohort:** A20-K4
**Path đã chạy:** Docker

<!-- Generated from notebook results by prepare_submission.py -->

## Câu trả lời

Trên 50 golden queries, Precision@10 trung bình: keyword 80.4%, semantic 95.2%, hybrid 90.0%. Nhóm exact: keyword/hybrid cao nhất (100.0%). Nhóm paraphrase: semantic cao nhất (86.7%). Nhóm mixed: keyword/hybrid cao nhất (100.0%). Hybrid chưa thắng cả hai mode về trung bình.

BM25 dựa vào từ khóa, vector dựa vào ý nghĩa; RRF cộng điểm theo thứ hạng của hai retriever. Chất lượng kết hợp phụ thuộc chất lượng từng danh sách ứng viên. Tôi dùng riêng BM25 khi cần khớp thuật ngữ hoặc mã chính xác; dùng riêng vector khi truy vấn diễn đạt lại và semantic đã đủ tốt. Hybrid phù hợp khi cần cả hai tín hiệu, nhưng có thêm chi phí xử lý và cần đánh giá trên dữ liệu thật.

## Bonus challenge

- [x] Đã chuẩn bị bonus: [kiến trúc](../bonus/ARCHITECTURE.md),
  `HybridMemoryAgent` và demo 5 queries; [hướng dẫn chạy](../bonus/README.md).
- [ ] Đã tự chạy `python bonus/demo.py`, kiểm chứng 5 context và lưu bằng chứng.
- [ ] Pair work với: _<tên đồng đội nếu có>_
