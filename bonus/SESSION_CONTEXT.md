# Context tiếp nối session trước

Đã đọc lịch sử hai session của đúng repo ngày 05/10/2026 và đối chiếu working
tree trước khi chuẩn bị bonus. Nội dung liên quan:

- Học viên chọn Docker trên Windows/PowerShell. Python và Jupyter chạy trong
  `.venv`; Qdrant, Redis, Postgres chạy bằng Docker Compose.
- Cấu hình đã chọn: `QDRANT_MODE=server`, `QDRANT_URL=http://localhost:6333`,
  `EMBEDDING_BACKEND=bge-m3` (1.024 chiều).
- Postgres host dùng `localhost:5433` vì cổng 5432 xung đột; cổng trong container
  vẫn 5432. Redis dùng `localhost:6379`.
- NB1 đã được học viên xác nhận hoàn thành. Session trước ghi nhận 1.000 point
  trong `lab19` và top-5 paraphrase có 5/5 tài liệu `cloud`.
- Session trước đang chuẩn bị phần bắt buộc NB2–NB4. Working tree có các sửa đổi
  ở search/API/evaluation, Feast config và notebook. Chưa kết luận NB2–NB4 đã
  chạy thành công; reflection phần bắt buộc vẫn cần output thực tế.
- Học viên yêu cầu trợ lý sửa code; học viên tự chạy kiểm thử và chụp screenshot.
  Yêu cầu đó tiếp tục áp dụng cho bonus.

Theo `README.md`, `rubric.md`, `BONUS-CHALLENGE.md`: NB1–NB4 là phần bắt buộc,
NB5–NB8 là nâng cao; bonus ở đây là **Build Your Own AI Memory**, gồm
`ARCHITECTURE.md`, `agent.py`, `demo.py`. Chưa làm NB5–NB8 trong phạm vi này.

Bonus dùng project Feast `lab19_bonus`, hai bảng `bonus_*`, registry trong
`bonus/feast_repo/`, và collection Qdrant `lab19_bonus_<dim>_<model-hash>`.
Không cần chạy xong NB4 trước để tạo dữ liệu giả lập của bonus.

Các session trước là nguồn bối cảnh, không phải bằng chứng chạy bonus.

## Cập nhật khi tiếp tục bonus

Học viên cho biết agent khác đã hoàn thiện phần bắt buộc. Nội dung NB2–NB4 ở
trên phản ánh thời điểm tiếp nhận context ban đầu. Bonus bổ sung checklist
20 điểm và lưu kết quả demo thật vào `submission/results/bonus/`; chưa chạy
demo/tests hay chụp screenshot thay học viên.
