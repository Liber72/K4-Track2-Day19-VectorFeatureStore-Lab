# Chạy và kiểm chứng bonus

Đề bài: [BONUS-CHALLENGE.md](../BONUS-CHALLENGE.md).
Thiết kế: [ARCHITECTURE.md](ARCHITECTURE.md).
Context kế thừa: [SESSION_CONTEXT.md](SESSION_CONTEXT.md).
Đối chiếu 20 điểm: [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md).

Code và tài liệu đã được chuẩn bị. **Demo, pytest và screenshot để học viên
tự thực hiện; chưa có kết quả runtime bonus được xác nhận.** Không cần API key
hay LLM thật. Không cần hoàn tất NB4 để chạy bonus.

## 1. Chuẩn bị môi trường hiện có

Mở PowerShell tại repo:

```powershell
Set-Location "D:\AI_Thucchien\K4-Track2-Day19-VectorFeatureStore-Lab"
docker compose up -d
docker compose ps
```

Ba container cần healthy. Giữ `.env` như đã chọn ở session trước:

```dotenv
QDRANT_MODE=server
QDRANT_URL=http://localhost:6333
EMBEDDING_BACKEND=bge-m3
```

Bonus đọc `.env` cho Qdrant/model. Feast đọc riêng
`bonus/feast_repo/feature_store.yaml`: Redis 6379, Postgres 5433, project
`lab19_bonus`. Các biến `FEAST_*` trong `.env` không thay cấu hình YAML này.
Nếu đổi cổng/password của Docker thì cập nhật YAML bonus tương ứng.

Môi trường từ session trước đã có thư viện. Nếu dựng máy sạch, cài theo
`requirements.txt` và `requirements-full.txt` trước khi tiếp tục. bge-m3 dùng
lại model đã tải từ NB1; lần đầu chưa có cache sẽ tải model.

## 2. Chạy demo 5 queries

```powershell
.\.venv\Scripts\python.exe bonus\demo.py
```

Lệnh này tự tạo bảng bonus nếu thiếu, đăng ký hai feature view, ghi profile
giả lập vào Postgres và push vào Redis. Sau đó tạo 5 ghi chú của `u_001`, một
ghi chú riêng của `u_002`, rồi in **5 context** từ `recall()`.

| Query | Mục cần quan sát |
|---|---|
| Tôi đã đọc gì về Kubernetes? | Memory liên quan Kubernetes trong top-3 |
| Recommend đọc gì tiếp | `topic_affinity=cloud`, tốc độ 240, ngôn ngữ `mix` |
| Tôi đang quan tâm gì gần đây? | `queries_last_hour=3`, topic của các câu đã hỏi |
| Tài liệu về tự động mở rộng hạ tầng? | Ghi chú autoscaling được truy hồi theo nghĩa |
| Cho tôi summary cloud security | Memory bảo mật cùng profile của `u_001` |

Mỗi context hiển thị profile, activity và tối đa 3 memory có point ID. Đây là
ngữ cảnh chuẩn bị cho LLM, chưa phải câu trả lời hoặc recommendation cuối cùng.
Thứ hạng cần quan sát thực tế; script không gán sẵn kết quả và không tuyên bố
hybrid luôn thắng vector. Counts lần lượt 1–5, bao gồm query hiện tại.

Dòng cuối khi các kiểm tra trong script đạt:

```text
PASS: 5 contexts; profile + fresh activity; u_002 marker absent from u_001.
```

Kiểm tra exit code ngay sau lệnh demo:

```powershell
$LASTEXITCODE
```

Kết quả cần là `0`. Script lỗi sẽ dừng với traceback; không thay Feast bằng
dict giả hoặc in PASS khi dịch vụ không hoạt động. Mỗi lần chạy demo khởi động
lại bộ đếm in-process; lưu lại cùng memory dùng cùng ID, nhưng snapshot feature
được append thêm vào Postgres. Đây là dữ liệu thử, chưa có cleanup tự động.

Demo tự lưu bằng chứng thực tế vào:

- `submission/results/bonus/bonus_demo.json`: trạng thái, timestamp UTC, model,
  collection, 5 query/context và các kiểm tra đã đạt.
- `submission/results/bonus/bonus_demo.txt`: bản text thuận tiện đọc/nộp.

Chỉ dùng report có `status: passed`, đúng 5 kết quả và timestamp của lần chạy
thành công. Nếu lỗi, report được cập nhật `failed`; không có kết quả giả lập
được sinh sẵn. Bắt đầu từ repo sạch chưa có hai file này là bình thường.
Có thể đổi thư mục lưu bằng tùy chọn `--output-dir`.

## 3. Kiểm thử logic bổ sung

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_bonus.py -q
```

Tests dùng Qdrant in-memory thật, embedding xác định và Feast giả để kiểm tra
filter cả hai nhánh, pagination, idempotency/chunk overlap, RRF rank 1-based,
window 1h/24h, thiếu profile và stale profile. Không tải model hay kết nối
Docker. Chúng không thay cho bước demo kiểm chứng Redis/Postgres/model thật.

Nếu muốn kiểm tra toàn repo sau khi hoàn tất phần bắt buộc:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 4. Bằng chứng và nộp bài

Bonus không quy định screenshot bắt buộc riêng. Nếu muốn lưu bằng chứng,
chụp bằng tay và đặt trong `submission/screenshots/`:

- `bonus_profile_activity.png`: Query 2–3 thấy profile và count mới.
- `bonus_memory_demo.png`: Query 4–5 cùng dòng PASS và exit code 0.
- `bonus_tests.png`: kết quả pytest phần bonus.

Có thể mở [Qdrant Dashboard](http://localhost:6333/dashboard) để thấy collection
tên in ở đầu demo. `lab19` của NB1 giữ nguyên 1.000 point; bonus nằm ở collection
khác. Snapshot Postgres chỉ ở `bonus_user_profile`, `bonus_recent_activity`.

Trong `submission/REFLECTION.md`, sau khi chạy đạt, đánh dấu mục tự chạy bonus;
phần reflection NB2 vẫn phải dựa trên output bắt buộc của bạn. Thêm `bonus/`
vào cùng repo public và nộp URL theo rubric. Không cần PR.

## Tài liệu API đã đối chiếu

- [Feast PushSource / Push API](https://docs.feast.dev/reference/data-sources/push):
  processor tính aggregate rồi push; POC tự ghi snapshot vào batch source.
- [Qdrant filtering](https://qdrant.tech/documentation/search/filtering/):
  điều kiện `user_id` áp dụng khi scroll và khi vector query.
