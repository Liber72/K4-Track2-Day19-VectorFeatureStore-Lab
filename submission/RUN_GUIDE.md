# Chạy phần bắt buộc Day 19 bằng Docker

Code NB1–NB4 đã được chuẩn bị. Bạn chạy notebook, quan sát kết quả, giữ output
và chụp ảnh. Phần nâng cao NB5–NB8 và bonus là tự chọn, ngoài lộ trình core này.

Tiếp tục NB5–NB8 theo [hướng dẫn nâng cao](ADVANCED_RUN_GUIDE.md):
bốn notebook `.ipynb`, bảng cần quan sát, kết quả JSON và ảnh cần lưu.

## 1. Môi trường

Trong PowerShell tại thư mục dự án:

```powershell
docker compose up -d
docker compose ps
.\.venv\Scripts\python.exe -X utf8 scripts\verify_docker.py
.\.venv\Scripts\python.exe -m jupyterlab --notebook-dir=notebooks
```

Ba dịch vụ cần healthy: Qdrant :6333, Redis :6379, Postgres :5433.
Chọn **Python (Lab 19)** trong từng notebook. Chạy cell theo thứ tự, dùng
Shift+Enter và đợi cell hoàn tất. Lỗi ở cell nào thì giữ nguyên traceback để xử lý.

NB1 đã được chạy và lưu index 1.000 vector. Tiếp tục NB2 → NB3 → NB4.
Sau mỗi notebook: Ctrl+S, chụp ảnh các output, rồi **Kernel → Shut Down Kernel**
để giải phóng model trong RAM trước khi mở notebook tiếp theo.

### Dựng lại trên máy khác

```powershell
uv venv --python 3.10 .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt -r requirements-full.txt
Copy-Item .env.docker.example .env
.\.venv\Scripts\python.exe -X utf8 scripts\seed_corpus.py
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name lab19 --display-name "Python (Lab 19)"
docker compose up -d
```

Chạy NB1 trước để tạo collection. Môi trường này dùng bge-m3 trong .env;
Feast dùng Redis/Postgres theo app/feast_repo/feature_store.yaml.

## 2. NB2 — Hybrid Search

Mở `02_hybrid_search_rrf.ipynb`, chạy toàn bộ cell theo thứ tự:

| Mục | Việc thực hiện | Bằng chứng |
|---|---|---|
| 1 | Dựng BM25, dùng lại vector NB1 | 1.000 docs, bge-m3, collection lab19 |
| 2–3 | Keyword, semantic và RRF | Top-3 của một câu diễn đạt lại |
| 4 | Đánh giá 50 golden queries | Bảng Precision@10 của ba mode |
| 5 | So sánh theo loại query | Bảng exact/paraphrase/mixed |

RRF: rank bắt đầu từ 1, k=60, lấy top-50 mỗi retriever rồi chọn top-10.
Mục tiêu rubric: hybrid trung bình cao hơn cả keyword và semantic.
Nếu chưa đạt, giữ số đo và xem từng query; model tốt hơn không bảo đảm RRF luôn thắng.

JSON tự lưu vào `submission/results/nb2_quality.json`.
Chụp `nb2_precision.png` và `nb2_slices.png`.

## 3. NB3 — API và benchmark

Mở `03_search_api_benchmark.ipynb`:

1. Khởi động API, đợi /healthz: ready=true, n_docs=1000, reused_index=true.
2. Xem response /search: query, mode, latency_ms và hits.
3. Warm-up 10 queries/mode.
4. Đo 100 requests/mode và xem P50/P95/P99.
5. Chạy cleanup để dừng API do notebook khởi động.

Mục tiêu: **hybrid P99 server-side < 50ms**. Cột wall-clock có thêm thời gian HTTP.
API embed câu hỏi cho mỗi request. Lần tải model đầu không nằm trong phép đo sau warm-up.
Nếu chưa đạt, xem tải máy và số đo thực tế, đo lại sau warm-up; giữ output.

JSON tự lưu vào `submission/results/nb3_latency.json`.
Chụp `nb3_response.png` và `nb3_latency.png`.
Log khi startup lỗi: `submission/results/api_server.log`.

## 4. NB4 — Feast

Mở `04_feast_feature_store.ipynb`:

| Mục | Việc thực hiện | Bằng chứng |
|---|---|---|
| 1 | Nạp ba bảng Postgres | user_profile: 200, item_popularity: 1.000, query_velocity: 100 rows |
| 2 | Feast apply, list views | Đủ ba feature views |
| 3 | Materialize và materialize-incremental | Log lệnh thành công |
| 4 | Online lookup cho u_001 và một doc | Tám feature có giá trị |
| 5 | Warm-up, đo 100 lookups | Bảng P50/P95/P99; mục tiêu P99 < 10ms |
| 6 | PIT join | Ba dòng; u_001 nhận tốc độ cũ 167, online nhận tốc độ mới 187 |
| 7 | Tạo reflection | REFLECTION.md điền từ các JSON đo thật |

Mục 1 thay dữ liệu demo trong ba bảng của database feast_offline khi chạy lại.
Dữ liệu PIT có hai phiên bản profile để kiểm chứng việc tránh dùng feature tương lai.
Mục 3 nạp đầy đủ để notebook chạy lại được, rồi xác minh lệnh incremental;
lần incremental ngay sau đó có thể không có hàng mới.

JSON tự lưu vào `submission/results/nb4_features.json`.
Chụp `nb4_views.png`, `nb4_materialize.png`, `nb4_online.png`, `nb4_pit.png`.

## 5. Ảnh và kiểm tra cuối

Các ảnh cần đặt vào `submission/screenshots/`:

- NB1: `nb1_indexed_1000.png` và `nb1_top5.png` (hai truy vấn).
- NB2: `nb2_precision.png`, `nb2_slices.png`.
- NB3: `nb3_response.png`, `nb3_latency.png`.
- NB4: `nb4_views.png`, `nb4_materialize.png`, `nb4_online.png`, `nb4_pit.png`.

Chạy các kiểm tra bạn thực hiện sau khi lưu notebook:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -X utf8 scripts\verify_docker.py
.\.venv\Scripts\python.exe -X utf8 scripts\prepare_submission.py
```

Pytest dùng Qdrant in-memory và model fastembed riêng để kiểm tra thư viện của lab.
Lệnh prepare_submission tạo/cập nhật reflection từ số đo và liệt kê bằng chứng còn thiếu.
Có thể đổi tên trong reflection bằng:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\prepare_submission.py --name "HoangThaiDat"
```

Đọc lại reflection và hai bảng chất lượng để giải thích được kết quả.
Chỉ khi kết quả và bằng chứng đủ mới chuẩn bị nộp.

## 6. Nộp bài

Repo origin hiện là `https://github.com/Liber72/K4-Track2-Day19-VectorFeatureStore-Lab.git`.
Kiểm tra repo public, rồi chạy sau khi hoàn tất output/ảnh:

```powershell
git status
git add -A
git commit -m "Lab 19 submission - HoangThaiDat"
git push -u origin main
```

Nộp **URL GitHub public** vào **VinUni LMS Day 19**. Không cần PR.
Giữ repo public đến khi có điểm.

NB2 reports three modes: keyword, semantic and Weighted RRF (keyword=0.05, semantic=0.95, depth=50 for top-10, k=60). The API uses the same weights. These weights were selected after inspecting the lab sweep; report this change in the reflection. The sweep measured 95.4% Hybrid versus 95.2% Semantic and 80.4% Keyword. The mixed slice ties Semantic at 99.5%.

Sau khi đổi trọng số, restart kernel NB2 và chạy lại để lưu bảng kết quả chính.
Nếu API đã chạy, dừng rồi khởi động lại trước NB3; NB3 kiểm tra trọng số qua `/healthz`
để tránh đo nhầm tiến trình còn dùng cấu hình cũ.

NB2 có thêm **mục 6 — Chẩn đoán**: chạy sau mục 5 để so sánh 25 cấu hình trọng số
và độ sâu lấy ứng viên. Xem cột `beats both` và dòng `Best measured configuration`.
Kết quả được lưu riêng ở `submission/results/nb2_tuning.json`; danh sách tài liệu
ở `nb2_rankings.json` giúp phân tích nguyên nhân tăng/giảm. Phần này không tự thay
đổi API hoặc kết quả chính. Chỉ áp dụng cấu hình sau khi xem số đo; khi thay đổi
phải đồng bộ NB2/API và chạy lại NB2, NB3. Đây là tối ưu trên bộ lab đã xem,
không phải kết quả trên một tập kiểm tra độc lập.
