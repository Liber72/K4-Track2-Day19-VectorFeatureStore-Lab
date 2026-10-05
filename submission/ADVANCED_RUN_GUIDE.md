# Chạy NB5–NB8 trên Windows / Docker

Code và notebook đã được chuẩn bị. Học viên tự chạy, lưu output và chụp ảnh.
Chưa có kết quả runtime nâng cao được xác nhận. Khối này là 50 điểm nâng cao
trong `rubric.md`; bonus AI Memory là phần khác.

## 1. Mở JupyterLab

Trong PowerShell:

```powershell
Set-Location "D:\AI_Thucchien\K4-Track2-Day19-VectorFeatureStore-Lab"
docker compose up -d
docker compose ps
.\.venv\Scripts\python.exe -m jupyterlab --notebook-dir=notebooks
```

Qdrant, Redis, Postgres cần healthy. Giữ `.env` hiện tại:
`QDRANT_MODE=server`, `QDRANT_URL=http://localhost:6333`, `EMBEDDING_BACKEND=bge-m3`.
NB1 phải có collection `lab19`; NB6 cần profile đã được materialize trong NB4.
Nếu Jupyter đang mở, chỉ refresh danh sách file, không cần mở thêm server.

Mở bốn file `.ipynb` theo thứ tự 05 → 06 → 07 → 08. Chọn kernel
**Python (Lab 19)**, chạy từng cell bằng **Shift+Enter** hoặc
**Run → Run All Cells**. Nếu chạy lại, chọn **Kernel → Restart Kernel and Run All Cells**.
Đợi cell hết dấu `[*]`, đọc kết quả rồi **Ctrl+S**. Khi xong một notebook,
**Kernel → Shut Down Kernel** để giải phóng model trước notebook tiếp theo.

## 2. NB5 — Filtered search

File: `notebooks/05_filtered_search.ipynb`.

- Mục 1 dùng lại vector NB1, tạo collection dẫn xuất `lab19_filtered_nb5`.
- Mục 2 in recall theo selectivity, so với cosine exact trên subset đúng.
- Mục 3 in over-fetch ladder và mốc đầu đạt recall 1.00 trên ba query.
- Mục 4 kiểm tra ba tenant; mục 5 lưu kết quả và các mục PASS/REVIEW.

Mục tiêu rubric: post-filter giảm rõ khi filter chặt, filtered recall đạt 1.00;
over-fetch cần khoảng nửa corpus. Mốc thực tế phụ thuộc model. Kiểm tra tự động
dùng khoảng 40–60% để diễn giải “khoảng nửa”; grader vẫn xem bảng thật.
Số ứng viên trả về không phải số vector engine đã quét.

Chụp `nb5_selectivity.png`, `nb5_overfetch.png`.

## 3. NB6 — Agentic retrieval

File: `notebooks/06_agent_retrieval.ipynb`.

- Mục 1 tự sinh 12 câu ghép và gold với model hiện tại; không cần chạy
  `gen_agent_queries.py` trước. Query/gold cũng được lưu trong báo cáo kết quả.
- Mục 3 so ba chiến lược với tổng `top_k=16` mỗi câu. Retry tắt trong bảng này.
  Doc unique có thể ít hơn 16 do trùng/filter; cột riêng thể hiện điều đó.
- So cả recall lẫn balance của agentic không filter với single-shot.
- Đọc query/trace ngay sau bảng để giải thích tác động của filter suy đoán.
- Mục 4 minh họa retry riêng; mục 5 in profile Feast và doc IDs thật.

Mục tiêu rubric: agentic tăng cả recall/balance; giải thích vì sao filter có thể
làm mất evidence; context có feature và doc IDs. Nếu Feast thiếu profile hoặc
báo `_error`, kiểm tra Docker và chạy lại NB4, rồi restart NB6.
Planner theo luật có thể tách nhầm liên từ; kết quả không được gán trước.

Chụp `nb6_comparison.png`, `nb6_reflection.png`, `nb6_context.png`.

## 4. NB7 — Semantic cache

File: `notebooks/07_semantic_cache.ipynb`.

- Dùng bge-m3 theo `.env` và đúng dimension; cache trong RAM, collection riêng.
- Mục 2: `correct_savings` là phần tổng request tái sử dụng đúng đáp án theo ID;
  `wrong_answer_rate` là phần tổng request nhận đáp án sai. `hit_rate` gồm cả sai.
- Cột `wrong_given_hit` là lỗi trên số hit. Chính sách minh họa chọn tối đa 5%
  lỗi trên hit và tối đa tiết kiệm đúng; không mặc định 0.75/0.85 là tốt.
- Mục 3 in ngưỡng chọn và số đo ở 0.75. Đây là tuning trên cùng probe, không
  phải kết quả trên tập kiểm thử độc lập. Proxy ID không thay nhãn answer thật.
- Mục 4 cần HIT tại 0/600s, MISS từ 1800s; mục 5 cần leak khi namespace=False
  và MISS khi namespace=True.

Chụp `nb7_threshold.png` (bảng và lựa chọn), `nb7_ttl.png`, `nb7_tenant.png`.

## 5. NB8 — Feature engineering

File: `notebooks/08_feature_engineering.ipynb`.

- Mục 1–2: các feature nhân quả và AUC train/holdout.
- Mục 3: `session_id`, target-naive gap >0.30; in-fold gần 0 (check |gap|<0.10).
- Mục 4: tỷ lệ dòng rò và chênh lệch AUC latest/PIT.
- Mục 5: ví dụ embedding làm feature, ghi rõ vector giả lập.
- Mục 6 tự sinh parquet, `feast apply`, materialize 7 ngày tới hiện tại, rồi
  gọi ODFV cho cùng user với hai amount. Hai ratio phải khác nhau và đúng công thức.
- Mục 7 lưu kết quả.

Feast `lab19_odfv` dùng SQLite/file riêng trong `app/feast_repo_ondemand/`.
Đây là thiết kế tự chứa của NB8, không cần đổi NB4 sang SQLite. Materialize dùng
cửa sổ tường minh để chạy lại được, tránh mốc tương lai cố định của bản cũ.

Chụp `nb8_leakage.png`, `nb8_pit.png`, `nb8_ondemand.png`.

## 6. Bằng chứng, test và nộp

Mỗi notebook lưu một báo cáo chỉ từ lần chạy của bạn:

| Notebook | Báo cáo |
|---|---|
| NB5 | `submission/results/nb5_filtered_search.json` |
| NB6 | `submission/results/nb6_agent_retrieval.json` |
| NB7 | `submission/results/nb7_semantic_cache.json` |
| NB8 | `submission/results/nb8_feature_engineering.json` |

`status=running` nghĩa là đã bắt đầu nhưng chưa tới cell cuối; có thể đã lỗi hoặc
bị dừng. `needs_review` nghĩa là chạy tới cuối nhưng còn tiêu chí chưa đạt.
`passed` chỉ xác nhận các điều kiện được code kiểm tra, không phải điểm do giảng
viên chấm. Luôn đối chiếu timestamp với notebook đã lưu. Cell khởi tạo báo cáo
đánh dấu lần chạy mới để không dùng nhầm PASS của lần trước.

Đặt ảnh trong `submission/screenshots/`. Sau khi đã lưu output NB5–NB8, tự chạy:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -X utf8 scripts\verify_lite.py
```

Đây là lệnh Windows tương đương `make test`, `make verify-lite` trong rubric.
Tests dùng Qdrant in-memory / fastembed riêng; có thể tải model fastembed lần đầu.
Giữ output test và chụp `advanced_tests.png`, `advanced_verify_lite.png`.
Chạy trên môi trường hiện có chưa chứng minh được tiêu chí “máy sạch”; nếu cần
bằng chứng đó phải dựng môi trường sạch riêng theo README.

Nếu có traceback, dừng notebook tại lỗi và giữ nguyên output. Nếu chỉ có REVIEW,
gửi bảng và JSON tương ứng để phân tích model/planner/threshold. Không đổi số đo
hoặc ghi sẵn kết quả đạt trong notebook.

Chỉ commit/push cùng repo khi notebook đã giữ output, các tiêu chí đã được xem
và bằng chứng đầy đủ. Không commit registry/database sinh ra bởi Feast.

## Tài liệu đối chiếu

- [Qdrant filtering](https://qdrant.tech/documentation/search-patterns/vector-search-filtering/):
  engine có thể chọn scan hoặc index tùy filter/dữ liệu; latency lab không chứng minh
  rằng chỉ top-K vector đã được quét.
- [Feast ingestion](https://docs.feast.dev/getting-started/concepts/data-ingestion):
  materialization nạp feature từ batch source vào online store.
