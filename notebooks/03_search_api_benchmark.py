# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python (Lab 19)
#     language: python
#     name: lab19
# ---

# %% [markdown]
# # NB3 — FastAPI /search và latency benchmark
#
# API dùng BM25 + bge-m3 + collection lab19 đã tạo ở NB1.
# Đo riêng thời gian xử lý server và thời gian toàn bộ HTTP request.
# Mục tiêu rubric: hybrid P99 **server-side < 50 ms** sau warm-up.
# Kết quả thực tế phụ thuộc máy; notebook in số đo và trạng thái đạt/chưa đạt.
#
# Lưu và shutdown kernel NB2 trước khi chạy để giải phóng model trong RAM.

# %%
import _setup  # noqa: F401
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from app.evaluation import RESULTS_DIR, percentile, save_result

ROOT = Path(_setup.__file__).resolve().parent.parent
URL = "http://127.0.0.1:8000"

# %% [markdown]
# ## 1. Khởi động API và chờ ready
#
# Notebook gọi uvicorn bằng đúng Python của kernel Lab 19. Trong Docker mode,
# API kiểm tra và dùng lại index NB1. Model câu hỏi sẽ được nạp ở request đầu.
# Log khởi động nằm ở submission/results/api_server.log.
# Nếu một API Lab 19 đúng cấu hình đã chạy trên cổng 8000, notebook dùng lại nó.

# %%
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
api_proc = globals().get("api_proc")
api_log_handle = globals().get("api_log_handle")
api_owned = api_proc is not None and api_proc.poll() is None
http_client = httpx.Client(base_url=URL, timeout=120.0, trust_env=False)

def stop_owned_api():
    if api_owned and api_proc is not None and api_proc.poll() is None:
        api_proc.terminate()
        try:
            api_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            api_proc.kill()
            api_proc.wait(timeout=10)
    if api_log_handle is not None and not api_log_handle.closed:
        api_log_handle.close()
    http_client.close()

def get_ready_status():
    try:
        response = http_client.get("/healthz", timeout=2)
        response.raise_for_status()
        status = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    expected = {
        "app": "lab19-search", "ready": True, "n_docs": 1000,
        "collection": "lab19", "embedding_backend": "bge-m3",
        "reused_index": True,
    }
    return status if all(status.get(k) == v for k, v in expected.items()) else None

with socket.socket() as sock:
    sock.settimeout(1)
    port_in_use = sock.connect_ex(("127.0.0.1", 8000)) == 0

if port_in_use:
    api_status = get_ready_status()
    if api_status is None:
        http_client.close()
        raise RuntimeError("Cổng 8000 đang chạy dịch vụ khác hoặc API sai cấu hình. Dừng dịch vụ đó rồi chạy lại cell.")
else:
    api_log_handle = (RESULTS_DIR / "api_server.log").open("w", encoding="utf-8")
    child_env = {**os.environ, "PYTHONUTF8": "1"}
    try:
        api_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"],
            cwd=str(ROOT), env=child_env,
            stdout=api_log_handle, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        api_owned = True
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if api_proc.poll() is not None:
                raise RuntimeError("API dừng khi khởi động; xem submission/results/api_server.log")
            api_status = get_ready_status()
            if api_status is not None:
                break
            time.sleep(1)
        else:
            raise RuntimeError("API chưa ready sau 180 giây; xem submission/results/api_server.log")
    except Exception:
        stop_owned_api()
        raise

print(json.dumps(api_status, ensure_ascii=False, indent=2))
print(f"API docs: {URL}/docs")

# %% [markdown]
# ## 2. Kiểm tra response của một truy vấn
#
# Request đầu có thể nạp model từ cache trên đĩa, nên latency ban đầu chưa dùng
# để kết luận benchmark. JSON có query, mode, top_k, latency_ms và danh sách hits.

# %%
response = http_client.get("/search", params={
    "q": "cloud computing tự động mở rộng", "mode": "hybrid", "top_k": 10,
})
response.raise_for_status()
sample_response = response.json()
assert sample_response["mode"] == "hybrid"
assert len(sample_response["hits"]) == 10
assert sample_response["latency_ms"] >= 0
print("Response preview — hiển thị 3 trong 10 hits:")
print(json.dumps({**sample_response, "hits": sample_response["hits"][:3]},
                 ensure_ascii=False, indent=2))

# %% [markdown]
# ## 3. Warm-up 10 queries cho mỗi mode
#
# Warm-up nạp model và làm nóng đường xử lý. API vẫn embed câu hỏi trong mỗi
# request; các latency phía server của warm-up không nằm trong bảng đo.

# %%
with (ROOT / "data/golden_set.jsonl").open(encoding="utf-8") as f:
    golden = [json.loads(line) for line in f if line.strip()]
assert len(golden) == 50

for mode in ("keyword", "semantic", "hybrid"):
    for q in golden[:10]:
        response = http_client.get("/search", params={"q": q["query"], "mode": mode})
        response.raise_for_status()
    print(f"Warm-up {mode}: 10 queries")

# %% [markdown]
# ## 4. Đo 100 requests cho mỗi mode
#
# 50 queries × 2 lượt = 100 requests/mode. P50/P95/P99 dùng nearest-rank.
# Server-side lấy từ latency_ms trong response; wall-clock gồm cả HTTP.
# Mỗi request đo gồm embedding câu hỏi và việc truy xuất/kết hợp kết quả.

# %%
def benchmark_mode(mode: str, reps: int = 2) -> dict:
    server_latencies, wall_latencies = [], []
    for _ in range(reps):
        for q in golden:
            start = time.perf_counter()
            response = http_client.get("/search", params={
                "q": q["query"], "mode": mode, "top_k": 10,
            })
            response.raise_for_status()
            body = response.json()
            wall_latencies.append((time.perf_counter() - start) * 1000)
            assert len(body["hits"]) == 10
            server_latencies.append(body["latency_ms"])
            if len(server_latencies) % 25 == 0:
                print(f"Measured {mode}: {len(server_latencies)}/{reps * len(golden)}")
    return {
        "n_calls": len(server_latencies),
        "p50_server": percentile(server_latencies, 0.50),
        "p95_server": percentile(server_latencies, 0.95),
        "p99_server": percentile(server_latencies, 0.99),
        "p99_wall": percentile(wall_latencies, 0.99),
    }

results = {mode: benchmark_mode(mode) for mode in ("keyword", "semantic", "hybrid")}
print("\nLatency (ms) — sau warm-up")
print(f"  {'mode':10} {'P50 server':>12} {'P95 server':>12} {'P99 server':>12} {'P99 wall':>12}")
for mode, metrics in results.items():
    print(f"  {mode:10} {metrics['p50_server']:>12.2f} {metrics['p95_server']:>12.2f} "
          f"{metrics['p99_server']:>12.2f} {metrics['p99_wall']:>12.2f}")

hybrid_p99 = results["hybrid"]["p99_server"]
print(f"Hybrid P99 server-side: {hybrid_p99:.2f}ms")
print("PASS — hybrid P99 < 50ms" if hybrid_p99 < 50
      else "Chưa đạt P99 < 50ms; giữ số đo, xem tải CPU/GPU và đo lại sau warm-up.")

result_path = save_result("nb3_latency.json", {
    "n_queries": len(golden), "warmup_per_mode": 10,
    "percentile_method": "nearest-rank", "api_status": api_status,
    "sample_response": sample_response, "latencies": results,
    "hybrid_p99_under_50ms": hybrid_p99 < 50,
})
print(f"Saved: {result_path}")

# %% [markdown]
# ## 5. Dọn tiến trình API
#
# Dừng API nếu notebook đã khởi động nó. Khi dùng API có sẵn, chỉ đóng HTTP
# client của notebook. Các bảng output vẫn được giữ để bạn chụp ảnh.

# %%
stop_owned_api()
print("API do notebook khởi động đã dừng." if api_owned
      else "Đã đóng HTTP client; API có sẵn tiếp tục chạy.")

# %% [markdown]
# ## Bằng chứng cần lưu
#
# - Mục 2: response hợp lệ, latency_ms và hits.
# - Mục 4: bảng P50/P95/P99 và kết luận hybrid từ số đo.
# - Ctrl+S, chụp `nb3_response.png` và `nb3_latency.png` vào submission/screenshots.
# - Khi cell đo bị lỗi, có thể chạy mục 5 để dừng API do notebook tạo.
