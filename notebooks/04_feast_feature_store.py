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
# # NB4 — Feast: Postgres offline → Redis online
#
# Ba feature views: user_profile, item_popularity và query_velocity.
# Sinh dữ liệu có timestamp, nạp Postgres :5433, đăng ký Feast, materialize
# sang Redis :6379, đo online lookup và thực hiện point-in-time (PIT) join.
# Mục tiêu: online P99 < 10 ms và PIT trả đúng feature tại thời điểm sự kiện.

# %%
import _setup  # noqa: F401
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml
from feast import FeatureStore
from feast.repo_config import RepoConfig
from sqlalchemy import DateTime, Float, create_engine, text
from sqlalchemy.engine import URL
from app.evaluation import percentile, save_result

ROOT = Path(_setup.__file__).resolve().parent.parent
FEAST_DIR = ROOT / "app/feast_repo"
with (FEAST_DIR / "feature_store.yaml").open(encoding="utf-8") as f:
    feast_config = RepoConfig(**yaml.safe_load(f))
assert feast_config.online_store.type == "redis"
assert feast_config.offline_store.type == "postgres"
assert feast_config.offline_store.port == 5433
pg = feast_config.offline_store
FEAST_CLI = Path(sys.executable).with_name("feast.exe" if os.name == "nt" else "feast")
assert FEAST_CLI.exists(), "Kernel phải là Python (Lab 19)"

feast_logs = []
def run_feast(*args):
    print("feast " + " ".join(args))
    res = subprocess.run(
        [str(FEAST_CLI), *args], cwd=str(FEAST_DIR),
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=180, check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    assert res.returncode == 0, f"Feast command failed: {res.stderr or res.stdout}"
    feast_logs.append({"command": list(args), "stdout": res.stdout, "stderr": res.stderr})
    return res

# %% [markdown]
# ## 1. Sinh dữ liệu và nạp ba bảng Postgres
#
# user_profile có hai phiên bản mỗi user để thấy PIT chọn dữ liệu cũ khi cần.
# item_popularity dùng doc_id thật trong corpus NB1. query_velocity mô tả hoạt
# động gần đây của 100 user. Timestamp luôn có múi giờ UTC.
# Chạy lại cell sẽ thay dữ liệu demo của ba bảng này trong database feast_offline.

# %%
NOW = datetime.now(timezone.utc).replace(microsecond=0)
with (ROOT / "data/corpus_vn.jsonl").open(encoding="utf-8") as f:
    docs = [json.loads(line) for line in f if line.strip()]
assert len(docs) == 1000

profile_rows = []
for i in range(100):
    common = {
        "user_id": f"u_{i:03d}",
        "preferred_language": "vi" if i % 3 else "en",
        "topic_affinity": ["ai_ml", "cloud", "security", "database", "devops"][i % 5],
    }
    speed = 180 + (i * 7) % 200
    profile_rows.extend([
        {**common, "reading_speed_wpm": speed - 20,
         "event_timestamp": NOW - timedelta(days=3)},
        {**common, "reading_speed_wpm": speed,
         "event_timestamp": NOW - timedelta(hours=i % 48)},
    ])

tables = {
    "user_profile": pd.DataFrame(profile_rows),
    "item_popularity": pd.DataFrame({
        "doc_id": [d["doc_id"] for d in docs],
        "click_count_24h": [(i * 13) % 500 for i in range(len(docs))],
        "ctr_7d": [((i * 7) % 100) / 100.0 for i in range(len(docs))],
        "avg_dwell_seconds": [10.0 + (i * 0.7) % 90 for i in range(len(docs))],
        "event_timestamp": [NOW - timedelta(minutes=i % 720) for i in range(len(docs))],
    }),
    "query_velocity": pd.DataFrame({
        "user_id": [f"u_{i:03d}" for i in range(100)],
        "queries_last_hour": [(i * 11) % 50 for i in range(100)],
        "distinct_topics_24h": [1 + (i * 3) % 10 for i in range(100)],
        "event_timestamp": [NOW - timedelta(minutes=i % 30) for i in range(100)],
    }),
}

engine = create_engine(
    URL.create("postgresql+psycopg", username=pg.user, password=pg.password,
               host=pg.host, port=pg.port, database=pg.database),
    connect_args={"sslmode": pg.sslmode},
)
try:
    with engine.begin() as conn:
        for table_name, frame in tables.items():
            sql_types = {"event_timestamp": DateTime(timezone=True)}
            if table_name == "item_popularity":
                sql_types.update({"ctr_7d": Float(precision=24), "avg_dwell_seconds": Float(precision=24)})
            frame.to_sql(
                table_name, conn, schema=pg.db_schema, if_exists="replace",
                index=False, dtype=sql_types,
            )
            count = conn.execute(
                text(f'SELECT COUNT(*) FROM "{pg.db_schema}"."{table_name}"')
            ).scalar_one()
            assert count == len(frame)
            print(f"Postgres {table_name}: {count} rows")
finally:
    engine.dispose()
print(tables["user_profile"].head(4).to_string(index=False))

# %% [markdown]
# ## 2. Feast apply và kiểm tra ba feature views
#
# Feast đọc definitions, kiểm tra nguồn Postgres và ghi metadata vào registry.db.
# Lệnh feature-views list cung cấp bằng chứng tên các view đã đăng ký.

# %%
run_feast("apply")
run_feast("feature-views", "list")
fs = FeatureStore(repo_path=str(FEAST_DIR))
registered_views = sorted(view.name for view in fs.list_feature_views())
expected_views = ["item_popularity_features", "query_velocity_features", "user_profile_features"]
assert registered_views == expected_views
print("Registered views:", registered_views)

# %% [markdown]
# ## 3. Materialize Postgres → Redis
#
# Materialize một khoảng đầy đủ giúp nạp lại dữ liệu demo khi chạy lại notebook.
# Sau đó chạy materialize-incremental để xác minh luồng cập nhật kể từ mốc đã nạp.
# Lần incremental ngay sau đó có thể không có hàng mới; log lần đầy đủ chứng minh
# dữ liệu đã được đưa vào Redis.

# %%
start_dt = (NOW - timedelta(days=31)).isoformat()
end_dt = (NOW + timedelta(seconds=1)).isoformat()
run_feast("materialize", start_dt, end_dt)
run_feast("materialize-incremental", (NOW + timedelta(seconds=2)).isoformat())
print("Materialize và materialize-incremental thành công")

# %% [markdown]
# ## 4. Online lookup cho u_001 và một doc thật
#
# Cùng một request lấy feature từ cả ba views. Redis giữ giá trị mới nhất
# từng entity. Kiểm tra tất cả feature có giá trị trước khi đo latency.

# %%
fs = FeatureStore(repo_path=str(FEAST_DIR))
REQUEST_FEATURES = [
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "user_profile_features:topic_affinity",
    "item_popularity_features:click_count_24h",
    "item_popularity_features:ctr_7d",
    "item_popularity_features:avg_dwell_seconds",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]
sample_entity = {"user_id": "u_001", "doc_id": docs[0]["doc_id"]}
start = time.perf_counter()
online = fs.get_online_features(
    features=REQUEST_FEATURES, entity_rows=[sample_entity], full_feature_names=True,
).to_dict()
single_latency_ms = (time.perf_counter() - start) * 1000
online_sample = {key: values[0] for key, values in online.items()}
for ref in REQUEST_FEATURES:
    assert online_sample[ref.replace(":", "__")] is not None, f"Missing {ref}"
assert online_sample["user_profile_features__reading_speed_wpm"] == 187
print(f"Single lookup: {single_latency_ms:.2f}ms")
print(json.dumps(online_sample, ensure_ascii=False, indent=2))

# %% [markdown]
# ## 5. Warm-up và đo 100 online lookups
#
# 10 lượt warm-up không tính vào bảng. P50/P95/P99 dùng nearest-rank,
# gồm thời gian Feast lấy feature và chuyển response thành dict.

# %%
for _ in range(10):
    fs.get_online_features(
        features=REQUEST_FEATURES, entity_rows=[sample_entity], full_feature_names=True,
    ).to_dict()

latencies = []
for i in range(100):
    entity = {"user_id": f"u_{i:03d}", "doc_id": docs[i]["doc_id"]}
    start = time.perf_counter()
    values = fs.get_online_features(
        features=REQUEST_FEATURES, entity_rows=[entity], full_feature_names=True,
    ).to_dict()
    latencies.append((time.perf_counter() - start) * 1000)
    assert all(values[ref.replace(":", "__")][0] is not None for ref in REQUEST_FEATURES)
lookup_metrics = {
    "n_calls": len(latencies),
    "p50": percentile(latencies, 0.50),
    "p95": percentile(latencies, 0.95),
    "p99": percentile(latencies, 0.99),
}
print("Online lookup latency — Redis (ms)")
for key in ("p50", "p95", "p99"):
    print(f"  {key.upper()} = {lookup_metrics[key]:.2f}ms")
print("PASS — online P99 < 10ms" if lookup_metrics["p99"] < 10
      else "Chưa đạt online P99 < 10ms; giữ số đo và kiểm tra tải máy.")

# %% [markdown]
# ## 6. PIT join: lấy feature tại đúng thời điểm
#
# u_001 được hỏi ở NOW - 2 giờ, trước phiên bản profile mới tại NOW - 1 giờ.
# PIT phải trả reading_speed_wpm = 167; online lookup trả phiên bản mới = 187.
# Hai user còn lại có phiên bản mới trước thời điểm truy vấn của họ.

# %%
entity_df = pd.DataFrame({
    "user_id": ["u_001", "u_002", "u_003"],
    "event_timestamp": [NOW - timedelta(hours=2), NOW - timedelta(hours=1), NOW],
})
historical = fs.get_historical_features(
    entity_df=entity_df,
    features=["user_profile_features:reading_speed_wpm", "user_profile_features:topic_affinity"],
    full_feature_names=True,
).to_df()
print(historical.to_string(index=False))
assert len(historical) == 3
speeds = historical.set_index("user_id")["user_profile_features__reading_speed_wpm"].to_dict()
assert speeds == {"u_001": 167, "u_002": 194, "u_003": 201}
print("PASS — PIT trả 3 dòng và chọn đúng phiên bản feature")

result_path = save_result("nb4_features.json", {
    "online_store": "redis", "offline_store": "postgres", "postgres_port": pg.port,
    "table_rows": {name: len(frame) for name, frame in tables.items()},
    "registered_views": registered_views, "materialize_passed": True,
    "online_sample": online_sample, "single_lookup_ms": single_latency_ms,
    "lookup_latency": lookup_metrics, "online_p99_under_10ms": lookup_metrics["p99"] < 10,
    "pit_rows": len(historical), "pit_passed": True,
    "pit_sample": json.loads(historical.to_json(orient="records", date_format="iso")),
    "feast_commands": feast_logs,
})
print(f"Saved: {result_path}")

# %% [markdown]
# ## 7. Tạo reflection từ số đo thật
#
# Đọc JSON của NB2–NB4 để điền kết quả thắng/thua và latency. Phần trả lời
# reflection không quá 200 từ. Nếu chưa chạy NB2/NB3, chạy lại cell sau khi có kết quả.

# %%
from scripts.prepare_submission import write_reflection
write_reflection(ROOT)

# %% [markdown]
# ## Bằng chứng cần lưu
#
# - Mục 2: ba feature views → nb4_views.png.
# - Mục 3: log materialize → nb4_materialize.png.
# - Mục 4 và 5: online lookup + bảng latency → nb4_online.png.
# - Mục 6: PIT join → nb4_pit.png.
# - Ctrl+S giữ output, chụp ảnh vào submission/screenshots.
# - Đối chiếu toàn bộ việc còn lại trong submission/RUN_GUIDE.md.
