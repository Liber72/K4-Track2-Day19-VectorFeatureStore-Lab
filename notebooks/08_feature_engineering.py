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
# # NB8 — Feature Engineering và leakage
# Đo feature nhân quả, target encoding, PIT join và on-demand feature.
# Dữ liệu tổng hợp dùng seed cố định. Kết quả lưu ở
# `submission/results/nb8_feature_engineering.json`.
# ODFV dùng Feast project riêng `lab19_odfv` với SQLite/file, kể cả khi NB4
# dùng Redis/Postgres. Không thay registry hoặc ba feature view của NB4.

# %%
import _setup  # noqa: F401
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from _advanced import Evidence
from app.features import (auc, generate_events, latest_join, leakage_experiment,
                          leaked_row_fraction, pit_join, window_aggregates)

ROOT = Path(_setup.__file__).resolve().parent.parent
evidence = Evidence("nb8_feature_engineering.json", notebook="NB8", seed=42)

# %% [markdown]
# ## 1. Event log và feature theo cửa sổ
# Bốn họ đầu: aggregation, ratio, lag/delta và recency. Mỗi feature chỉ dùng
# lịch sử trước sự kiện. Dòng đầu mỗi user không có lịch sử nên lag/ratio là NaN.

# %%
events = generate_events(n_users=200, n_days=30, seed=42)
print(f"Events={len(events)}; users={events.user_id.nunique()}; "
      f"sessions={events.session_id.nunique()}; click rate={events.clicked.mean():.3f}")
print(events.head(3).to_string(index=False))
features = window_aggregates(events)
columns = ["searches_1h", "searches_24h", "searches_7d", "query_len_vs_user_avg",
           "prev_query_len", "query_len_delta", "seconds_since_last"]
print(features[columns].describe().loc[["mean", "50%", "max"]].round(2).to_string())

# %% [markdown]
# ## 2. AUC train/holdout của feature nhân quả
# Dùng AUC trực tiếp của feature, không train classifier. Đây là minh họa IID;
# bài toán dự báo tương lai còn cần chia theo thời gian/user tùy cách triển khai.

# %%
mask = np.random.default_rng(0).random(len(features)) < 0.7
honest = []
for column in ("searches_24h", "searches_7d", "seconds_since_last"):
    train_auc = auc(features.loc[mask, column], features.loc[mask, "clicked"])
    test_auc = auc(features.loc[~mask, column], features.loc[~mask, "clicked"])
    honest.append({"feature": column, "train_auc": train_auc, "holdout_auc": test_auc,
                   "gap": train_auc - test_auc})
print(pd.DataFrame(honest).round(3).to_string(index=False))

# %% [markdown]
# ## 3. Họ categorical: target encoding leakage
# Chia train/holdout trước, fit encoder trên train. `target-naive` dùng cả nhãn
# của chính dòng train; `target-in-fold` mã hóa dòng đó từ các fold còn lại.
# Mục tiêu: gap naive >0.30 với `session_id`, gap in-fold gần 0.

# %%
session_leakage = leakage_experiment(events, "session_id")
user_leakage = leakage_experiment(events, "user_id")
print("session_id:\n", session_leakage.round(3).to_string(index=False))
print("\nuser_id:\n", user_leakage.round(3).to_string(index=False))
session_scores = session_leakage.set_index("encoding")
print("Naive gap:", session_scores.loc["target-naive", "gap"])
print("In-fold gap:", session_scores.loc["target-in-fold", "gap"])

# %% [markdown]
# ## 4. Latest join so với point-in-time join
# Feature snapshot ghi số hoạt động tích lũy. Latest join kéo snapshot tương lai
# về dòng huấn luyện; PIT lấy snapshot gần nhất có timestamp <= thời điểm sự kiện.
# Đo tỷ lệ dòng bị rò và chênh lệch AUC, không gán trước dấu của chênh lệch.

# %%
snapshots = events[["user_id", "event_timestamp"]].copy().sort_values("event_timestamp")
snapshots["feature_value"] = snapshots.groupby("user_id").cumcount() + 1
sample = np.random.default_rng(1).random(len(events)) < 0.4
entities = events.loc[sample, ["user_id", "event_timestamp", "clicked"]].copy()
latest, pit = latest_join(entities, snapshots), pit_join(entities, snapshots)
pit_result = {
    "rows": len(entities), "leaked_row_fraction": leaked_row_fraction(entities, snapshots),
    "latest_auc": auc(latest["feature_value"], latest["clicked"]),
    "pit_auc": auc(pit["feature_value"], pit["clicked"]),
}
pit_result["auc_difference"] = pit_result["latest_auc"] - pit_result["pit_auc"]
print(f"Dòng bị rò: {pit_result['leaked_row_fraction']:.2%} / {len(entities)} rows")
print(f"Latest AUC: {pit_result['latest_auc']:.3f}; PIT AUC: {pit_result['pit_auc']:.3f}")
print(f"Delta latest - PIT: {pit_result['auc_difference']:+.3f}")

# %% [markdown]
# ## 5. Họ embedding: vector làm feature
# Dùng vector profile của user từ trung bình ba vector tài liệu đã đọc để minh họa.
# Trong bài này dùng vector nhỏ giả lập, không tải thêm model. Hệ thật có thể dùng
# vector NB1 và chỉ tổng hợp lịch sử **trước** thời điểm request để tránh leakage.

# %%
history_vectors = np.random.default_rng(42).normal(size=(3, 8))
user_vector = history_vectors.mean(axis=0)
user_vector /= np.linalg.norm(user_vector)
print("Ví dụ user embedding (synthetic, 8d):", user_vector.round(3))
print("Đây là ví dụ họ feature thứ 6; không dùng vector giả này để chấm AUC.")

# %% [markdown]
# ## 6. On-demand feature view
# `amount_vs_avg = amount / avg_amount_7d`: baseline lưu trong Feast, `amount`
# đến từ request. Chạy bằng Python của kernel hiện tại để không nhầm môi trường.
# Nạp lại cửa sổ 7 ngày tới thời điểm hiện tại, nên chạy lại vẫn lấy dữ liệu mới.
# Log stdout/stderr được in đầy đủ nếu Feast gặp lỗi.

# %%
repo = ROOT / "app/feast_repo_ondemand"
command_logs = []


def run_command(args, cwd=ROOT):
    process = subprocess.run(
        [str(arg) for arg in args], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        timeout=180,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    print(process.stdout)
    if process.stderr:
        print(process.stderr)
    command_logs.append({"args": [str(arg) for arg in args], "exit_code": process.returncode,
                         "stdout": process.stdout, "stderr": process.stderr})
    process.check_returncode()


run_command([sys.executable, ROOT / "scripts/gen_spend.py"])
feast_executable = Path(sys.executable).with_name("feast.exe" if os.name == "nt" else "feast")
assert feast_executable.exists(), "Chọn kernel Python (Lab 19) có cài Feast"
feast_cli = [str(feast_executable)]
run_command([*feast_cli, "apply"], cwd=repo)
end = datetime.now(timezone.utc)
start = end - timedelta(days=7)
run_command([*feast_cli, "materialize", start.isoformat(), end.isoformat()], cwd=repo)

# %%
from feast import FeatureStore

store = FeatureStore(repo_path=str(repo))
requests = [
    {"user_id": "u_000", "amount": 100_000.0},
    {"user_id": "u_000", "amount": 15_000_000.0},
    {"user_id": "u_001", "amount": 250_000.0},
]
online = store.get_online_features(
    features=["user_spend_stats:avg_amount_7d", "amount_vs_avg:amount_vs_avg",
              "amount_vs_avg:is_spike"], entity_rows=requests,
).to_dict()
odfv_rows = []
for i, request in enumerate(requests):
    average = online["avg_amount_7d"][i]
    ratio = online["amount_vs_avg"][i]
    assert average is not None and average > 0, "Baseline chưa được materialize"
    assert ratio is not None, "ODFV chưa trả về amount_vs_avg"
    odfv_rows.append({**request, "avg_amount_7d": average,
                      "amount_vs_avg": ratio, "is_spike": online["is_spike"][i]})
print(pd.DataFrame(odfv_rows).round(3).to_string(index=False))

# %% [markdown]
# ## 7. Lưu bằng chứng
# Chụp mục 3, 4, 6 và Ctrl+S. Ngưỡng `abs(in-fold gap)<0.10` là mức kiểm tra
# “gần 0” đã dùng trong tests của lab. Hai request cùng user phải dùng cùng
# baseline nhưng ra hai ratio khác nhau, đúng công thức.

# %%
evidence.finish(
    checks={
        "session_naive_gap_above_030": session_scores.loc["target-naive", "gap"] > 0.30,
        "session_infold_gap_near_zero": abs(session_scores.loc["target-in-fold", "gap"]) < 0.10,
        "latest_join_has_future_rows": pit_result["leaked_row_fraction"] > 0,
        "pit_auc_measured": all(np.isfinite(pit_result[key])
                                for key in ("latest_auc", "pit_auc", "auc_difference")),
        "same_user_same_stored_average": odfv_rows[0]["avg_amount_7d"] == odfv_rows[1]["avg_amount_7d"],
        "same_user_different_ratios": odfv_rows[0]["amount_vs_avg"] != odfv_rows[1]["amount_vs_avg"],
        "odfv_matches_formula": all(np.isclose(row["amount_vs_avg"], row["amount"] / row["avg_amount_7d"])
                                    for row in odfv_rows),
    },
    n_events=len(events), honest_features=honest,
    session_leakage=session_leakage.to_dict("records"),
    user_leakage=user_leakage.to_dict("records"), pit_join=pit_result,
    on_demand=odfv_rows, feast_project=store.project, commands=command_logs,
)

# %%
