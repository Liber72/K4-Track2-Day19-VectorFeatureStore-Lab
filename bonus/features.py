"""Small PostgreSQL/Feast setup used by the demo and activity processor."""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg
from feast import FeatureStore
from feast.data_source import PushMode
from psycopg import sql

from bonus.feast_repo.feature_views import (
    activity_features, activity_push, profile_features, profile_push, user,
)

FEATURE_REPO = Path(__file__).resolve().parent / "feast_repo"
PROFILE_FIELDS = [field.name for field in profile_features.features]
ACTIVITY_FIELDS = [field.name for field in activity_features.features]
FEATURE_REFS = ([f"memory_profile:{name}" for name in PROFILE_FIELDS]
                + [f"memory_activity:{name}" for name in ACTIVITY_FIELDS])
TABLES = {"memory_profile": "bonus_user_profile",
          "memory_activity": "bonus_recent_activity"}


def connect_offline(store):
    config = store.config.offline_store
    return psycopg.connect(
        host=config.host, port=config.port, dbname=config.database,
        user=config.user, password=config.password,
        sslmode=config.sslmode, connect_timeout=5,
    )


def persist_snapshot(store, view_name: str, frame: pd.DataFrame) -> None:
    """Append timestamped features before publishing them to Redis."""
    columns = list(frame.columns)
    statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(store.config.offline_store.db_schema, TABLES[view_name]),
        sql.SQL(", ").join(map(sql.Identifier, columns)),
        sql.SQL(", ").join(sql.Placeholder() for _ in columns),
    )
    with connect_offline(store) as connection, connection.cursor() as cursor:
        cursor.executemany(statement, list(frame.itertuples(index=False, name=None)))


def prepare_demo_features() -> FeatureStore:
    """Create only bonus tables/views and seed two explicitly synthetic users."""
    store = FeatureStore(repo_path=str(FEATURE_REPO))
    profile_columns = """
        user_id TEXT NOT NULL, preferred_language TEXT, reading_speed_wpm BIGINT,
        topic_affinity TEXT, active_hours_local TEXT, profile_updated_at_ts BIGINT,
        event_timestamp TIMESTAMPTZ NOT NULL
    """
    activity_columns = """
        user_id TEXT NOT NULL, queries_last_hour BIGINT, distinct_topics_24h BIGINT,
        recent_topic TEXT, activity_updated_at_ts BIGINT,
        event_timestamp TIMESTAMPTZ NOT NULL
    """
    with connect_offline(store) as connection:
        for table, columns in ((TABLES["memory_profile"], profile_columns),
                               (TABLES["memory_activity"], activity_columns)):
            connection.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
                sql.Identifier(store.config.offline_store.db_schema, table),
                sql.SQL(columns),
            ))
    store.apply([user, profile_push, activity_push, profile_features, activity_features])
    now = datetime.now(timezone.utc)
    profiles = pd.DataFrame([
        {"user_id": "u_001", "preferred_language": "mix", "reading_speed_wpm": 240,
         "topic_affinity": "cloud", "active_hours_local": "20:00-23:00 Asia/Ho_Chi_Minh",
         "profile_updated_at_ts": int(now.timestamp()), "event_timestamp": now},
        {"user_id": "u_002", "preferred_language": "vi", "reading_speed_wpm": 180,
         "topic_affinity": "ai_ml", "active_hours_local": "07:00-09:00 Asia/Ho_Chi_Minh",
         "profile_updated_at_ts": int(now.timestamp()), "event_timestamp": now},
    ])
    activity = pd.DataFrame([
        {"user_id": uid, "queries_last_hour": 0, "distinct_topics_24h": 0,
         "recent_topic": "unknown", "activity_updated_at_ts": int(now.timestamp()),
         "event_timestamp": now} for uid in ("u_001", "u_002")
    ])
    for view, source, frame in (("memory_profile", "bonus_profile_push", profiles),
                                ("memory_activity", "bonus_activity_push", activity)):
        persist_snapshot(store, view, frame)
        store.push(source, frame, to=PushMode.ONLINE)
    return store
