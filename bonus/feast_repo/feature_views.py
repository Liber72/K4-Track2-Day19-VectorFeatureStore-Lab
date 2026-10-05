"""Two push-backed views; PostgreSQL snapshots support PIT joins."""
from datetime import timedelta

from feast import Entity, FeatureView, Field, PushSource, ValueType
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Int64, String

user = Entity(name="user", join_keys=["user_id"], value_type=ValueType.STRING)
profile_batch = PostgreSQLSource(
    name="bonus_profile_batch", table="bonus_user_profile",
    timestamp_field="event_timestamp",
)
activity_batch = PostgreSQLSource(
    name="bonus_activity_batch", table="bonus_recent_activity",
    timestamp_field="event_timestamp",
)
profile_push = PushSource(name="bonus_profile_push", batch_source=profile_batch)
activity_push = PushSource(name="bonus_activity_push", batch_source=activity_batch)

profile_features = FeatureView(
    name="memory_profile", entities=[user], ttl=timedelta(days=30),
    schema=[
        Field(name="preferred_language", dtype=String),
        Field(name="reading_speed_wpm", dtype=Int64),
        Field(name="topic_affinity", dtype=String),
        Field(name="active_hours_local", dtype=String),
        Field(name="profile_updated_at_ts", dtype=Int64),
    ],
    source=profile_push, online=True,
)
activity_features = FeatureView(
    name="memory_activity", entities=[user], ttl=timedelta(hours=1),
    schema=[
        Field(name="queries_last_hour", dtype=Int64),
        Field(name="distinct_topics_24h", dtype=Int64),
        Field(name="recent_topic", dtype=String),
        Field(name="activity_updated_at_ts", dtype=Int64),
    ],
    source=activity_push, online=True,
)
