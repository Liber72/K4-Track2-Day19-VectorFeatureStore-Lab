"""Single-process rolling aggregation; Feast consumes computed features."""
import re
import time
import unicodedata
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone

import pandas as pd
from feast.data_source import PushMode

from bonus.features import persist_snapshot

TOPIC_TERMS = {
    "security": ("security", "bảo mật", "iam", "zero trust", "mã hóa"),
    "cloud": ("cloud", "kubernetes", "hạ tầng", "đám mây", "autoscaling", "aws"),
    "ai_ml": ("embedding", "llm", "học máy", "ai"),
    "database": ("postgres", "redis", "database", "cơ sở dữ liệu"),
}


def detect_topic(query: str) -> str:
    text = unicodedata.normalize("NFC", query).casefold()
    for topic, terms in TOPIC_TERMS.items():
        if any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)
               for term in terms):
            return topic
    return "other"


class RecentActivity:
    """The count includes the current recall, and resets on process restart."""

    def __init__(self, store, *, clock=time.time, snapshot_writer=persist_snapshot):
        self.store, self.clock, self.snapshot_writer = store, clock, snapshot_writer
        self.events = defaultdict(deque)

    def record(self, query: str, user_id: str) -> None:
        now = self.clock()
        events = self.events[user_id]
        while events and events[0][0] <= now - 86400:
            events.popleft()
        events.append((now, detect_topic(query)))
        last_hour = [topic for timestamp, topic in events if timestamp > now - 3600]
        topics = Counter(topic for topic in last_hour if topic != "other")
        frame = pd.DataFrame([{
            "user_id": user_id, "queries_last_hour": len(last_hour),
            "distinct_topics_24h": len({topic for _, topic in events if topic != "other"}),
            "recent_topic": topics.most_common(1)[0][0] if topics else "unknown",
            "activity_updated_at_ts": int(now),
            "event_timestamp": datetime.fromtimestamp(now, timezone.utc),
        }])
        self.snapshot_writer(self.store, "memory_activity", frame)
        self.store.push("bonus_activity_push", frame, to=PushMode.ONLINE)
