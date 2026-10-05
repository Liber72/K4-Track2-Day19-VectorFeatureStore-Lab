"""Feature engineering the six families, plus the two classic ways to leak (NB8).

Deck reference: "Feature Engineering: 6 Ho Feature Ban Se Viet Di Viet Lai".

The families:
  1 windowed aggregation   count/sum/avg over 1h / 24h / 7d
  2 ratio / normalisation   this event vs THIS user's own baseline
  3 lag & delta             previous value, and the change
  4 recency                 seconds since the user's last event
  5 categorical encoding    frequency + target encoding  <- leaks if done wrong
  6 embedding as feature    the vector itself becomes a column

Two leakage demonstrations are built in, both measurable with numpy alone
(no sklearn -- the lab stays a ~250 MB install):

  * target encoding fitted on ALL rows vs fitted IN-FOLD
  * "latest value" join vs point-in-time join

Both are scored with `auc()` on the raw feature, so no model training is
needed: if a feature is leaking, its own AUC on seen rows is inflated versus
held-out rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOPICS = ["cloud", "ai_ml", "security", "database", "networking",
          "devops", "mobile", "frontend", "backend", "data_eng"]


# ── synthetic event log ────────────────────────────────────────────────
def generate_events(n_users: int = 200, n_days: int = 30, seed: int = 42) -> pd.DataFrame:
    """A search-event log: who searched what, when, and did they click.

    `clicked` is the label. It depends on a per-user "engagement" propensity
    and on topic affinity -- so genuine features (recency, affinity) carry
    real signal and leaky features can be shown to carry FAKE signal.
    """
    rng = np.random.default_rng(seed)
    users = [f"u_{i:03d}" for i in range(n_users)]
    engagement = rng.beta(2, 3, size=n_users)          # per-user click propensity
    affinity = rng.integers(0, len(TOPICS), size=n_users)

    rows = []
    start = pd.Timestamp("2026-07-01", tz="UTC")
    for ui, user in enumerate(users):
        # Engaged users also search MORE -- so activity counts carry genuine
        # (non-leaky) signal about clicking, which is what family 1 exploits.
        n_events = int(5 + engagement[ui] * 90 + rng.integers(0, 10))
        offsets = np.sort(rng.uniform(0, n_days * 86400, size=n_events))
        for off in offsets:
            # 60% of a user's searches fall in their affinity topic
            topic = (TOPICS[affinity[ui]] if rng.random() < 0.6
                     else TOPICS[int(rng.integers(0, len(TOPICS)))])
            on_affinity = topic == TOPICS[affinity[ui]]
            p = engagement[ui] * (1.6 if on_affinity else 0.6)
            rows.append({
                "user_id": user,
                # store the raw offset; timestamps are built vectorised below
                "_offset_s": float(off),
                "topic": topic,
                # High-cardinality key: ~2-3 events per session. Target encoding
                # on a key this granular is dominated by the row's own label.
                "session_id": f"s_{ui:03d}_{int(off // 10800):04d}",
                "query_len": int(rng.integers(3, 25)),
                "results_shown": int(rng.integers(5, 20)),
                "clicked": int(rng.random() < min(p, 0.95)),
            })
    df = pd.DataFrame(rows)
    # Vectorised timestamp construction. Building these one `pd.Timedelta` at a
    # time emits a numpy-2.5 deprecation warning per row -- thousands of lines
    # of noise for a 9k-row log.
    df["event_timestamp"] = start + pd.to_timedelta(df.pop("_offset_s"), unit="s")
    cols = ["user_id", "event_timestamp", "topic", "session_id",
            "query_len", "results_shown", "clicked"]
    return df[cols].sort_values("event_timestamp").reset_index(drop=True)


_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def _window_seconds(window: str) -> int:
    """Parse "1h" / "24h" / "7d" -> seconds.

    Deliberately avoids `pd.Timedelta`: pandas 2.3 builds it through a
    generic-unit numpy timedelta, which numpy 2.5 deprecates. This loop runs
    once per row per window, so that warning fires thousands of times and
    drowns real output.
    """
    window = window.strip().lower()
    value, unit = window[:-1], window[-1]
    if unit not in _UNIT_SECONDS or not value.isdigit():
        raise ValueError(f"unsupported window {window!r}; use e.g. '1h', '24h', '7d'")
    return int(value) * _UNIT_SECONDS[unit]


# ── families 1-4: the honest ones ──────────────────────────────────────
def window_aggregates(events: pd.DataFrame, windows=("1h", "24h", "7d")) -> pd.DataFrame:
    """Family 1 + 3 + 4, computed causally (each row sees only its own past)."""
    df = events.sort_values("event_timestamp").copy()
    out = df[["user_id", "event_timestamp", "topic", "query_len", "clicked"]].copy()

    for w in windows:
        col = f"searches_{w}"
        vals = []
        # Build an explicit second-resolution timedelta. `pd.Timedelta(w)` has a
        # "generic" numpy unit, which numpy 2.5 deprecates (and will later raise)
        # when subtracted from a datetime64 array.
        delta = np.timedelta64(_window_seconds(w), "s")
        for _user, g in df.groupby("user_id", sort=False):
            ts = g["event_timestamp"].values.astype("datetime64[ns]")
            # count of *strictly previous* events inside the window
            cnt = [int(((ts[:i] > ts[i] - delta) & (ts[:i] < ts[i])).sum())
                   for i in range(len(ts))]
            vals.append(pd.Series(cnt, index=g.index))
        out[col] = pd.concat(vals).sort_index()

    g = df.groupby("user_id")["query_len"]
    # Family 2: ratio against the user's OWN expanding mean (shifted = causal)
    user_mean = g.transform(lambda s: s.shift().expanding().mean())
    out["query_len_vs_user_avg"] = (df["query_len"] / user_mean).replace([np.inf], np.nan)
    # Family 3: lag + delta
    out["prev_query_len"] = g.shift()
    out["query_len_delta"] = df["query_len"] - out["prev_query_len"]
    # Family 4: recency
    prev_ts = df.groupby("user_id")["event_timestamp"].shift()
    out["seconds_since_last"] = (df["event_timestamp"] - prev_ts).dt.total_seconds()
    return out


# ── family 5: categorical encoding, done wrong then right ──────────────
def frequency_encode(df: pd.DataFrame, col: str) -> pd.Series:
    """Safe: depends only on the feature distribution, never on the label."""
    return df[col].map(df[col].value_counts(normalize=True))


def target_encode_naive(df: pd.DataFrame, col: str, target: str) -> pd.Series:
    """LEAKY: every row's encoding includes that row's own label."""
    means = df.groupby(col)[target].mean()
    return df[col].map(means)


def target_encode_in_fold(df: pd.DataFrame, col: str, target: str,
                          n_folds: int = 5, seed: int = 42) -> pd.Series:
    """Correct: a row is encoded using only the OTHER folds' labels."""
    if n_folds < 2:
        raise ValueError("in-fold encoding requires at least two folds")
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, size=len(df))
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        means = df.loc[tr].groupby(col)[target].mean()
        # Even the fallback must not use held-out labels. For an empty
        # training fold use a fixed, label-independent binary prior.
        prior = df.loc[tr, target].mean() if tr.any() else 0.5
        out.loc[te] = df.loc[te, col].map(means).fillna(prior).values
    return out


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC. Pure numpy -- no sklearn dependency."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties, or ties alone can fake a signal
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts));  np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ── the point-in-time trap ─────────────────────────────────────────────
def latest_join(entity_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """WRONG: attach each user's most recent feature value, ignoring time.

    This is what a `GROUP BY user_id` + `MAX(timestamp)` query gives you, and
    it silently pulls values recorded AFTER the label event.
    """
    latest = (feature_df.sort_values("event_timestamp")
              .groupby("user_id").tail(1)[["user_id", "feature_value"]])
    return entity_df.merge(latest, on="user_id", how="left")


def pit_join(entity_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """RIGHT: as-of join -- each row sees only values that already existed."""
    e = entity_df.sort_values("event_timestamp")
    f = feature_df.sort_values("event_timestamp")
    return pd.merge_asof(e, f[["user_id", "event_timestamp", "feature_value"]],
                         on="event_timestamp", by="user_id", direction="backward")


def leaked_row_fraction(entity_df: pd.DataFrame, feature_df: pd.DataFrame) -> float:
    """Share of training rows whose 'latest' feature was recorded in the future."""
    last_ts = feature_df.groupby("user_id")["event_timestamp"].max()
    mapped = entity_df["user_id"].map(last_ts)
    return float((mapped > entity_df["event_timestamp"]).mean())


def leakage_experiment(df: pd.DataFrame, col: str, target: str = "clicked",
                       test_frac: float = 0.3, seed: int = 7) -> pd.DataFrame:
    """Split FIRST, then encode -- the only way to see target-encoding leakage.

    The common mistake when *demonstrating* leakage is to encode the whole
    frame and then split: the holdout labels are already baked into the
    encoding, so both sides look equally good and the leak hides.

    Protocol here:
      split -> fit the encoder on TRAIN ONLY -> score train and test.
      A leaky encoder scores far better on rows it memorised than on new rows.
      Read the `gap` column: large gap = the feature is remembering, not learning.
    """
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(df)) < test_frac
    train, test = df.loc[~is_test].copy(), df.loc[is_test].copy()
    prior = train[target].mean()

    rows = []

    # (a) frequency encoding -- never touches the label
    freq = train[col].value_counts(normalize=True)
    rows.append(("frequency", auc(train[col].map(freq), train[target]),
                 auc(test[col].map(freq).fillna(0.0), test[target])))

    # (b) naive target encoding -- each train row's encoding contains its label
    means = train.groupby(col)[target].mean()
    rows.append(("target-naive", auc(train[col].map(means), train[target]),
                 auc(test[col].map(means).fillna(prior), test[target])))

    # (c) in-fold target encoding -- train rows encoded from OTHER folds only
    infold = target_encode_in_fold(train, col, target, seed=seed)
    rows.append(("target-in-fold", auc(infold, train[target]),
                 auc(test[col].map(means).fillna(prior), test[target])))

    out = pd.DataFrame(rows, columns=["encoding", "train_auc", "test_auc"])
    out["gap"] = out["train_auc"] - out["test_auc"]
    return out
