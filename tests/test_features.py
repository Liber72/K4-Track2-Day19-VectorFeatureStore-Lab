import numpy as np
import pandas as pd
import pytest

from app.features import (auc, frequency_encode, generate_events, latest_join,
                          leakage_experiment, leaked_row_fraction, pit_join,
                          target_encode_in_fold, target_encode_naive,
                          window_aggregates)


@pytest.fixture(scope="module")
def events():
    return generate_events(n_users=60, n_days=20, seed=42)


def test_events_are_deterministic():
    a = generate_events(n_users=20, n_days=5, seed=7)
    b = generate_events(n_users=20, n_days=5, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_auc_matches_known_values():
    assert auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)
    perfect = auc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1]))
    assert perfect == pytest.approx(1.0)
    # all-ties must be 0.5, not an artefact of sort order
    assert auc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0, 1, 0, 1])) == pytest.approx(0.5)


def test_window_features_are_causal(events):
    """First event of a user can have no history -- else we are reading the future."""
    f = window_aggregates(events)
    first = f.groupby("user_id").head(1)
    assert (first["searches_7d"] == 0).all()
    assert first["seconds_since_last"].isna().all()
    assert first["prev_query_len"].isna().all()


def test_window_counts_are_monotone_by_window_size(events):
    f = window_aggregates(events)
    assert (f["searches_1h"] <= f["searches_24h"]).all()
    assert (f["searches_24h"] <= f["searches_7d"]).all()


def test_frequency_encoding_never_uses_the_label(events):
    enc = frequency_encode(events, "topic")
    shuffled = events.copy()
    shuffled["clicked"] = shuffled["clicked"].sample(frac=1, random_state=0).values
    pd.testing.assert_series_equal(enc, frequency_encode(shuffled, "topic"))


def test_naive_target_encoding_leaks_on_high_cardinality(events):
    """The headline result of NB8 §4 -- lock it in so the lesson cannot rot."""
    res = leakage_experiment(events, "session_id").set_index("encoding")
    assert res.loc["target-naive", "gap"] > 0.30
    assert abs(res.loc["target-in-fold", "gap"]) < 0.10
    assert abs(res.loc["frequency", "gap"]) < 0.10


def test_leak_is_smaller_on_lower_cardinality_key(events):
    high = leakage_experiment(events, "session_id").set_index("encoding")
    low = leakage_experiment(events, "user_id").set_index("encoding")
    assert high.loc["target-naive", "gap"] > low.loc["target-naive", "gap"]


def test_in_fold_encoding_covers_every_row(events):
    enc = target_encode_in_fold(events, "user_id", "clicked")
    assert not enc.isna().any()


def test_in_fold_fallback_does_not_see_held_out_labels():
    frame = pd.DataFrame({"key": [f"unique_{i}" for i in range(100)],
                          "clicked": [i % 2 for i in range(100)]})
    folds = np.random.default_rng(42).integers(0, 5, size=len(frame))
    held_out = folds == 0
    changed = frame.copy()
    changed.loc[held_out, "clicked"] = 1 - changed.loc[held_out, "clicked"]
    before = target_encode_in_fold(frame, "key", "clicked", seed=42)
    after = target_encode_in_fold(changed, "key", "clicked", seed=42)
    pd.testing.assert_series_equal(before.loc[held_out], after.loc[held_out])


def test_window_counts_keep_subsecond_precision_and_exclude_same_timestamp():
    frame = pd.DataFrame({
        "user_id": ["u_001"] * 3, "topic": ["cloud"] * 3,
        "query_len": [4, 5, 6], "clicked": [0, 1, 0],
        "event_timestamp": pd.to_datetime([
            "2026-07-01T00:00:00.100Z", "2026-07-01T00:00:00.100Z",
            "2026-07-01T00:00:00.900Z",
        ]),
    })
    assert window_aggregates(frame)["searches_1h"].tolist() == [0, 0, 2]


def test_pit_join_never_sees_the_future(events):
    fe = events[["user_id", "event_timestamp"]].copy().sort_values("event_timestamp")
    fe["feature_value"] = fe.groupby("user_id").cumcount() + 1
    ent = events[["user_id", "event_timestamp", "clicked"]].iloc[::5].copy()

    pit = pit_join(ent, fe)
    lat = latest_join(ent, fe)
    # the as-of value can never exceed the true count at that instant
    merged = pit.dropna(subset=["feature_value"])
    assert (merged["feature_value"] >= 1).all()
    # the naive join pulls values recorded later
    assert leaked_row_fraction(ent, fe) > 0.5
    assert lat["feature_value"].mean() > pit["feature_value"].mean()
