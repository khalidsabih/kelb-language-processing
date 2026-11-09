import numpy as np
import pandas as pd
import pytest

from klp.data.schema import ALL_SENDER_FEATURES, CONTEXT_FEATURES, DYADIC_FEATURES, TARGET_COLUMNS
from klp.data.splits import dog_disjoint_split, split_stats
from klp.data.tracks import build_windows, prepare_tracks
from klp.evaluation.metrics import regression_metrics
from klp.experiments.core import null_sender, repeated_split_experiments, sender_null_experiment
from klp.features.select import baseline_features


def _pairs(n_videos: int, shared_dogs: bool = False) -> pd.DataFrame:
    """One sender/receiver row per video; optionally reuse dogs across videos."""
    dogs = [(f"d{i % 4}", f"d{(i + 1) % 4}") if shared_dogs else (f"v{i}:a", f"v{i}:b") for i in range(n_videos)]
    return pd.DataFrame({
        "video_id": [f"v{i}" for i in range(n_videos)],
        "sender_id": [s for s, _ in dogs],
        "receiver_id": [r for _, r in dogs],
    })


def _toy_tracks(n_videos: int = 12, n_frames: int = 40, dt: float = 0.25) -> pd.DataFrame:
    """Test-only tracks: a wandering sender and a receiver that drifts toward it."""
    rng = np.random.default_rng(0)
    rows = []
    for v in range(n_videos):
        a = np.cumsum(rng.normal(scale=0.3, size=(n_frames, 2)), axis=0)
        b = np.empty_like(a)
        b[0] = a[0] + 5.0
        for t in range(1, n_frames):
            b[t] = b[t - 1] + 0.2 * (a[t - 1] - b[t - 1]) + rng.normal(scale=0.1, size=2)
        for track, pos in (("a", a), ("b", b)):
            heading = np.arctan2(*np.gradient(pos, axis=0)[:, ::-1].T)
            for t in range(n_frames):
                rows.append({
                    "video_id": f"v{v}", "track_id": track, "time_s": t * dt,
                    "x": pos[t, 0], "y": pos[t, 1], "heading": heading[t],
                })
    return prepare_tracks(pd.DataFrame(rows))


def test_component_split_keeps_every_row():
    df = _pairs(12)
    train, test = dog_disjoint_split(df, test_fraction=0.25, seed=1)
    stats = split_stats(df, train, test)
    assert stats["n_dropped"] == 0
    assert (set(train.sender_id) | set(train.receiver_id)).isdisjoint(set(test.sender_id) | set(test.receiver_id))


def test_component_split_needs_more_than_one_component():
    with pytest.raises(ValueError, match="unit='dog'"):
        dog_disjoint_split(_pairs(12, shared_dogs=True), unit="component")


def test_video_split_keeps_track_fragments_together():
    # Each video holds two fragment pairs that never co-occur, as when a tracker loses dogs.
    df = pd.DataFrame([
        {"video_id": f"v{i}", "sender_id": f"v{i}:{s}", "receiver_id": f"v{i}:{r}"}
        for i in range(12) for s, r in (("a", "b"), ("c", "d"))
    ])
    with pytest.raises(ValueError, match="unit='video'"):
        for seed in range(10):
            dog_disjoint_split(df, seed=seed, unit="component")

    for seed in range(10):
        train, test = dog_disjoint_split(df, seed=seed, unit="video")
        assert set(train.video_id).isdisjoint(test.video_id)
        assert len(train) + len(test) == len(df)


def test_channel_split_keeps_channels_together():
    # 6 channels x 3 videos; dog IDs are per video, as when identity across videos is unknown.
    df = pd.DataFrame([
        {"video_id": f"c{c}v{v}", "channel_id": f"c{c}", "sender_id": f"c{c}v{v}:a", "receiver_id": f"c{c}v{v}:b"}
        for c in range(6) for v in range(3)
    ])
    split_channels = []
    for seed in range(10):
        train, test = dog_disjoint_split(df, seed=seed, unit="channel")
        assert set(train.channel_id).isdisjoint(test.channel_id)
        _, video_test = dog_disjoint_split(df, seed=seed, unit="video")
        split_channels.append(video_test.groupby("channel_id").size().lt(3).any())
    assert any(split_channels)  # the video split does separate a channel's videos

    with pytest.raises(ValueError, match="channel_id"):
        dog_disjoint_split(df.drop(columns="channel_id"), unit="channel")


def test_dog_split_is_identity_disjoint_and_counts_drops():
    pairs = [(f"d{i}", f"d{j}") for i in range(8) for j in range(8) if i != j]
    df = pd.DataFrame({
        "video_id": [f"v{k}" for k in range(len(pairs))],
        "sender_id": [s for s, _ in pairs],
        "receiver_id": [r for _, r in pairs],
    })
    train, test = dog_disjoint_split(df, test_fraction=0.5, seed=0, unit="dog")
    stats = split_stats(df, train, test)
    assert stats["n_train"] + stats["n_test"] + stats["n_dropped"] == len(df)
    assert (set(train.sender_id) | set(train.receiver_id)).isdisjoint(set(test.sender_id) | set(test.receiver_id))


def test_split_rejects_video_shared_across_partitions():
    # Two disjoint dog pairs recorded in the same video.
    df = pd.DataFrame({
        "video_id": ["v0", "v0"] + [f"v{i}" for i in range(1, 9)],
        "sender_id": ["a", "c"] + ["a", "c"] * 4,
        "receiver_id": ["b", "d"] + ["b", "d"] * 4,
    })
    # Only seeds that separate {a, b} from {c, d} avoid an empty partition;
    # every one of those must hit the shared-video check.
    errors = []
    for seed in range(20):
        with pytest.raises(ValueError) as exc:
            dog_disjoint_split(df, test_fraction=0.5, seed=seed, unit="dog")
        errors.append(str(exc.value))
    assert any("video" in e for e in errors)


def test_baseline_has_no_sender_signals():
    features = baseline_features()
    assert set(DYADIC_FEATURES + CONTEXT_FEATURES) <= set(features)
    assert set(features).isdisjoint(ALL_SENDER_FEATURES)


def test_metrics_report_per_target_and_mean():
    rng = np.random.default_rng(0)
    y = pd.DataFrame(rng.normal(size=(50, 3)), columns=TARGET_COLUMNS)
    metrics = regression_metrics(y, y.to_numpy() + 0.1)
    for target in TARGET_COLUMNS:
        assert {f"mae_{target}", f"rmse_{target}", f"r2_{target}"} <= metrics.keys()
    assert metrics["mae"] == pytest.approx(0.1)


def test_experiments_run_end_to_end_on_tracks():
    windows = build_windows(_toy_tracks())
    modalities = ["motion", "orientation"]
    results = repeated_split_experiments(windows, n_splits=3, modalities=modalities)

    assert len(results["increment"]) == 3
    assert np.isfinite(results["increment"].sender_increment_r2).all()
    assert set(results["ablation"].dropped_modality) == set(modalities)
    assert (results["splits"].n_dropped == 0).all()


def test_null_sender_breaks_alignment_but_keeps_values():
    windows = build_windows(_toy_tracks(n_frames=80))
    cols = ["sender_speed", "sender_facing_receiver"]
    rng = np.random.default_rng(0)

    shuffled = null_sender(windows, cols, "shuffle", rng)
    for video, group in windows.groupby("video_id"):
        assert sorted(group.sender_speed) == sorted(shuffled.loc[group.index, "sender_speed"])
    assert not np.allclose(shuffled.sender_speed, windows.sender_speed)

    shifted = null_sender(windows, cols, "time_shift", rng, min_shift_s=5.0)
    lookup = windows.set_index(["video_id", "sender_id", "receiver_id", "sender_speed"]).anchor_time_s
    for row in shifted.dropna(subset=cols).itertuples():
        donor_time = lookup.loc[(row.video_id, row.sender_id, row.receiver_id, row.sender_speed)]
        assert np.min(np.abs(np.atleast_1d(donor_time) - row.anchor_time_s)) >= 5.0
    assert shifted[cols].notna().all(axis=1).any()


def test_sender_null_experiment_runs():
    windows = build_windows(_toy_tracks(n_frames=80))
    result = sender_null_experiment(windows, "time_shift", n_null=3, n_splits=2, modalities=["motion"])
    assert result["n_rows"] > 0
    assert np.isfinite(result["real_increment_r2"]) and 0 < result["p_value"] <= 1
