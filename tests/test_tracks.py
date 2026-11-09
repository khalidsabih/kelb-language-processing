from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from klp.data.candidates import apply_footage_filter, candidate_videos, normalize_questions, youtube_id
from klp.data.tracks import (
    build_windows,
    cap_windows_per_video,
    filter_interacting,
    load_tracks,
    move_onsets,
    prepare_tracks,
)
from klp.features.select import available_modalities

FIXTURE = Path(__file__).parent / "fixtures" / "tracks_two_dogs.csv"


@pytest.fixture
def tracks():
    # Dog a stands at the origin facing +x; dog b walks toward it along x at 1 unit/s.
    return load_tracks(FIXTURE)


def test_load_tracks_fills_defaults(tracks):
    assert set(tracks.dog_id) == {"v1:a", "v1:b"}
    assert (tracks.visibility == 1.0).all()
    assert tracks.audio_energy.isna().all()


def test_load_tracks_rejects_missing_columns(tracks):
    with pytest.raises(ValueError, match="heading"):
        prepare_tracks(tracks.drop(columns="heading"))


def test_windows_have_expected_count(tracks):
    # 7 frames at 0.5 s; 2 past + 2 future frames leaves anchors at frames 2, 3, 4; both directions.
    windows = build_windows(tracks, past_s=1.0, future_s=1.0, stride_s=0.5)
    assert len(windows) == 6
    assert sorted(windows.anchor_time_s.unique()) == [1.0, 1.5, 2.0]


def test_window_features_match_geometry(tracks):
    windows = build_windows(tracks, past_s=1.0, future_s=1.0, stride_s=0.5)
    row = windows[(windows.sender_id == "v1:a") & (windows.anchor_time_s == 1.0)].iloc[0]

    assert row.pair_distance == pytest.approx(9.0)
    assert row.receiver_speed_past == pytest.approx(1.0)
    assert row.receiver_distance_past == pytest.approx(-1.0)  # approaching the sender
    assert row.receiver_orientation_past == pytest.approx(1.0)  # facing the sender
    assert row.sender_speed == pytest.approx(0.0)
    assert row.sender_acceleration == pytest.approx(0.0)
    assert row.sender_facing_receiver == pytest.approx(1.0)
    assert row.visibility_score == pytest.approx(1.0)
    assert np.isnan(row.sender_audio_energy)

    assert row.delta_distance == pytest.approx(-1.0)
    assert row.delta_orientation == pytest.approx(0.0)
    assert row.delta_velocity == pytest.approx(0.0)


def test_targets_ignore_sender_future_movement(tracks):
    # Move dog a after t=1.0; b's targets at anchor 1.0 must not change.
    moved = tracks.copy()
    moved.loc[(moved.track_id == "a") & (moved.time_s > 1.0), "x"] = 5.0
    original = build_windows(tracks).set_index("sample_id")
    shifted = build_windows(prepare_tracks(moved)).set_index("sample_id")
    key = "v1|0|a->b|1.000"
    assert shifted.loc[key, "delta_distance"] == pytest.approx(original.loc[key, "delta_distance"])


def test_body_size_makes_features_scale_invariant(tracks):
    # Zooming the scene by 3x (positions and body sizes alike) must not change any feature.
    base = build_windows(prepare_tracks(tracks.assign(body_size=2.0)))
    zoomed = build_windows(prepare_tracks(tracks.assign(x=tracks.x * 3, y=tracks.y * 3, body_size=6.0)))
    numeric = base.select_dtypes("number").columns
    pd.testing.assert_frame_equal(base[numeric], zoomed[numeric])
    assert base.pair_distance.iat[0] == pytest.approx(9.0 / 2.0)  # 9 units = 4.5 body sizes


def test_delta_distance_ignores_sideways_movement(tracks):
    # Receiver 0.5 units from the sender steps 2 units sideways: straight-line distance
    # grows by ~1.56, but the receiver did not move away along the line to the sender.
    sideways = tracks.copy()
    b = sideways.track_id == "b"
    sideways.loc[b, "x"] = 0.5
    sideways.loc[b, "y"] = np.where(sideways.loc[b, "time_s"] > 1.0, 2.0, 0.0)
    row = build_windows(prepare_tracks(sideways)).set_index("sample_id").loc["v1|0|a->b|1.000"]
    assert row.delta_distance == pytest.approx(0.0)


def test_filter_interacting(tracks):
    windows = build_windows(tracks)  # one 3 s run; pair distance 9 down to 8 units at the anchors
    assert windows.run_s.tolist() == pytest.approx([3.0] * len(windows))
    assert len(filter_interacting(windows)) == len(windows)
    assert filter_interacting(windows, min_run_s=5.0).empty
    close = filter_interacting(windows, max_pair_distance=8.6)
    assert len(close) > 0 and (close.pair_distance <= 8.6).all()


def test_lag_shifts_target_window(tracks):
    # 7 frames at 0.5 s; 2 past + 1 lag + 2 future frames leaves anchors at frames 2 and 3.
    lagged = build_windows(tracks, lag_s=0.5)
    assert sorted(lagged.anchor_time_s.unique()) == [1.0, 1.5]
    row = lagged.set_index("sample_id").loc["v1|0|a->b|1.000"]
    assert row.delta_distance == pytest.approx(-1.0)  # b still walks 1 unit/s toward a
    pd.testing.assert_frame_equal(build_windows(tracks, lag_s=0.0), build_windows(tracks))


def test_move_onsets():
    # A dog standing still for 2 s, then walking 2 units/s, sampled at 10 fps.
    t = np.round(np.arange(0, 4, 0.1), 3)
    x = np.where(t < 2.0, 0.0, (t - 2.0) * 2.0)
    still_then_moving = prepare_tracks(pd.DataFrame({"video_id": "v", "track_id": "a", "time_s": t, "x": x, "y": 0.0, "heading": 0.0}))
    onsets = move_onsets(still_then_moving)
    assert len(onsets) == 1 and 2.0 < onsets.time_s.iat[0] <= 2.4
    always_moving = still_then_moving.assign(x=t * 2.0)
    assert move_onsets(always_moving).empty


def test_cap_windows_per_video():
    windows = pd.DataFrame({"video_id": ["big"] * 50 + ["small"] * 5, "value": range(55)})
    capped = cap_windows_per_video(windows, max_windows=10, seed=0)
    assert capped.video_id.value_counts().to_dict() == {"big": 10, "small": 5}
    pd.testing.assert_frame_equal(capped, cap_windows_per_video(windows, max_windows=10, seed=0))
    assert cap_windows_per_video(windows, max_windows=None) is windows


def test_windows_do_not_span_shots_or_gaps(tracks):
    shots = tracks.assign(shot_id=np.where(tracks.time_s < 1.5, "0", "1"))
    assert build_windows(prepare_tracks(shots)).empty

    gap = tracks[tracks.time_s != 1.5]
    assert build_windows(prepare_tracks(gap)).empty


def test_sparse_shot_does_not_break_frame_interval(tracks):
    # A later shot where both dogs are detected in only two frames, 0.8 s apart.
    sparse = pd.DataFrame([
        {"video_id": "v1", "track_id": tid, "time_s": t, "x": x, "y": 0.0, "heading": 0.0, "shot_id": "1"}
        for tid, x in (("a", 0.0), ("b", 5.0)) for t in (10.0, 10.8)
    ])
    windows = build_windows(prepare_tracks(pd.concat([tracks.assign(shot_id="0"), sparse])))
    assert len(windows) == 6  # the dense shot's windows; the sparse shot yields none


def test_available_modalities_skips_missing_measurements(tracks):
    windows = build_windows(tracks)
    assert available_modalities(windows) == ["motion", "orientation"]

    with_pose = build_windows(prepare_tracks(tracks.assign(pose_compactness=0.5, head_height=0.3)))
    assert "pose" in available_modalities(with_pose)


def test_candidate_videos_from_both_benchmarks():
    pawgaze = pd.DataFrame({
        "scene_name": ["abcdefghijk", "abcdefghijk", "zzzzzzzzzzz"],
        "question": ["How does the other dog react?", "What does the dog eat?", "Why does the dog bark?"],
    })
    k9bench = pd.DataFrame({
        "video_url": ["https://www.youtube.com/watch?v=ABCDEFGHIJK"],
        "question": ["What do both dogs do next?"],
    })
    questions = pd.concat([
        normalize_questions(pawgaze, "pawgaze"),
        normalize_questions(k9bench, "k9bench"),
    ])
    videos = candidate_videos(questions)
    assert set(videos.video_id) == {"abcdefghijk", "ABCDEFGHIJK"}
    assert youtube_id("https://youtu.be/ABCDEFGHIJK?t=3") == "ABCDEFGHIJK"


@pytest.mark.parametrize("question, tier", [
    ("How do both dogs play together in the snow?", "dog_dog"),
    ("Trace the interaction loop between the dog and the human.", "none"),
    ("What command makes the dogs sit?", "human_led"),
    ("How do the dogs behave at the beach?", "multi_dog"),
    ("How does the dog react to videos of other dogs on a laptop?", "none"),
])
def test_candidate_tiers(question, tier):
    questions = normalize_questions(pd.DataFrame({"scene_name": ["abcdefghijk"], "question": [question]}), "pawgaze")
    assert candidate_videos(questions, include_all=True).tier.iat[0] == tier


def test_footage_filter_flags_short_portrait_and_missing_videos():
    videos = pd.DataFrame({"video_id": ["long", "brief", "tall", "short", "gone", "unknown"]})
    metadata = pd.DataFrame({
        "video_id": ["long", "brief", "tall", "short", "gone"],
        "available": [True, True, True, True, False],
        "duration_s": [300, 60, 300, 300, None],
        "width": [854, 854, 480, 854, None],
        "height": [480, 480, 854, 480, None],
        "is_short": [False, False, False, True, None],
        "channel_id": ["c1", "c2", "c3", "c4", None],
    })
    issues = apply_footage_filter(videos, metadata).set_index("video_id").footage_issue
    assert pd.isna(issues["long"])
    assert issues.drop("long").to_dict() == {
        "brief": "too_short", "tall": "portrait", "short": "short", "gone": "unavailable", "unknown": "no_metadata",
    }
