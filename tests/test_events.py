import numpy as np
import pandas as pd
import pytest

from klp.data.events import detect_events
from klp.data.schema import TARGET_COLUMNS
from klp.data.tracks import build_windows, prepare_tracks
from klp.experiments.events import event_response, flag_event_windows, holm


def _two_dogs(x_a, heading_a, t, heading_source="pose"):
    """Dog a moves along x toward dog b, which stands at x = 10; 10 fps."""
    a = pd.DataFrame({"track_id": "a", "time_s": t, "x": x_a, "y": 0.0, "heading": heading_a})
    b = pd.DataFrame({"track_id": "b", "time_s": t, "x": 10.0, "y": 0.0, "heading": np.pi})
    tracks = pd.concat([a, b]).assign(video_id="v", heading_source=heading_source)
    return prepare_tracks(tracks)


def test_detect_events_approach_and_turn():
    t = np.round(np.arange(0, 6, 0.1), 3)
    # a stands still facing away from b, turns toward it at 1.5 s, then walks toward it at 2 units/s from 3 s.
    x_a = np.where(t < 3.0, 0.0, (t - 3.0) * 2.0)
    heading_a = np.where(t < 1.5, np.pi, 0.0)
    events = detect_events(_two_dogs(x_a, heading_a, t))
    a_to_b = events[events.sender_id == "v:a"]
    assert set(a_to_b.kind) == {"move_onset", "approach", "turn_to"}
    for kind, start in [("move_onset", 3.0), ("approach", 3.0), ("turn_to", 1.5)]:
        time = a_to_b.loc[a_to_b.kind == kind, "time_s"]
        assert len(time) == 1 and start < time.iat[0] <= start + 0.4
    assert set(events[events.sender_id == "v:b"].kind) == set()  # b never moves or turns


def test_detect_events_ignores_motion_headings_for_turns():
    t = np.round(np.arange(0, 6, 0.1), 3)
    heading_a = np.where(t < 1.5, np.pi, 0.0)
    events = detect_events(_two_dogs(0.0, heading_a, t, heading_source="motion"))
    assert "turn_to" not in set(events.kind)


def _synthetic_windows(effect, n_videos=6, seed=0):
    """Windows of one pair per video; delta_velocity rises by ``effect`` after events every 10 s."""
    rng = np.random.default_rng(seed)
    rows, events = [], []
    for v in range(n_videos):
        for k in range(200):
            t = k * 0.5
            is_event = k % 20 == 10
            rows.append({"video_id": f"v{v}", "channel_id": f"c{v}", "sender_id": f"v{v}:a", "receiver_id": f"v{v}:b",
                         "anchor_time_s": t, "receiver_speed_past": rng.normal(), "receiver_orientation_past": rng.normal(),
                         "receiver_distance_past": rng.normal(), "pair_distance": rng.normal(), "visibility_score": 1.0,
                         "delta_distance": rng.normal(), "delta_orientation": rng.normal(),
                         "delta_velocity": rng.normal() + effect * is_event})
            if is_event:
                events.append({"video_id": f"v{v}", "channel_id": f"c{v}", "sender_id": f"v{v}:a",
                               "receiver_id": f"v{v}:b", "kind": "move_onset", "time_s": t})
    return pd.DataFrame(rows), pd.DataFrame(events)


def test_event_response_finds_effect_and_not_without():
    windows, events = _synthetic_windows(effect=1.5)
    res = event_response(windows, events, "move_onset", n_placebo=50, n_boot=200)
    assert res["events"] == 60 and res["channels"] == 6
    assert res["delta_velocity"] == pytest.approx(1.5, abs=0.4)
    assert res["delta_velocity_ci_low"] > 0 and res["delta_velocity_p"] < 0.05

    # With no effect the placebo p-values stay large (the channel bootstrap is too narrow with 6 channels).
    windows, events = _synthetic_windows(effect=0.0)
    res = event_response(windows, events, "move_onset", n_placebo=50, n_boot=200)
    assert all(res[f"{t}_p"] > 0.01 for t in TARGET_COLUMNS)
    assert all(abs(res[f"{t}_placebo_mean"]) < 0.2 for t in TARGET_COLUMNS)


def test_flag_event_windows_controls_and_simultaneous():
    windows, events = _synthetic_windows(effect=0.0, n_videos=1)
    # The receiver also had an event at the same moment as the sender's first one.
    events = pd.concat([events, events.iloc[:1].assign(sender_id="v0:b", receiver_id="v0:a")])
    flagged = flag_event_windows(windows, events, "move_onset", control_gap_s=3.0, simultaneous_s=0.8)
    assert flagged.is_event.sum() == len(events) - 2  # the shared first event is dropped
    # Controls are the windows at least 3 s from every event.
    gaps = np.abs(flagged.anchor_time_s.to_numpy()[:, None] - events.time_s.to_numpy()[None, :]).min(axis=1)
    assert (flagged.is_control == (gaps >= 3.0)).all()
    assert not (flagged.is_event & flagged.is_control).any()


def test_holm():
    p = pd.Series([0.01, 0.04, 0.03, 0.5])
    assert holm(p).tolist() == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_event_windows_line_up_with_detected_events():
    t = np.round(np.arange(0, 6, 0.1), 3)
    x_a = np.where(t < 3.0, 0.0, (t - 3.0) * 2.0)
    tracks = _two_dogs(x_a, 0.0, t)
    events = detect_events(tracks)
    windows = build_windows(tracks, stride_s=0.0)
    flagged = flag_event_windows(windows, events, "approach", control_gap_s=3.0, simultaneous_s=0.8)
    assert flagged.is_event.sum() == 1
    row = flagged[flagged.is_event].iloc[0]
    assert row.sender_id == "v:a" and row.sender_speed > 0


def test_event_kinds_match_schema():
    from klp.data.events import DEFAULT_KINDS
    from klp.data.schema import EVENT_KINDS, SENDER_FEATURES
    assert list(DEFAULT_KINDS) == EVENT_KINDS
    assert SENDER_FEATURES["events"] == [f"sender_event_{k}" for k in EVENT_KINDS]


def test_add_event_features_marks_past_window():
    from klp.data.events import add_event_features
    t = np.round(np.arange(0, 6, 0.1), 3)
    x_a = np.where(t < 3.0, 0.0, (t - 3.0) * 2.0)
    tracks = _two_dogs(x_a, 0.0, t)
    events = detect_events(tracks)
    onset = events.loc[(events.kind == "approach") & (events.sender_id == "v:a"), "time_s"].iat[0]
    windows = add_event_features(build_windows(tracks, stride_s=0.0), events, past_s=1.0)
    assert windows.sender_event_withdraw.eq(0).all()
    a_to_b = windows[windows.sender_id == "v:a"]
    marked = a_to_b.loc[a_to_b.sender_event_approach == 1, "anchor_time_s"]
    expected = a_to_b.anchor_time_s[(a_to_b.anchor_time_s >= onset - 1e-9) & (a_to_b.anchor_time_s < onset + 1.0 - 1e-9)]
    assert sorted(marked) == sorted(expected) and len(marked) > 0
    assert windows.loc[windows.sender_id == "v:b", "sender_event_approach"].eq(0).all()
