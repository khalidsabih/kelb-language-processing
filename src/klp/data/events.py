"""Candidate communication events: moments a sender abruptly changes what it does relative to a receiver."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from .schema import EVENT_KINDS
from .tracks import _contiguous_runs

EVENT_COLUMNS = ["video_id", "channel_id", "sender_id", "receiver_id", "kind", "time_s"]

# Default event kinds. Each is a transition in one per-frame sender signal: the mean over
# ``before_s`` and the mean over the following ``after_s`` must meet the bounds.
# Signals (speeds in body sizes/s):
# - ``speed``: sender speed;
# - ``approach``: sender velocity toward the receiver (negative = moving away);
# - ``facing``: cosine between the sender's pose body heading and the direction to the receiver.
DEFAULT_KINDS = {
    "move_onset": {"signal": "speed", "before_max": 0.3, "after_min": 1.0},
    "stop": {"signal": "speed", "before_min": 1.0, "after_max": 0.3},
    "approach": {"signal": "approach", "before_max": 0.3, "after_min": 1.0},
    "withdraw": {"signal": "approach", "before_min": -0.3, "after_max": -1.0},
    "turn_to": {"signal": "facing", "before_max": 0.5, "after_min": 0.87},
    "turn_away": {"signal": "facing", "before_min": 0.87, "after_max": 0.5},
}


def detect_events(
    tracks: pd.DataFrame,
    kinds: dict[str, dict] | None = None,
    before_s: float = 0.5,
    after_s: float = 0.3,
    refractory_s: float = 2.0,
    pose_heading_only: bool = True,
) -> pd.DataFrame:
    """Sender events for every ordered pair of dogs tracked together, one row per event.

    An event at frame ``k`` means the transition ended at ``k``: the sender's change is
    in the past window of the window anchored at ``k``, and the receiver's response is
    measured after it. Only frames where both dogs are tracked without a gap count, so
    every event has a receiver. Events of one kind closer than ``refractory_s`` to an
    earlier one of the same pair are dropped.

    With ``pose_heading_only``, the ``facing`` signal uses only headings from pose
    (``heading_source == "pose"``): a heading taken from the motion direction turns
    whenever the dog starts walking, which would make every move onset a turn.
    """
    kinds = DEFAULT_KINDS if kinds is None else kinds
    rows = []
    video_dt = tracks.groupby("video_id").time_s.agg(
        lambda t: float(np.median(np.diff(np.sort(t.unique())))) if t.nunique() > 1 else np.nan
    )
    for (video_id, _), shot in tracks.groupby(["video_id", "shot_id"], sort=False):
        dt = video_dt[video_id]
        n_before, n_after = max(1, round(before_s / dt)), max(1, round(after_s / dt))
        by_track = {tid: g.set_index("time_s") for tid, g in shot.groupby("track_id")}
        for sender, receiver in permutations(by_track, 2):
            a, b = by_track[sender], by_track[receiver]
            ids = {"video_id": video_id, "channel_id": a.channel_id.iat[0],
                   "sender_id": a.dog_id.iat[0], "receiver_id": b.dog_id.iat[0]}
            for run in _contiguous_runs(np.intersect1d(a.index, b.index), dt):
                if len(run) <= n_before + n_after:
                    continue
                signals = _pair_signals(a.loc[run], b.loc[run], dt, pose_heading_only)
                for kind, spec in kinds.items():
                    for k in _transitions(signals[spec["signal"]], spec, n_before, n_after, run, refractory_s):
                        rows.append({**ids, "kind": kind, "time_s": float(run[k])})
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def add_event_features(windows: pd.DataFrame, events: pd.DataFrame, past_s: float) -> pd.DataFrame:
    """Fill ``sender_event_<kind>``: 1.0 if the sender had that event toward this receiver in the past window.

    The past window is ``(anchor_time_s - past_s, anchor_time_s]``, so an event at the
    anchor counts and the response is measured after it.
    """
    windows = windows.copy()
    keys = list(zip(windows.video_id, windows.sender_id, windows.receiver_id))
    t = windows.anchor_time_s.to_numpy()
    for kind in EVENT_KINDS:
        times = events[events.kind == kind].groupby(["video_id", "sender_id", "receiver_id"]).time_s.apply(
            lambda s: np.sort(s.to_numpy())).to_dict()
        flag = np.zeros(len(windows))
        for i, (key, anchor) in enumerate(zip(keys, t)):
            d = times.get(key)
            if d is not None:
                # Number of events at or before the anchor minus those at or before anchor - past_s.
                flag[i] = float(np.searchsorted(d, anchor + 1e-9) > np.searchsorted(d, anchor - past_s + 1e-9))
        windows[f"sender_event_{kind}"] = flag
    return windows


def _pair_signals(a: pd.DataFrame, b: pd.DataFrame, dt: float, pose_heading_only: bool) -> dict[str, np.ndarray]:
    pa, pb = a[["x", "y"]].to_numpy(float), b[["x", "y"]].to_numpy(float)
    size = a.body_size.to_numpy(float)
    step = np.diff(pa, axis=0) / dt  # step k is the move from frame k to k+1, reported at k+1
    to_b = pb - pa
    unit = to_b / np.maximum(np.linalg.norm(to_b, axis=1, keepdims=True), 1e-9)
    heading = a.heading.to_numpy(float)
    if pose_heading_only and "heading_source" in a:
        heading = np.where(a.heading_source.to_numpy() == "pose", heading, np.nan)
    return {
        "speed": np.r_[np.nan, np.linalg.norm(step, axis=1) / size[1:]],
        "approach": np.r_[np.nan, (step * unit[:-1]).sum(axis=1) / size[1:]],
        "facing": np.cos(heading - np.arctan2(to_b[:, 1], to_b[:, 0])),
    }


def _transitions(x, spec, n_before, n_after, times, refractory_s) -> list[int]:
    bounds = [(key, spec[key]) for key in ("before_min", "before_max", "after_min", "after_max") if key in spec]
    out, last = [], -np.inf
    for k in range(n_before + n_after, len(x)):
        before = x[k - n_after - n_before + 1:k - n_after + 1]
        after = x[k - n_after + 1:k + 1]
        if np.isnan(before).any() or np.isnan(after).any():
            continue
        means = {"before": before.mean(), "after": after.mean()}
        ok = all(means[key.split("_")[0]] >= v if key.endswith("min") else means[key.split("_")[0]] <= v
                 for key, v in bounds)
        if ok and times[k] - last >= refractory_s:
            out.append(k)
            last = times[k]
    return out
