"""Turn per-frame dog tracks into sender-receiver interaction windows."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    ALL_SENDER_FEATURES,
    CONTEXT_FEATURES,
    DYADIC_FEATURES,
    ID_COLUMNS,
    RECEIVER_PAST_FEATURES,
    REQUIRED_TRACK_COLUMNS,
    TARGET_COLUMNS,
    TRACK_MEASUREMENT_FEATURES,
)

# Window features measured in position units; divided by body size when available.
LENGTH_FEATURES = [
    "receiver_speed_past",
    "receiver_distance_past",
    "pair_distance",
    "sender_speed",
    "sender_acceleration",
    "delta_distance",
    "delta_velocity",
]

WINDOW_COLUMNS = (
    ID_COLUMNS
    + ["anchor_time_s", "run_s"]
    + RECEIVER_PAST_FEATURES
    + DYADIC_FEATURES
    + CONTEXT_FEATURES
    + ALL_SENDER_FEATURES
    + TARGET_COLUMNS
)


def load_tracks(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet track file and validate it with ``prepare_tracks``."""
    path = Path(path)
    tracks = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return prepare_tracks(tracks)


def prepare_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    """Validate tracks and fill optional columns.

    Defaults: ``dog_id`` = ``"<video_id>:<track_id>"`` (identity unknown across
    videos), ``channel_id`` = ``video_id`` (each video its own channel), ``shot_id`` = one shot per video, ``source`` = ``"unknown"``,
    ``visibility`` = 1.0, ``body_size`` = 1.0 (no scaling), measurement columns = NaN.
    """
    missing = set(REQUIRED_TRACK_COLUMNS) - set(tracks.columns)
    if missing:
        raise ValueError(f"Track file is missing required columns: {sorted(missing)}")

    tracks = tracks.copy()
    tracks["video_id"] = tracks["video_id"].astype(str)
    tracks["track_id"] = tracks["track_id"].astype(str)
    if "dog_id" not in tracks:
        tracks["dog_id"] = tracks["video_id"] + ":" + tracks["track_id"]
    if "channel_id" not in tracks:
        tracks["channel_id"] = tracks["video_id"]
    tracks["channel_id"] = tracks["channel_id"].fillna(tracks["video_id"]).astype(str)
    defaults = {"shot_id": "0", "source": "unknown", "visibility": 1.0, "body_size": 1.0}
    defaults.update({col: np.nan for col in TRACK_MEASUREMENT_FEATURES})
    for col, value in defaults.items():
        if col not in tracks:
            tracks[col] = value

    key = ["video_id", "shot_id", "track_id", "time_s"]
    if tracks.duplicated(key).any():
        raise ValueError("Tracks contain duplicate (video_id, shot_id, track_id, time_s) rows.")
    if (tracks.groupby(["video_id", "track_id"]).dog_id.nunique() > 1).any():
        raise ValueError("A track maps to more than one dog_id.")
    if (tracks.groupby("video_id").channel_id.nunique() > 1).any():
        raise ValueError("A video maps to more than one channel_id.")

    return tracks.sort_values(key).reset_index(drop=True)


def build_windows(
    tracks: pd.DataFrame,
    past_s: float = 1.0,
    future_s: float = 1.0,
    stride_s: float = 0.5,
    lag_s: float = 0.0,
) -> pd.DataFrame:
    """Build one row per ordered dog pair and anchor time.

    The past window ends at the anchor; the target window of ``future_s`` starts
    ``lag_s`` later, so a response that comes with a delay can be measured.

    Windows never span a shot boundary or a gap in either dog's track. Every
    ordered pair in a shot is used, so each dog is both sender and receiver.

    Length features (``LENGTH_FEATURES``) are divided by the two dogs' mean
    ``body_size`` over the past window, so they are in body sizes rather than
    pixels and comparable across videos and camera distances.

    Targets are measured against the sender's position at the anchor time, so the
    sender's own future movement does not leak into the receiver's targets:

    - ``delta_distance``: receiver displacement along the line from that position
      to the receiver (positive = away). Sideways movement does not count, so a
      close receiver stepping past the sender is not scored as retreating; this
      is the same projection as ``receiver_distance_past``.
    - ``delta_orientation``: change in how directly the receiver faces it (cosine).
    - ``delta_velocity``: receiver mean speed, future minus past window.
    """
    rows = []
    # The frame interval is estimated per video: a shot with only a few frames in
    # which dogs were detected would otherwise look sparsely sampled.
    video_dt = tracks.groupby("video_id").time_s.agg(
        lambda t: float(np.median(np.diff(np.sort(t.unique())))) if t.nunique() > 1 else np.nan
    )
    for (video_id, shot_id), shot in tracks.groupby(["video_id", "shot_id"], sort=False):
        if shot.time_s.nunique() < 2:
            continue
        dt = video_dt[video_id]
        n_past = round(past_s / dt)
        n_future = max(1, round(future_s / dt))
        n_lag = round(lag_s / dt)
        stride = max(1, round(stride_s / dt))
        if n_past < 2:
            raise ValueError(f"past_s={past_s} spans fewer than 2 frames in video {video_id}.")

        by_track = {tid: g.set_index("time_s") for tid, g in shot.groupby("track_id")}
        for sender, receiver in permutations(by_track, 2):
            a_track, b_track = by_track[sender], by_track[receiver]
            if a_track.dog_id.iat[0] == b_track.dog_id.iat[0]:
                raise ValueError(f"Two tracks in video {video_id} share dog_id {a_track.dog_id.iat[0]}.")
            common = np.intersect1d(a_track.index, b_track.index)
            for run in _contiguous_runs(common, dt):
                a, b = _arrays(a_track.loc[run], dt), _arrays(b_track.loc[run], dt)
                for i in range(n_past, len(run) - n_lag - n_future, stride):
                    rows.append({
                        "sample_id": f"{video_id}|{shot_id}|{sender}->{receiver}|{run[i]:.3f}",
                        "video_id": video_id,
                        "channel_id": a_track.channel_id.iat[0],
                        "sender_id": a_track.dog_id.iat[0],
                        "receiver_id": b_track.dog_id.iat[0],
                        "source": a_track.source.iat[0],
                        "anchor_time_s": float(run[i]),
                        # How long both dogs stay tracked together around this window.
                        "run_s": float(run[-1] - run[0]),
                        **_window_features(a, b, i, n_past, n_future, dt, n_lag),
                    })

    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def cap_windows_per_video(windows: pd.DataFrame, max_windows: int | None, seed: int = 42) -> pd.DataFrame:
    """Keep at most ``max_windows`` randomly chosen windows per video (``None`` = no cap).

    Stops a few long or many-dog videos from dominating training and evaluation.
    """
    if max_windows is None:
        return windows
    shuffled = windows.sample(frac=1.0, random_state=seed)
    kept = shuffled[shuffled.groupby("video_id").cumcount() < max_windows]
    return kept.sort_index().reset_index(drop=True)


def move_onsets(
    tracks: pd.DataFrame,
    still_speed: float = 0.3,
    moving_speed: float = 1.0,
    still_s: float = 0.5,
    burst_s: float = 0.3,
    refractory_s: float = 2.0,
) -> pd.DataFrame:
    """Times at which a still dog starts moving, one row per onset (``dog_id``, ``video_id``, ``time_s``).

    Speeds are in body sizes per second (position units when ``body_size`` is 1).
    An onset is a frame where the mean speed over the preceding ``burst_s`` exceeds
    ``moving_speed`` and the mean over the ``still_s`` before that is below
    ``still_speed``, with no track gap in that span. Onsets closer than
    ``refractory_s`` to an earlier onset of the same dog are dropped.
    """
    rows = []
    for (video_id, _, _), track in tracks.groupby(["video_id", "shot_id", "track_id"], sort=False):
        t = track.time_s.to_numpy(float)
        if len(t) < 3:
            continue
        dt = float(np.median(np.diff(t)))
        pos = track[["x", "y"]].to_numpy(float)
        speed = np.r_[np.nan, np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(t) / track.body_size.to_numpy()[1:]]
        n_burst, n_still = max(1, round(burst_s / dt)), max(1, round(still_s / dt))
        last = -np.inf
        for k in range(n_still + n_burst, len(t)):
            span = slice(k - n_still - n_burst, k + 1)
            if np.diff(t[span]).max() > 1.5 * dt:
                continue
            burst = speed[k - n_burst + 1:k + 1].mean()
            still = speed[k - n_burst - n_still + 1:k - n_burst + 1].mean()
            if burst > moving_speed and still < still_speed and t[k] - last >= refractory_s:
                rows.append({"video_id": video_id, "dog_id": track.dog_id.iat[0], "time_s": float(t[k])})
                last = t[k]
    return pd.DataFrame(rows, columns=["video_id", "dog_id", "time_s"])


def filter_interacting(
    windows: pd.DataFrame, min_run_s: float | None = None, max_pair_distance: float | None = None
) -> pd.DataFrame:
    """Keep windows from continuously tracked pairs that are close enough to interact.

    ``min_run_s``: both dogs tracked together without a gap for at least this long
    (``run_s``), which drops fragments where the tracker briefly sees a second dog.
    ``max_pair_distance``: pair distance at the anchor time, in the same units as
    ``pair_distance`` (body sizes when ``body_size`` is given). ``None`` disables a filter.
    """
    keep = pd.Series(True, index=windows.index)
    if min_run_s is not None:
        keep &= windows.run_s >= min_run_s
    if max_pair_distance is not None:
        keep &= windows.pair_distance <= max_pair_distance
    return windows[keep].reset_index(drop=True)


def _contiguous_runs(times: np.ndarray, dt: float) -> list[np.ndarray]:
    breaks = np.flatnonzero(np.diff(times) > 1.5 * dt) + 1
    return np.split(times, breaks)


def _arrays(track: pd.DataFrame, dt: float) -> dict[str, np.ndarray]:
    pos = track[["x", "y"]].to_numpy(float)
    vel = np.diff(pos, axis=0) / dt  # vel[k] is the step from frame k to k+1
    arrays = {
        "pos": pos,
        "vel": vel,
        "speed": np.linalg.norm(vel, axis=1),
        "heading": track.heading.to_numpy(float),
        "visibility": track.visibility.to_numpy(float),
        "body_size": track.body_size.to_numpy(float),
    }
    arrays.update({col: track[col].to_numpy(float) for col in TRACK_MEASUREMENT_FEATURES})
    return arrays


def _facing(heading, from_pos, to_pos):
    """Cosine between a heading and the direction from ``from_pos`` to ``to_pos``."""
    d = np.asarray(to_pos) - np.asarray(from_pos)
    return np.cos(heading - np.arctan2(d[..., 1], d[..., 0]))


def _window_features(a, b, i, n_past, n_future, dt, n_lag=0) -> dict[str, float]:
    past = slice(i - n_past, i + 1)  # frames, anchor included
    past_steps = slice(i - n_past, i)
    start = i + n_lag  # target window: start .. end
    future_steps = slice(start, start + n_future)
    end = start + n_future

    away = b["pos"][past_steps] - a["pos"][past_steps]
    away_unit = away / np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-9)
    receiver_speed_past = b["speed"][past_steps].mean()

    features = {
        "receiver_speed_past": receiver_speed_past,
        "receiver_orientation_past": _facing(b["heading"][past], b["pos"][past], a["pos"][past]).mean(),
        # Rate at which the receiver's own movement changes the distance (positive = away).
        "receiver_distance_past": (b["vel"][past_steps] * away_unit).sum(axis=1).mean(),
        "pair_distance": np.linalg.norm(b["pos"][i] - a["pos"][i]),
        "visibility_score": np.minimum(a["visibility"][past], b["visibility"][past]).mean(),
        "sender_speed": a["speed"][past_steps].mean(),
        "sender_acceleration": np.diff(a["speed"][past_steps]).mean() / dt,
        "sender_facing_receiver": _facing(a["heading"][past], a["pos"][past], b["pos"][past]).mean(),
    }
    for col, feature in TRACK_MEASUREMENT_FEATURES.items():
        features[feature] = a[col][past].mean()

    anchor = a["pos"][i]
    away_now = b["pos"][start] - anchor
    features["delta_distance"] = (b["pos"][end] - b["pos"][start]) @ (away_now / max(np.linalg.norm(away_now), 1e-9))
    features["delta_orientation"] = (
        _facing(b["heading"][end], b["pos"][end], anchor) - _facing(b["heading"][start], b["pos"][start], anchor)
    )
    features["delta_velocity"] = b["speed"][future_steps].mean() - receiver_speed_past

    scale = np.concatenate([a["body_size"][past], b["body_size"][past]]).mean()
    for key in LENGTH_FEATURES:
        features[key] /= scale
    return {key: float(value) for key, value in features.items()}
