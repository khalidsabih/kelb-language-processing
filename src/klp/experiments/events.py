"""Receiver responses after sender events, compared with matched control moments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from klp.data.schema import TARGET_COLUMNS
from klp.features.select import baseline_features


def _pair_times(events: pd.DataFrame) -> dict[tuple, np.ndarray]:
    return events.groupby(["video_id", "sender_id", "receiver_id"]).time_s.apply(np.asarray).to_dict()


def _near(times_by_key: dict, keys, t, gap: float, before_only: bool = False) -> np.ndarray:
    """Whether any time stored under each row's key is within ``gap`` of the row's time.

    With ``before_only``, only stored times at or before the row's time count.
    """
    out = np.zeros(len(t), dtype=bool)
    for i, (key, ti) in enumerate(zip(keys, t)):
        d = times_by_key.get(key)
        if d is None:
            continue
        lag = ti - d
        out[i] = np.any((lag >= -1e-9) & (lag <= gap)) if before_only else np.any(np.abs(lag) < gap)
    return out


def flag_event_windows(
    windows: pd.DataFrame, events: pd.DataFrame, kind: str, control_gap_s: float, simultaneous_s: float
) -> pd.DataFrame:
    """Add ``is_event`` and ``is_control`` for one event kind.

    Event windows are anchored at an event of that pair. Events are dropped when the
    receiver had an event of the same kind (toward any dog) in the ``simultaneous_s``
    before: two dogs changing together point to a shared cause (a person, a noise), not
    a response. Control windows are at least ``control_gap_s`` from every event of that
    kind in the pair.
    """
    ev = events[events.kind == kind]
    pair_keys = list(zip(windows.video_id, windows.sender_id, windows.receiver_id))
    t = windows.anchor_time_s.to_numpy()
    at_event = _near(_pair_times(ev), pair_keys, t, 1e-6)
    by_dog = ev.groupby(["video_id", "sender_id"]).time_s.apply(np.asarray).to_dict()
    receiver_started = _near(by_dog, list(zip(windows.video_id, windows.receiver_id)), t, simultaneous_s, before_only=True)
    return windows.assign(
        is_event=at_event & ~receiver_started,
        is_control=~_near(_pair_times(ev), pair_keys, t, control_gap_s),
    )


class Matcher:
    """Matched-control differences for windows flagged with ``is_control``.

    For a window, the controls are the ``n_controls`` nearest control windows of the
    same video in the receiver's own recent behavior and the pair distance
    (``features``, divided by their SD), so they are moments where the receiver was in
    a similar state but the sender had no event. The difference is the window's targets
    minus the controls' mean, bias-corrected for the remaining mismatch: a linear model
    fitted on all controls predicts how much of the difference the mismatch alone
    explains, and that is subtracted. A window is never its own control.
    """

    def __init__(self, windows: pd.DataFrame, features: list[str], n_controls: int = 5):
        self.windows, self.n_controls = windows, n_controls
        self.X = ((windows[features] - windows[features].mean()) / windows[features].std().replace(0, 1.0)).to_numpy()
        self.target = windows[TARGET_COLUMNS].to_numpy()
        self.features = features
        control = windows.is_control.to_numpy()
        design = np.c_[np.ones(control.sum()), self.X[control]]
        self.coef = np.linalg.lstsq(design, self.target[control], rcond=None)[0][1:]
        self.pools = {}
        for video_id, pool in windows[control].groupby("video_id").indices.items():
            pool = np.flatnonzero(control)[pool]
            if len(pool) > n_controls:
                self.pools[video_id] = (pool, NearestNeighbors(n_neighbors=n_controls + 1).fit(self.X[pool]))

    def differences(self, rows: np.ndarray, labels: np.ndarray | None = None) -> pd.DataFrame:
        """One row per matched window; ``labels`` (aligned with ``rows``) are carried along as ``label``."""
        labels = np.zeros(len(rows), dtype=int) if labels is None else np.asarray(labels)
        out = []
        for video_id, idx in self.windows.iloc[rows].groupby("video_id").indices.items():
            if video_id not in self.pools:
                continue
            query = rows[idx]
            pool, nn = self.pools[video_id]
            neighbors = pool[nn.kneighbors(self.X[query], return_distance=False)]
            # Drop the window itself if it is in the pool, keeping n_controls neighbors.
            neighbors = np.array([[c for c in row if c != q][:self.n_controls] for row, q in zip(neighbors, query)])
            balance = self.X[query] - self.X[neighbors].mean(axis=1)
            diff = self.target[query] - self.target[neighbors].mean(axis=1) - balance @ self.coef
            out.append(pd.DataFrame({
                "video_id": video_id, "channel_id": self.windows.channel_id.iat[query[0]], "label": labels[idx],
                **{t: diff[:, j] for j, t in enumerate(TARGET_COLUMNS)},
                **{f"balance_{f}": balance[:, j] for j, f in enumerate(self.features)},
            }))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def event_response(
    windows: pd.DataFrame,
    events: pd.DataFrame,
    kind: str,
    n_controls: int = 5,
    control_gap_s: float = 3.0,
    simultaneous_s: float = 0.8,
    n_placebo: int = 2000,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Mean receiver response after ``kind`` events relative to matched controls.

    Per target: the event-weighted mean difference, a 95% interval that resamples whole
    channels, and a placebo p-value. Placebos re-run the same matching from a random
    control moment of each event's pair (or its video, if the pair has none), so they
    show what the procedure gives when nothing happened. ``_p`` is the share of placebo
    means at least as far from the placebo average as the real one (two-sided), so a
    bias of the matching itself does not count as an effect.
    """
    windows = flag_event_windows(windows, events, kind, control_gap_s, simultaneous_s).reset_index(drop=True)
    matcher = Matcher(windows, baseline_features(), n_controls)
    rows = np.flatnonzero(windows.is_event.to_numpy())
    real = matcher.differences(rows)
    result = {"kind": kind, "events": len(real)}
    if real.empty:
        return result
    result.update(videos=real.video_id.nunique(), channels=real.channel_id.nunique(),
                  max_abs_balance=float(real.filter(like="balance_").mean().abs().max()))

    rng = np.random.default_rng(seed)
    point = real[TARGET_COLUMNS].mean()
    per_channel = real.groupby("channel_id")[TARGET_COLUMNS].agg(["sum", "size"])
    channels = per_channel.index.to_numpy()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(channels), len(channels), replace=True)
        sub = per_channel.iloc[pick]
        boots.append({t: sub[(t, "sum")].sum() / sub[(t, "size")].sum() for t in TARGET_COLUMNS})
    boots = pd.DataFrame(boots)

    controls = windows[windows.is_control]
    pair_pool = controls.groupby(["video_id", "sender_id", "receiver_id"]).indices
    video_pool = controls.groupby("video_id").indices
    control_rows = np.flatnonzero(windows.is_control.to_numpy())
    events_used = windows.iloc[rows][windows.iloc[rows].video_id.isin(real.video_id.unique())]
    pools = [pair_pool.get((v, s, r), video_pool[v])
             for v, s, r in zip(events_used.video_id, events_used.sender_id, events_used.receiver_id)]
    picks = np.array([[control_rows[rng.choice(pool)] for pool in pools] for _ in range(n_placebo)])
    draws = np.repeat(np.arange(n_placebo), len(pools))
    placebo = matcher.differences(picks.ravel(), draws).groupby("label")[TARGET_COLUMNS].mean()

    sd = windows.loc[windows.is_control, TARGET_COLUMNS].std()
    for t in TARGET_COLUMNS:
        result[f"{t}"] = float(point[t])
        result[f"{t}_ci_low"] = float(boots[t].quantile(0.025))
        result[f"{t}_ci_high"] = float(boots[t].quantile(0.975))
        result[f"{t}_d"] = float(point[t] / sd[t])
        center = placebo[t].mean()
        result[f"{t}_placebo_mean"] = float(center)
        result[f"{t}_p"] = float((1 + ((placebo[t] - center).abs() >= abs(point[t] - center)).sum()) / (1 + n_placebo))
    return result


def holm(p_values: pd.Series) -> pd.Series:
    """Holm-adjusted p-values (family-wise error across all tests given)."""
    order = p_values.sort_values()
    m = len(order)
    adjusted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * order.to_numpy()))
    return pd.Series(adjusted, index=order.index).reindex(p_values.index)
