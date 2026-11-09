"""Do receivers respond more to sender events they can see? (cycle-5 follow-up)

    python3.13 scripts/check_event_facing.py <tracks_stabilized files...> [--config configs/v0.yaml]

A response to a sender's signal should be stronger when the receiver faces the sender
at the event than when it faces away; a stimulus that reaches both dogs (a person, a
noise) predicts no such difference. For the 8 event responses that passed Holm
correction (cells below), each event's matched-control difference is split by the
receiver's facing at the event frame: cosine between its pose body heading and the
direction to the sender (> 0.5 = sender within 60° of straight ahead; < 0 = sender
beside or behind). Frames whose heading is not from pose are left out.

Prediction, fixed before running: the sign-aligned, SD-scaled response (positive = in
the direction of the cell's overall effect) is larger when facing. Pooled over cells,
facing minus not-facing is tested one-sided with a permutation of facing labels across
events of the same kind (an event in several cells keeps one label).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from klp.data.events import detect_events
from klp.data.schema import TARGET_COLUMNS
from klp.data.tracks import build_windows, filter_interacting, load_tracks, prepare_tracks
from klp.experiments.events import Matcher, flag_event_windows
from klp.features.select import baseline_features

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "v0.yaml"
CELLS = [(0.0, "move_onset", "delta_distance"), (0.0, "stop", "delta_velocity"), (1.0, "turn_to", "delta_distance"),
         (1.0, "approach", "delta_velocity"), (1.0, "withdraw", "delta_velocity"), (2.0, "approach", "delta_distance"),
         (2.0, "withdraw", "delta_distance"), (2.0, "turn_to", "delta_velocity")]
FACING, NOT_FACING = 0.5, 0.0


def receiver_facing(tracks: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    """Cosine between the receiver's pose heading and the direction to the sender at each event frame."""
    frame = tracks.set_index(["video_id", "dog_id", "time_s"])
    out = []
    for v, s, r, t in zip(events.video_id, events.sender_id, events.receiver_id, events.anchor_time_s):
        rec, snd = frame.loc[(v, r, t)], frame.loc[(v, s, t)]
        if rec.get("heading_source") != "pose":
            out.append(np.nan)
            continue
        out.append(np.cos(rec.heading - np.arctan2(snd.y - rec.y, snd.x - rec.x)))
    return np.array(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tracks", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--facing", type=float, default=FACING, help="cosine above which the receiver faces the sender")
    parser.add_argument("--not-facing", type=float, default=NOT_FACING, help="cosine below which it does not")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    ev = cfg["communication_events"]
    tracks = prepare_tracks(pd.concat([load_tracks(p) for p in args.tracks], ignore_index=True))
    events = detect_events(tracks, before_s=ev["before_s"], after_s=ev["after_s"], refractory_s=ev["refractory_s"])

    rows = []
    for lag in sorted({c[0] for c in CELLS}):
        w = build_windows(tracks, cfg["past_s"], cfg["future_s"], stride_s=0.0, lag_s=lag)
        w = filter_interacting(w.dropna(subset=baseline_features() + TARGET_COLUMNS), ev["min_run_s"], ev["max_pair_distance"])
        for cell_lag, kind, target in CELLS:
            if cell_lag != lag:
                continue
            f = flag_event_windows(w, events, kind, ev["control_gap_s"], ev["simultaneous_s"]).reset_index(drop=True)
            event_rows = np.flatnonzero(f.is_event.to_numpy())
            d = Matcher(f, baseline_features(), ev["n_controls"]).differences(event_rows, labels=event_rows)
            e = f.loc[d.label, ["video_id", "sender_id", "receiver_id", "anchor_time_s"]].reset_index(drop=True)
            sd = f.loc[f.is_control, target].std()
            rows.append(e.assign(lag_s=lag, kind=kind, target=target, response=d[target].to_numpy(),
                                 z=d[target].to_numpy() / sd * np.sign(d[target].mean())))
    data = pd.concat(rows, ignore_index=True)
    data["facing"] = receiver_facing(tracks, data)
    data["group"] = np.select([data.facing > args.facing, data.facing < args.not_facing], ["facing", "not_facing"], "side")
    data.loc[data.facing.isna(), "group"] = "no_pose_heading"

    pd.set_option("display.width", 220)
    print(f"Receiver facing at the event: 'facing' cos > {args.facing}, 'not_facing' cos < {args.not_facing}; "
          f"z = response / control SD, signed so that + is the cell's overall direction\n")
    per_cell = data.groupby(["lag_s", "kind", "target", "group"]).agg(n=("z", "size"), response=("response", "mean"),
                                                                       z=("z", "mean")).unstack("group")
    print(per_cell.round(3).to_string())

    used = data[data.group.isin(["facing", "not_facing"])]
    def stat(groups):
        return used.z[groups == "facing"].mean() - used.z[groups == "not_facing"].mean()
    real = stat(used.group.to_numpy())

    # Permute labels over distinct events within kind, so one event keeps one label across cells.
    key = list(zip(used.video_id, used.sender_id, used.receiver_id, used.anchor_time_s, used.kind))
    uniq = pd.DataFrame(dict.fromkeys(key).keys(), columns=["v", "s", "r", "t", "kind"])
    label_of = dict(zip(key, used.group))
    uniq["group"] = [label_of[tuple(k)] for k in uniq.itertuples(index=False)]
    pos = {k: i for i, k in enumerate(map(tuple, uniq[["v", "s", "r", "t", "kind"]].to_numpy()))}
    idx = np.array([pos[k] for k in key])
    rng = np.random.default_rng(cfg["seed"])
    perm = []
    for _ in range(args.n_perm):
        g = uniq.group.to_numpy().copy()
        for _, members in uniq.groupby("kind").indices.items():
            g[members] = rng.permutation(g[members])
        perm.append(stat(g[idx]))
    perm = np.array(perm)
    p = (1 + (perm >= real).sum()) / (1 + args.n_perm)
    n_f, n_n = (uniq.group == "facing").sum(), (uniq.group == "not_facing").sum()
    print(f"\nPooled over the 8 cells ({n_f} facing and {n_n} not-facing distinct events; "
          f"{(data.group == 'side').sum()} side and {(data.group == 'no_pose_heading').sum()} no-pose cell rows left out):")
    print(f"mean z facing {used.z[used.group == 'facing'].mean():+.3f}, not facing {used.z[used.group == 'not_facing'].mean():+.3f}, "
          f"difference {real:+.3f}; one-sided permutation p = {p:.4f}; "
          f"permutation SD {perm.std():.3f}, so a difference of about {2.5 * perm.std():.2f} would be detectable (p < 0.05 with 80% power)")


if __name__ == "__main__":
    main()
