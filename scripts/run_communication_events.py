"""Cycle 5: receiver responses after candidate sender behavioral events.

    python3.13 scripts/run_communication_events.py --tracks <tracks files...> [--config configs/v0.yaml]

Detects sender events per ordered dog pair (move onset, stop, approach, withdraw, turn
toward, turn away; ``klp.data.events``). For each event kind and response lag, the
receiver's change in distance, orientation and speed after the event is compared with
matched control moments: the nearest moments in the same video where the receiver was
in a similar state (own speed, orientation, distance change, pair distance) but the
sender had no such event. Intervals resample whole channels; placebo p-values come from
running the same matching at random control moments of the same pairs, and are
Holm-adjusted across every kind, lag and target.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from klp.data.events import detect_events
from klp.data.schema import TARGET_COLUMNS
from klp.data.tracks import build_windows, filter_interacting, load_tracks, prepare_tracks
from klp.experiments.events import event_response, holm
from klp.features.select import baseline_features

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "v0.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracks", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, help="Optional CSV for the full results table")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    ev = cfg["communication_events"]
    tracks = prepare_tracks(pd.concat([load_tracks(p) for p in args.tracks], ignore_index=True))
    pd.set_option("display.width", 250)

    events = detect_events(tracks, before_s=ev["before_s"], after_s=ev["after_s"], refractory_s=ev["refractory_s"])
    print(f"Sender events in all co-tracked pairs ({events.video_id.nunique()} videos):")
    print(events.groupby("kind").agg(events=("time_s", "size"), videos=("video_id", "nunique"),
                                     channels=("channel_id", "nunique")).to_string())

    rows = []
    for lag in ev["response_lags_s"]:
        # Every frame is an anchor, so every event frame has a window.
        windows = build_windows(tracks, cfg["past_s"], cfg["future_s"], stride_s=0.0, lag_s=lag)
        windows = windows.dropna(subset=baseline_features() + TARGET_COLUMNS)
        windows = filter_interacting(windows, ev["min_run_s"], ev["max_pair_distance"])
        for kind in events.kind.unique():
            rows.append({"lag_s": lag, **event_response(
                windows, events, kind, n_controls=ev["n_controls"], control_gap_s=ev["control_gap_s"],
                simultaneous_s=ev["simultaneous_s"], n_placebo=ev["n_placebo"], seed=cfg["seed"])})
        print(f"lag {lag} s done ({len(windows)} windows)", flush=True)
    table = pd.DataFrame(rows)

    p_cols = [f"{t}_p" for t in TARGET_COLUMNS]
    stacked = table[p_cols].stack()
    adjusted = holm(stacked).unstack()
    for t in TARGET_COLUMNS:
        table[f"{t}_p_holm"] = adjusted[f"{t}_p"]
    if args.out:
        table.to_csv(args.out, index=False)

    print(f"\nReceiver response after sender events minus matched controls "
          f"(pair distance <= {ev['max_pair_distance']} body sizes; units: body sizes, cosine, body sizes/s; "
          f"d = difference / control SD; p = placebo, Holm-adjusted over {len(stacked)} tests)")
    print(f"Max |mean standardized imbalance| of matched features per kind and lag: "
          f"{table.max_abs_balance.max():.3f}")
    for t in TARGET_COLUMNS:
        print(f"\n--- {t} ---")
        cols = ["lag_s", "kind", "events", "videos", "channels", t, f"{t}_ci_low", f"{t}_ci_high", f"{t}_d",
                f"{t}_placebo_mean", f"{t}_p", f"{t}_p_holm"]
        print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
