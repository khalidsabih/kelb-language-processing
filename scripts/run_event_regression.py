"""Sender event indicators as regression features (cycle-5 follow-up).

    python3.13 scripts/run_event_regression.py --tracks <tracks files...> [--config configs/v0.yaml]

Adds the ``events`` sender modality: one indicator per event kind, 1 if the sender had
that event toward the receiver in the past window (``klp.data.events``). Then, on three
window sets:

- ``clean``: the regression pipeline of ``run_experiments.py`` (pair filter
  ``min_run_s`` / ``max_pair_distance``, cap per video);
- ``relaxed``: the looser pair filter of ``communication_events``, cap per video;
- ``event windows``: relaxed windows whose sender had any event in the past window.

reports (1) the increment of the event indicators alone over the receiver-only baseline,
with the shuffled-sender null (indicators permuted within video), and (2) what they add
on top of the other sender modalities (leave-one-modality-out delta: full minus without
events). Splits hold out whole channels, as in the main experiments.
"""

import argparse
from functools import partial
from pathlib import Path

import pandas as pd
import yaml

from klp.data.events import add_event_features, detect_events
from klp.data.schema import SENDER_FEATURES, TARGET_COLUMNS
from klp.data.tracks import build_windows, cap_windows_per_video, filter_interacting, load_tracks, prepare_tracks
from klp.experiments.core import repeated_split_experiments, sender_null_experiment
from klp.features.select import available_modalities, sender_receiver_features
from klp.models.baselines import make_interaction_ridge, make_ridge

MODELS = {"interaction": make_interaction_ridge, "linear": make_ridge}
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "v0.yaml"
MIN_SPLIT_WINDOWS = 300


def evaluate(name, data, modalities, cfg, model_factory):
    any_event = data[SENDER_FEATURES["events"]].sum(axis=1) > 0
    row = {"set": name, "windows": len(data), "with_event": int(any_event.sum()),
           "videos": data.video_id.nunique(), "channels": data.channel_id.nunique()}
    if len(data) < MIN_SPLIT_WINDOWS or data.channel_id.nunique() < 4:
        print(f"Skipping {name}: needs >= {MIN_SPLIT_WINDOWS} windows and >= 4 channels")
        return row
    kwargs = dict(n_splits=cfg["n_splits"], test_fraction=cfg["test_fraction"], seed=cfg["seed"],
                  model_factory=model_factory, split_unit=cfg["split_unit"])
    alone = repeated_split_experiments(data, modalities=["events"], **kwargs)["increment"]
    null = sender_null_experiment(data, "shuffle", n_null=cfg["null_draws"], modalities=["events"], **kwargs)
    full = repeated_split_experiments(data, modalities=modalities, **kwargs)
    ablation = full["ablation"].set_index("dropped_modality")
    row.update(
        baseline_r2=alone.baseline_r2.mean(),
        events_alone_increment=alone.sender_increment_r2.mean(), events_alone_std=alone.sender_increment_r2.std(),
        **{f"alone_{t}": alone[f"sender_increment_r2_{t}"].mean() for t in TARGET_COLUMNS},
        shuffle_null=null["null_increment_r2_mean"], p_shuffle=null["p_value"],
        all_modalities_increment=full["increment"].sender_increment_r2.mean(),
        events_on_top=ablation.loc["events", "delta_r2_vs_full_mean"],
        events_on_top_std=ablation.loc["events", "delta_r2_vs_full_std"],
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracks", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    ev = cfg["communication_events"]
    model_factory = partial(MODELS[cfg.get("model", "linear")], alpha=cfg["ridge_alpha"])
    tracks = prepare_tracks(pd.concat([load_tracks(p) for p in args.tracks], ignore_index=True))

    events = detect_events(tracks, before_s=ev["before_s"], after_s=ev["after_s"], refractory_s=ev["refractory_s"])
    windows = add_event_features(build_windows(tracks, cfg["past_s"], cfg["future_s"], cfg["stride_s"]),
                                 events, cfg["past_s"])
    modalities = available_modalities(windows, cfg["min_modality_coverage"])
    complete = windows.dropna(subset=sender_receiver_features(modalities) + TARGET_COLUMNS).reset_index(drop=True)
    print(f"Model: {cfg.get('model', 'linear')} ridge; sender modalities: {modalities}; "
          f"{len(events)} events; {len(complete)} complete windows")

    clean = cap_windows_per_video(filter_interacting(complete, cfg["min_run_s"], cfg["max_pair_distance"]),
                                  cfg["max_windows_per_video"], cfg["seed"])
    relaxed_all = filter_interacting(complete, ev["min_run_s"], ev["max_pair_distance"])
    relaxed = cap_windows_per_video(relaxed_all, cfg["max_windows_per_video"], cfg["seed"])
    event_windows = relaxed_all[relaxed_all[SENDER_FEATURES["events"]].sum(axis=1) > 0].reset_index(drop=True)

    rows = [evaluate(name, data, modalities, cfg, model_factory)
            for name, data in [("clean", clean), ("relaxed", relaxed), ("event windows", event_windows)]]
    pd.set_option("display.width", 250)
    print("\nR² increments (full minus baseline; events_on_top = full minus full-without-events), mean over splits")
    print(pd.DataFrame(rows).round(4).T.to_string(header=False))


if __name__ == "__main__":
    main()
