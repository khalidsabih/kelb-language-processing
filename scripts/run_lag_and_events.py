"""Response-lag sweep and sender-event-centred analysis.

    python3.13 scripts/run_lag_and_events.py --tracks <tracks files...> [--config configs/v0.yaml]

1. **Lag sweep.** For each lag in ``lags_s``, the receiver's target window starts that
   long after the sender's past window ends. Reports the sender increment and the
   shuffled-sender null at each lag, on the same filtered pipeline as
   ``run_experiments.py``.
2. **Sender move onsets.** Windows anchored where the sender goes from still to moving.
   A receiver-only baseline is fitted on non-event windows of other channels
   (cross-fitted by channel) and its prediction errors after sender onsets are compared
   with its errors at control times of the same pairs. A sender response shows up as a
   systematic error difference; the 95% interval resamples whole channels. Onsets where
   the receiver itself started moving in the preceding ``still_s + burst_s`` are dropped,
   since a shared start (group movement, a call from a person) is not a response.
   Then the regular sender-increment experiment and shuffle null are run on the onset
   windows only.
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from klp.data.schema import TARGET_COLUMNS
from klp.data.tracks import build_windows, cap_windows_per_video, filter_interacting, load_tracks, move_onsets, prepare_tracks
from klp.experiments.core import repeated_split_experiments, sender_null_experiment
from klp.features.select import available_modalities, baseline_features, sender_receiver_features
from klp.models.baselines import make_interaction_ridge, make_ridge

MODELS = {"interaction": make_interaction_ridge, "linear": make_ridge}
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "v0.yaml"
MIN_SPLIT_WINDOWS = 300


def _prepare(windows, modalities, cfg, cap=True):
    complete = windows.dropna(subset=sender_receiver_features(modalities) + TARGET_COLUMNS).reset_index(drop=True)
    complete = filter_interacting(complete, cfg.get("min_run_s"), cfg.get("max_pair_distance"))
    return cap_windows_per_video(complete, cfg["max_windows_per_video"], cfg["seed"]) if cap else complete


def lag_sweep(tracks, cfg, model_factory):
    rows = []
    for lag in cfg["lags_s"]:
        windows = build_windows(tracks, cfg["past_s"], cfg["future_s"], cfg["stride_s"], lag_s=lag)
        modalities = available_modalities(windows, cfg["min_modality_coverage"])
        data = _prepare(windows, modalities, cfg)
        kwargs = dict(n_splits=cfg["n_splits"], test_fraction=cfg["test_fraction"], seed=cfg["seed"],
                      model_factory=model_factory, modalities=modalities, split_unit=cfg["split_unit"])
        inc = repeated_split_experiments(data, **kwargs)["increment"]
        null = sender_null_experiment(data, "shuffle", n_null=cfg["null_draws"], **kwargs)
        rows.append({
            "lag_s": lag, "windows": len(data), "videos": data.video_id.nunique(),
            "baseline_r2": inc.baseline_r2.mean(),
            "increment_r2": inc.sender_increment_r2.mean(), "increment_r2_std": inc.sender_increment_r2.std(),
            **{f"inc_{t}": inc[f"sender_increment_r2_{t}"].mean() for t in TARGET_COLUMNS},
            "shuffle_null_r2": null["null_increment_r2_mean"], "p_shuffle": null["p_value"],
        })
    return pd.DataFrame(rows)


def _event_flags(windows, onsets, cfg):
    """Mark windows anchored at a sender onset; drop those where the receiver also just started."""
    ev = cfg["events"]
    key = set(zip(onsets.video_id, onsets.dog_id, onsets.time_s.round(3)))
    windows = windows.assign(
        sender_onset=[k in key for k in zip(windows.video_id, windows.sender_id, windows.anchor_time_s.round(3))])
    by_dog = onsets.groupby(["video_id", "dog_id"]).time_s.apply(np.asarray).to_dict()
    window_s = ev["still_s"] + ev["burst_s"]
    receiver_started = [
        bool(np.any((d := by_dog.get((v, r), np.array([]))) <= t) and np.any((t - d[d <= t]) <= window_s))
        for v, r, t in zip(windows.video_id, windows.receiver_id, windows.anchor_time_s)
    ]
    windows["receiver_onset"] = receiver_started
    # Controls: same pair, at least control_gap_s from any onset of that sender.
    sender_times = [by_dog.get((v, s), np.array([])) for v, s in zip(windows.video_id, windows.sender_id)]
    windows["far_from_onset"] = [
        len(d) == 0 or np.abs(d - t).min() >= ev["control_gap_s"] for d, t in zip(sender_times, windows.anchor_time_s)
    ]
    return windows


def event_response(windows, cfg, n_boot=2000):
    """Cross-fitted baseline residuals after sender onsets minus residuals at control times."""
    events = windows[windows.sender_onset & ~windows.receiver_onset]
    controls = windows[windows.far_from_onset & ~windows.sender_onset]
    pairs = set(zip(events.video_id, events.sender_id, events.receiver_id))
    controls = controls[[p in pairs for p in zip(controls.video_id, controls.sender_id, controls.receiver_id)]]
    data = pd.concat([events.assign(is_event=True), controls.assign(is_event=False)], ignore_index=True)

    # Baseline fitted on control windows of the other channels only.
    resid = np.full((len(data), len(TARGET_COLUMNS)), np.nan)
    pool = windows[windows.far_from_onset & ~windows.sender_onset]
    for channel in data.channel_id.unique():
        train = pool[pool.channel_id != channel]
        model = make_ridge(alpha=cfg["ridge_alpha"]).fit(train[baseline_features()], train[TARGET_COLUMNS])
        rows = (data.channel_id == channel).to_numpy()
        resid[rows] = data.loc[rows, TARGET_COLUMNS].to_numpy() - model.predict(data.loc[rows, baseline_features()])
    data[[f"resid_{t}" for t in TARGET_COLUMNS]] = resid

    # Per channel: mean event residual minus mean control residual (within the same pairs).
    cols = [f"resid_{t}" for t in TARGET_COLUMNS]
    per_channel = data.groupby(["channel_id", "is_event"])[cols].mean().unstack("is_event")
    diff = pd.DataFrame({t: per_channel[(f"resid_{t}", True)] - per_channel[(f"resid_{t}", False)]
                         for t in TARGET_COLUMNS}).dropna()
    weights = events.groupby("channel_id").size().reindex(diff.index)

    rng = np.random.default_rng(cfg["seed"])
    point = (diff.mul(weights, axis=0).sum() / weights.sum())
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(diff), len(diff), replace=True)
        boots.append(diff.iloc[pick].mul(weights.iloc[pick].to_numpy(), axis=0).sum() / weights.iloc[pick].sum())
    boots = pd.DataFrame(boots)
    summary = pd.DataFrame({
        "event_minus_control": point,
        "ci95_low": boots.quantile(0.025), "ci95_high": boots.quantile(0.975),
        "control_resid_sd": data.loc[~data.is_event, cols].std().to_numpy(),
    })
    return summary, len(events), len(controls), events.video_id.nunique(), len(diff)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracks", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    model_name = cfg.get("model", "linear")
    model_factory = partial(MODELS[model_name], alpha=cfg["ridge_alpha"])
    tracks = prepare_tracks(pd.concat([load_tracks(p) for p in args.tracks], ignore_index=True))
    pd.set_option("display.width", 220)
    print(f"Model: {model_name} ridge, alpha {cfg['ridge_alpha']}; filter: run >= {cfg.get('min_run_s')} s, "
          f"distance <= {cfg.get('max_pair_distance')} body sizes; cap {cfg['max_windows_per_video']} per video")

    print("\n=== Response-lag sweep (target window starts lag_s after the anchor) ===")
    print(lag_sweep(tracks, cfg, model_factory).round(4).to_string(index=False))

    ev = cfg["events"]
    onsets = move_onsets(tracks, ev["still_speed"], ev["moving_speed"], ev["still_s"], ev["burst_s"], ev["refractory_s"])
    print(f"\n=== Sender move onsets (still < {ev['still_speed']} then > {ev['moving_speed']} body sizes/s) ===")
    print(f"Onsets: {len(onsets)} from {onsets.dog_id.nunique()} dogs in {onsets.video_id.nunique()} videos")
    # Every frame is an anchor, so onset frames are anchors.
    windows = build_windows(tracks, cfg["past_s"], cfg["future_s"], stride_s=0.0)
    modalities = available_modalities(windows, cfg["min_modality_coverage"])
    windows = _event_flags(_prepare(windows, modalities, cfg, cap=False), onsets, cfg)
    print(f"Filtered windows anchored at a sender onset: {windows.sender_onset.sum()}, "
          f"of which the receiver also just started: {(windows.sender_onset & windows.receiver_onset).sum()}")

    summary, n_events, n_controls, n_videos, n_channels = event_response(windows, cfg)
    print(f"\nReceiver response after sender onsets vs control times of the same pairs "
          f"({n_events} onset windows, {n_controls} controls, {n_videos} videos, {n_channels} channels; "
          f"baseline cross-fitted by channel; units: body sizes, cosine, body sizes/s):")
    print(summary.round(4).to_string())

    event_windows = cap_windows_per_video(
        windows[windows.sender_onset & ~windows.receiver_onset].reset_index(drop=True),
        cfg["max_windows_per_video"], cfg["seed"])
    if event_windows.channel_id.nunique() < 4 or len(event_windows) < MIN_SPLIT_WINDOWS:
        print(f"\nSkipping the split experiment on onset windows: {len(event_windows)} windows from "
              f"{event_windows.channel_id.nunique()} channels (needs >= {MIN_SPLIT_WINDOWS} windows and >= 4 channels; "
              f"test sets would be a handful of windows and R² meaningless).")
        return
    kwargs = dict(n_splits=cfg["n_splits"], test_fraction=cfg["test_fraction"], seed=cfg["seed"],
                  model_factory=model_factory, modalities=modalities, split_unit=cfg["split_unit"])
    inc = repeated_split_experiments(event_windows, **kwargs)["increment"]
    null = sender_null_experiment(event_windows, "shuffle", n_null=cfg["null_draws"], **kwargs)
    print(f"\n=== Sender increment on onset windows only ({len(event_windows)} windows, "
          f"{event_windows.channel_id.nunique()} channels) ===")
    print(f"baseline r2 {inc.baseline_r2.mean():.4f} ± {inc.baseline_r2.std():.4f}; "
          f"increment {inc.sender_increment_r2.mean():+.4f} ± {inc.sender_increment_r2.std():.4f}; "
          f"shuffle null {null['null_increment_r2_mean']:+.4f}, p = {null['p_value']:.3f}")


if __name__ == "__main__":
    main()
