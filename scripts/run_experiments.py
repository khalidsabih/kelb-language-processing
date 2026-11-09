import argparse
from functools import partial
from pathlib import Path

import pandas as pd
import yaml

from klp.data.schema import TARGET_COLUMNS
from klp.data.tracks import build_windows, cap_windows_per_video, filter_interacting, load_tracks, prepare_tracks
from klp.experiments.core import repeated_split_experiments, sender_null_experiment
from klp.features.select import available_modalities, sender_receiver_features
from klp.models.baselines import make_interaction_ridge, make_ridge

MODELS = {"interaction": make_interaction_ridge, "linear": make_ridge}

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "v0.yaml"


def _mean_std(series) -> str:
    return f"{series.mean():.4f} ± {series.std():.4f}"


def main():
    parser = argparse.ArgumentParser(description="Run KLP experiments on per-frame dog tracks.")
    parser.add_argument("--tracks", type=Path, nargs="+", required=True, help="CSV or Parquet track file(s)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    tracks = prepare_tracks(pd.concat([load_tracks(path) for path in args.tracks], ignore_index=True))
    windows = build_windows(tracks, cfg["past_s"], cfg["future_s"], cfg["stride_s"])
    print(f"Tracks: {tracks.video_id.nunique()} videos, {tracks.dog_id.nunique()} dogs, {len(tracks)} rows")
    print(f"Windows: {len(windows)}")

    model_factory = partial(MODELS[cfg.get("model", "linear")], alpha=cfg["ridge_alpha"])
    print(f"Model: {cfg.get('model', 'linear')} ridge, alpha {cfg['ridge_alpha']}")
    modalities = available_modalities(windows, cfg["min_modality_coverage"])
    if not modalities:
        raise SystemExit("No sender modality has enough coverage to run experiments.")
    used = sender_receiver_features(modalities) + TARGET_COLUMNS
    complete = windows.dropna(subset=used).reset_index(drop=True)
    print(f"Sender modalities used: {modalities}")
    print(f"Windows with complete features: {len(complete)} of {len(windows)}")
    complete = filter_interacting(complete, cfg.get("min_run_s"), cfg.get("max_pair_distance"))
    print(f"Continuous (>= {cfg.get('min_run_s')} s) and close (<= {cfg.get('max_pair_distance')} body sizes): "
          f"{len(complete)} windows from {complete.video_id.nunique()} videos, {complete.channel_id.nunique()} channels")
    complete = cap_windows_per_video(complete, cfg["max_windows_per_video"], cfg["seed"])
    per_video = complete.video_id.value_counts()
    print(f"After cap of {cfg['max_windows_per_video']} per video: {len(complete)} windows from {len(per_video)} videos, "
          f"largest video share {per_video.iat[0] / len(complete):.0%}")

    results = repeated_split_experiments(
        complete,
        n_splits=cfg["n_splits"],
        test_fraction=cfg["test_fraction"],
        seed=cfg["seed"],
        model_factory=model_factory,
        modalities=modalities,
        split_unit=cfg["split_unit"],
    )

    splits = results["splits"]
    print(f"\n{cfg['n_splits']} identity-disjoint splits by {cfg['split_unit']} (mean ± std)")
    print(f"Rows: train={_mean_std(splits.n_train)}, test={_mean_std(splits.n_test)}")
    print(f"Dropped (cross-partition) rows: {_mean_std(splits.dropped_fraction)} of total")
    print(f"Test dogs per split: {_mean_std(splits.n_test_dogs)}")

    inc = results["increment"]
    print("\n=== Sender incremental-value experiment ===")
    print(f"baseline r2:             {_mean_std(inc.baseline_r2)}")
    print(f"sender_plus_baseline r2: {_mean_std(inc.sender_plus_baseline_r2)}")
    print(f"sender_increment_r2:     {_mean_std(inc.sender_increment_r2)}")
    print(f"sender_increment_mae:    {_mean_std(inc.sender_increment_mae)}")
    for target in TARGET_COLUMNS:
        print(f"  increment r2 {target}: {_mean_std(inc[f'sender_increment_r2_{target}'])}")

    print("\n=== Leave-one-modality-out ablation ===")
    print(results["ablation"].to_string(index=False))

    print(f"\n=== Sender nulls ({cfg['null_draws']} draws each; same splits, same rows as the real data) ===")
    nulls = pd.DataFrame([
        sender_null_experiment(
            complete, kind, n_null=cfg["null_draws"], n_splits=cfg["n_splits"], test_fraction=cfg["test_fraction"],
            seed=cfg["seed"], model_factory=model_factory,
            modalities=modalities, split_unit=cfg["split_unit"], min_shift_s=cfg["null_min_shift_s"],
        )
        for kind in ("shuffle", "time_shift")
    ])
    print(nulls.to_string(index=False))


if __name__ == "__main__":
    main()
