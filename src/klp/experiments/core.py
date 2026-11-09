from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from klp.data.schema import SENDER_FEATURES, TARGET_COLUMNS
from klp.data.splits import dog_disjoint_splits, split_stats
from klp.features.select import baseline_features, sender_receiver_features
from klp.models.baselines import make_ridge
from klp.evaluation.metrics import regression_metrics

ModelFactory = Callable[[], object]


def _fit_eval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_factory: ModelFactory,
) -> dict[str, float]:
    model = model_factory()
    model.fit(train[features], train[TARGET_COLUMNS])
    pred = model.predict(test[features])
    return regression_metrics(test[TARGET_COLUMNS], pred)


def sender_increment_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_factory: ModelFactory = make_ridge,
    modalities: list[str] | None = None,
) -> dict[str, float]:
    """Compare the baseline (receiver past + dyadic + context) with baseline + sender."""
    baseline = _fit_eval(train, test, baseline_features(), model_factory)
    full = _fit_eval(train, test, sender_receiver_features(modalities), model_factory)

    result = {f"baseline_{k}": v for k, v in baseline.items()}
    result.update({f"sender_plus_baseline_{k}": v for k, v in full.items()})
    result["sender_increment_r2"] = full["r2"] - baseline["r2"]
    result["sender_increment_mae"] = baseline["mae"] - full["mae"]
    for target in TARGET_COLUMNS:
        result[f"sender_increment_r2_{target}"] = full[f"r2_{target}"] - baseline[f"r2_{target}"]
    return result


def modality_ablation_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_factory: ModelFactory = make_ridge,
    modalities: list[str] | None = None,
) -> pd.DataFrame:
    modalities = list(SENDER_FEATURES) if modalities is None else list(modalities)
    full_metrics = _fit_eval(train, test, sender_receiver_features(modalities), model_factory)
    rows = []

    for dropped in modalities:
        kept = [m for m in modalities if m != dropped]
        metrics = _fit_eval(train, test, sender_receiver_features(kept), model_factory)
        rows.append({
            "dropped_modality": dropped,
            "r2": metrics["r2"],
            "mae": metrics["mae"],
            "delta_r2_vs_full": full_metrics["r2"] - metrics["r2"],
            "delta_mae_vs_full": metrics["mae"] - full_metrics["mae"],
        })

    return pd.DataFrame(rows).sort_values("delta_r2_vs_full", ascending=False)


def null_sender(
    df: pd.DataFrame, columns: list[str], kind: str, rng: np.random.Generator, min_shift_s: float = 5.0
) -> pd.DataFrame:
    """Copy of ``df`` whose sender features no longer line up in time with the receiver.

    - ``"shuffle"``: sender features permuted across windows within each video, which
      keeps each video's distribution of sender behavior but breaks the pairing.
    - ``"time_shift"``: within each sender-receiver pair, sender features circularly
      shifted by a random offset so every window gets the pair's sender behavior from
      at least ``min_shift_s`` away. This also keeps the sender's own autocorrelation.
      Pairs too short for that get NaN sender features.

    A real sender effect should beat both: the increment must exceed what sender
    features that carry no moment-to-moment information achieve.
    """
    out = df.copy()
    if kind == "shuffle":
        for _, idx in df.groupby("video_id").indices.items():
            out.iloc[idx, out.columns.get_indexer(columns)] = df.iloc[rng.permutation(idx)][columns].to_numpy()
        return out
    if kind != "time_shift":
        raise ValueError(f"Unknown null kind: {kind!r}")

    out[columns] = np.nan
    for _, idx in df.groupby(["video_id", "sender_id", "receiver_id"]).indices.items():
        idx = idx[np.argsort(df.anchor_time_s.to_numpy()[idx])]
        times = df.anchor_time_s.to_numpy()[idx]
        offsets = [k for k in range(1, len(idx)) if np.abs(np.roll(times, -k) - times).min() >= min_shift_s]
        if offsets:
            donor = np.roll(idx, -rng.choice(offsets))
            out.iloc[idx, out.columns.get_indexer(columns)] = df.iloc[donor][columns].to_numpy()
    return out


def sender_null_experiment(
    df: pd.DataFrame,
    kind: str,
    n_null: int = 20,
    n_splits: int = 10,
    test_fraction: float = 0.25,
    seed: int = 42,
    model_factory: ModelFactory = make_ridge,
    modalities: list[str] | None = None,
    split_unit: str = "component",
    min_shift_s: float = 5.0,
) -> dict:
    """Real sender increment versus increments from ``n_null`` null sender datasets.

    Every null draw is evaluated on the same splits as the real data, and only rows
    that have sender features in the null (all rows for ``"shuffle"``) are used for
    both, so the comparison differs only in whether sender features are aligned.
    ``p_value`` is the share of null draws whose mean increment across splits is at
    least the real one (with the usual +1 correction).
    """
    rng = np.random.default_rng(seed)
    sender = [f for f in sender_receiver_features(modalities) if f not in baseline_features()]
    nulls = [null_sender(df, sender, kind, rng, min_shift_s) for _ in range(n_null)]
    keep = np.logical_and.reduce([n[sender].notna().all(axis=1).to_numpy() for n in nulls])
    df = df[keep].reset_index(drop=True).assign(_row=lambda d: np.arange(len(d)))
    nulls = [n[keep].reset_index(drop=True) for n in nulls]

    full_features = sender_receiver_features(modalities)
    real, null = [], np.zeros((n_null, n_splits))
    for i, (train, test) in enumerate(dog_disjoint_splits(df, n_splits, test_fraction, seed, unit=split_unit)):
        base_r2 = _fit_eval(train, test, baseline_features(), model_factory)["r2"]
        real.append(_fit_eval(train, test, full_features, model_factory)["r2"] - base_r2)
        for j, n in enumerate(nulls):
            null_train, null_test = n.iloc[train._row], n.iloc[test._row]
            null[j, i] = _fit_eval(null_train, null_test, full_features, model_factory)["r2"] - base_r2

    real_mean, null_means = float(np.mean(real)), null.mean(axis=1)
    return {
        "kind": kind,
        "n_rows": len(df),
        "real_increment_r2": real_mean,
        "null_increment_r2_mean": float(null_means.mean()),
        "null_increment_r2_p95": float(np.quantile(null_means, 0.95)),
        "real_minus_null_r2": real_mean - float(null_means.mean()),
        "p_value": float((1 + (null_means >= real_mean).sum()) / (1 + n_null)),
    }


def repeated_split_experiments(
    df: pd.DataFrame,
    n_splits: int = 10,
    test_fraction: float = 0.25,
    seed: int = 42,
    model_factory: ModelFactory = make_ridge,
    modalities: list[str] | None = None,
    split_unit: str = "component",
) -> dict[str, pd.DataFrame]:
    """Run both experiments over ``n_splits`` dog-disjoint splits.

    Returns per-split ``splits`` stats and ``increment`` results, plus an
    ``ablation`` table with the mean and std of each delta across splits.
    """
    split_rows, increment_rows, ablation_tables = [], [], []
    splits = dog_disjoint_splits(df, n_splits, test_fraction, seed, unit=split_unit)
    for i, (train, test) in enumerate(splits):
        split_rows.append({"split": i, **split_stats(df, train, test)})
        increment_rows.append({"split": i, **sender_increment_experiment(train, test, model_factory, modalities)})
        ablation = modality_ablation_experiment(train, test, model_factory, modalities)
        ablation_tables.append(ablation.assign(split=i))

    ablation = (
        pd.concat(ablation_tables)
        .groupby("dropped_modality")[["delta_r2_vs_full", "delta_mae_vs_full"]]
        .agg(["mean", "std"])
    )
    ablation.columns = ["_".join(col) for col in ablation.columns]
    ablation = ablation.sort_values("delta_r2_vs_full_mean", ascending=False).reset_index()

    return {
        "splits": pd.DataFrame(split_rows),
        "increment": pd.DataFrame(increment_rows),
        "ablation": ablation,
    }
