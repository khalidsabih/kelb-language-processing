from __future__ import annotations

from klp.data.schema import (
    CONTEXT_FEATURES,
    DYADIC_FEATURES,
    RECEIVER_PAST_FEATURES,
    SENDER_FEATURES,
)


def baseline_features() -> list[str]:
    """Receiver history plus shared dyadic state and context (no sender signals)."""
    return list(RECEIVER_PAST_FEATURES) + list(DYADIC_FEATURES) + list(CONTEXT_FEATURES)


def available_modalities(df, min_coverage: float = 0.5) -> list[str]:
    """Sender modalities whose features are all present in at least ``min_coverage`` of rows."""
    return [
        modality
        for modality, columns in SENDER_FEATURES.items()
        if df[columns].notna().all(axis=1).mean() >= min_coverage
    ]


def sender_receiver_features(modalities: list[str] | None = None) -> list[str]:
    if modalities is None:
        modalities = list(SENDER_FEATURES)
    unknown = set(modalities) - set(SENDER_FEATURES)
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")

    sender = [feature for m in modalities for feature in SENDER_FEATURES[m]]
    return baseline_features() + sender
