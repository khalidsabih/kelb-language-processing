from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def dog_disjoint_split(
    df: pd.DataFrame,
    test_fraction: float = 0.25,
    seed: int = 42,
    unit: str = "component",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so no sender/receiver dog appears in both sets.

    ``unit="component"`` holds out whole connected components of the dog
    co-occurrence graph (dogs linked by sharing a row). No rows are dropped and
    ``test_fraction`` is the fraction of components held out. This suits data
    where dogs rarely meet across videos, e.g. web clips with per-video dog IDs.

    ``unit="video"`` is the same, but dogs that appear in the same video are also
    linked, so whole videos are held out. Use it when tracks fragment (a tracker
    gives one real dog several IDs, so fragments of one recording would otherwise
    land on both sides). With per-video dog IDs, ``test_fraction`` is the fraction
    of videos.

    ``unit="channel"`` also links dogs whose videos share a ``channel_id``, so
    whole channels are held out. Use it for web video, where one channel often
    films the same dogs across many videos.

    ``unit="dog"`` holds out individual dogs; ``test_fraction`` is the fraction of
    dogs. Rows whose sender and receiver fall on opposite sides are excluded, so
    the test set holds roughly ``test_fraction**2`` of the rows; use
    ``split_stats`` to report this. Needed when dogs form one large social network.

    If a ``video_id`` column is present, no video may contribute rows to both
    partitions (windows from one recording share background and conditions).
    """
    rng = np.random.default_rng(seed)

    if unit in ("component", "video", "channel"):
        link_by = {"component": None, "video": "video_id", "channel": "channel_id"}[unit]
        if link_by and link_by not in df.columns:
            raise ValueError(f"unit={unit!r} needs a {link_by!r} column.")
        component = _dog_components(df, link_by=link_by)
        groups = df.sender_id.map(component)
        ids = np.unique(groups)
        rng.shuffle(ids)
        test_ids = set(ids[: max(1, int(round(len(ids) * test_fraction)))])
        test_mask = groups.isin(test_ids)
        train_mask = ~test_mask
    elif unit == "dog":
        dogs = np.unique(np.concatenate([df.sender_id.unique(), df.receiver_id.unique()]))
        rng.shuffle(dogs)
        n_test = max(1, int(round(len(dogs) * test_fraction)))
        test_dogs = set(dogs[:n_test])
        train_dogs = set(dogs[n_test:])
        train_mask = df.sender_id.isin(train_dogs) & df.receiver_id.isin(train_dogs)
        test_mask = df.sender_id.isin(test_dogs) & df.receiver_id.isin(test_dogs)
    else:
        raise ValueError(f"unit must be 'component', 'video', 'channel' or 'dog', got {unit!r}")

    train = df.loc[train_mask].reset_index(drop=True)
    test = df.loc[test_mask].reset_index(drop=True)

    if train.empty or test.empty:
        hint = " All dogs form one connected component; use unit='dog'." if unit != "dog" else ""
        raise ValueError(f"Identity-disjoint split produced an empty partition.{hint}")

    if "video_id" in df.columns:
        shared = set(train.video_id) & set(test.video_id)
        if shared:
            raise ValueError(
                f"{len(shared)} video(s) contribute rows to both train and test "
                f"(e.g. {sorted(shared)[:3]}); if tracks fragment, use unit='video'."
            )

    return train, test


def dog_disjoint_splits(
    df: pd.DataFrame,
    n_splits: int = 10,
    test_fraction: float = 0.25,
    seed: int = 42,
    unit: str = "component",
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield ``n_splits`` independent dog-disjoint splits (seeds ``seed .. seed+n-1``)."""
    for i in range(n_splits):
        yield dog_disjoint_split(df, test_fraction=test_fraction, seed=seed + i, unit=unit)


def split_stats(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    n_dropped = len(df) - len(train) - len(test)
    return {
        "n_total": len(df),
        "n_train": len(train),
        "n_test": len(test),
        "n_dropped": n_dropped,
        "dropped_fraction": n_dropped / len(df),
        "n_train_dogs": len(set(train.sender_id) | set(train.receiver_id)),
        "n_test_dogs": len(set(test.sender_id) | set(test.receiver_id)),
    }


def _dog_components(df: pd.DataFrame, link_by: str | None = None) -> dict[str, str]:
    """Map each dog to a representative of its connected component (union-find).

    Dogs are linked when they share a row and, with ``link_by``, a value of that
    column (e.g. the same video or channel).
    """
    parent: dict[str, str] = {}

    def find(dog: str) -> str:
        parent.setdefault(dog, dog)
        while parent[dog] != dog:
            parent[dog] = parent[parent[dog]]
            dog = parent[dog]
        return dog

    for sender, receiver in zip(df.sender_id, df.receiver_id):
        parent[find(sender)] = find(receiver)
    if link_by:
        for dogs in df.groupby(link_by).sender_id.unique():
            for dog in dogs[1:]:
                parent[find(dog)] = find(dogs[0])
    return {dog: find(dog) for dog in parent}
