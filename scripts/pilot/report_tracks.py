"""Pilot: per-video quality report for extracted tracks.

    python3.13 scripts/pilot/report_tracks.py

Answers whether web footage yields usable two-dog data: how often two dogs are
detected together, how fragmented the tracks are, how often the footage cuts,
and how many KLP interaction windows survive.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from klp.data.schema import TARGET_COLUMNS
from klp.data.tracks import build_windows, prepare_tracks
from klp.features.select import sender_receiver_features

ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = ROOT / "data" / "external" / "pilot"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tracks", type=Path, nargs="+", default=[PILOT_DIR / "tracks.parquet"],
        help="Track file(s), each with a video_stats.csv beside it",
    )
    parser.add_argument("--out", type=Path, help="Default: quality_report.csv beside the first track file")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "v0.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    tracks = prepare_tracks(pd.concat([pd.read_parquet(path) for path in args.tracks], ignore_index=True))
    stats = pd.concat(
        [pd.read_csv(path.with_name("video_stats.csv")) for path in args.tracks], ignore_index=True
    ).set_index("video_id")
    windows = build_windows(tracks, cfg["past_s"], cfg["future_s"], cfg["stride_s"])
    usable = windows.dropna(subset=sender_receiver_features(["motion", "orientation"]) + TARGET_COLUMNS)

    rows = []
    for video_id, video in tracks.groupby("video_id"):
        dogs_per_frame = video.groupby("time_s").track_id.nunique()
        track_len = video.groupby("track_id").time_s.agg(lambda t: t.max() - t.min())
        s = stats.loc[video_id]
        duration = s.sampled_frames / (s.fps / round(s.fps / 10))
        rows.append({
            "video_id": video_id,
            "minutes": duration / 60,
            "shots_per_min": s.shots / (duration / 60),
            "frac_2plus_dogs": (dogs_per_frame >= 2).sum() / s.sampled_frames,
            "tracks": video.track_id.nunique(),
            "median_track_s": track_len.median(),
            "windows": int((windows.video_id == video_id).sum()),
            "usable_windows": int((usable.video_id == video_id).sum()),
        })
    for video_id in stats.index.difference(tracks.video_id.unique()):
        rows.append({"video_id": video_id, "windows": 0, "usable_windows": 0})

    report = pd.DataFrame(rows).sort_values("usable_windows", ascending=False)
    report.to_csv(args.out or args.tracks[0].with_name("quality_report.csv"), index=False)
    pd.set_option("display.width", 200)
    print(report.round(2).to_string(index=False))
    print(f"\nTotal: {len(usable)} usable windows of {len(windows)} from {usable.video_id.nunique()} videos")


if __name__ == "__main__":
    main()
