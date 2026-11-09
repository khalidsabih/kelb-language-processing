import argparse
from pathlib import Path

import pandas as pd

from klp.data.candidates import apply_footage_filter, candidate_videos, normalize_questions


def main():
    parser = argparse.ArgumentParser(
        description="List candidate two-dog YouTube videos from Pawgaze / K9-Bench question files."
    )
    parser.add_argument("--pawgaze", type=Path, help="Pawgaze test parquet (huggingface.co/datasets/pawgaze/pawgaze)")
    parser.add_argument("--k9bench", type=Path, help="K9-Bench test parquet (gated: huggingface.co/datasets/ogmen/K9Bench)")
    parser.add_argument(
        "--metadata", type=Path,
        help="YouTube metadata from scripts/pilot/fetch_metadata.py; keeps only long, landscape, non-Short videos",
    )
    parser.add_argument("--min-duration", type=float, default=120.0, help="Seconds; used with --metadata")
    parser.add_argument(
        "--out", type=Path,
        help="Default: candidate_videos.csv, or candidate_videos_long.csv with --metadata",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Include videos with no multi-dog questions or with footage issues",
    )
    args = parser.parse_args()

    sources = {"pawgaze": args.pawgaze, "k9bench": args.k9bench}
    tables = [normalize_questions(pd.read_parquet(path), name) for name, path in sources.items() if path]
    if not tables:
        parser.error("Pass at least one of --pawgaze or --k9bench.")

    videos = candidate_videos(pd.concat(tables, ignore_index=True), include_all=args.all)
    out = args.out or Path("candidate_videos_long.csv" if args.metadata else "candidate_videos.csv")
    if args.metadata:
        videos = apply_footage_filter(videos, pd.read_csv(args.metadata), args.min_duration)
        print("Footage issues:", videos.footage_issue.value_counts(dropna=False).to_dict())
        if not args.all:
            videos = videos[videos.footage_issue.isna()]

    videos.to_csv(out, index=False)
    print(f"Wrote {len(videos)} videos to {out}")
    print(pd.crosstab(videos.tier, videos.dataset, margins=True, margins_name="total").to_string())


if __name__ == "__main__":
    main()
