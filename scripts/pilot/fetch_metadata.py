"""Fetch YouTube metadata (no video download) for candidate videos.

    .venv-pilot/bin/python scripts/pilot/fetch_metadata.py

Writes duration, resolution, channel and whether each video is still available,
so candidates can be filtered for long, landscape, non-Short footage before
downloading anything.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from yt_dlp import YoutubeDL

ROOT = Path(__file__).resolve().parents[2]
BOT_CHECK = "confirm you"


def fetch(video_id: str) -> dict:
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # removed, private, region-locked, rate-limited, ...
        error = str(exc).splitlines()[0][:200]
        # YouTube's bot check means "rate-limited", not "gone": leave availability unknown.
        available = None if BOT_CHECK in error else False
        return {"video_id": video_id, "available": available, "error": error}
    return {
        "video_id": video_id,
        "available": True,
        "duration_s": info.get("duration"),
        "width": info.get("width"),
        "height": info.get("height"),
        "fps": info.get("fps"),
        "channel_id": info.get("channel_id"),
        "title": info.get("title"),
        "is_short": "/shorts/" in (info.get("webpage_url") or "") or "#shorts" in (info.get("title") or "").lower(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", type=Path, default=ROOT / "candidate_videos.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "external" / "video_metadata.csv")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--retry-blocked", action="store_true",
        help="Re-fetch only videos blocked by YouTube's bot check in --out, one at a time",
    )
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between requests with --retry-blocked")
    args = parser.parse_args()

    if args.retry_blocked:
        meta = pd.read_csv(args.out)
        blocked = meta.error.fillna("").str.contains(BOT_CHECK)
        rows = []
        for video_id in meta.loc[blocked, "video_id"]:
            rows.append(fetch(video_id))
            time.sleep(args.delay)
        meta = pd.concat([meta[~blocked], pd.DataFrame(rows)], ignore_index=True)
    else:
        ids = pd.read_csv(args.candidates).video_id.drop_duplicates().tolist()
        with ThreadPoolExecutor(args.workers) as pool:
            meta = pd.DataFrame(pool.map(fetch, ids))
    meta.to_csv(args.out, index=False)
    still_blocked = meta.error.fillna("").str.contains(BOT_CHECK).sum()
    print(f"Wrote metadata for {len(meta)} videos to {args.out} "
          f"({meta.available.eq(True).sum()} available, {still_blocked} blocked by bot check)")


if __name__ == "__main__":
    main()
