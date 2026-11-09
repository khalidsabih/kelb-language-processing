"""Find more two-dog YouTube videos by search (metadata only, no download).

    .venv-pilot/bin/python scripts/pilot/search_videos.py --out data/external/search_candidates.csv

Runs each query as a YouTube search, keeps results that are long enough, not Shorts and
whose titles mention dogs and do not suggest edited footage (compilations, music, funny
clips) or talks (guides, tips), and drops videos already listed in ``--exclude`` files. Requests are spaced out to avoid YouTube's
bot check. The output has the columns ``fetch_metadata.py`` and ``extract_tracks.py``
need (``video_id``, ``youtube_url``, ``channel_id``, ``duration_s``, ``title``) plus the
matching ``query``. Whether a video really shows two dogs interacting still has to be
checked on the footage.
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
from yt_dlp import YoutubeDL

QUERIES = [
    "two dogs playing raw footage",
    "dogs playing uncut no music",
    "dog park footage unedited",
    "dog play session full length",
    "dogs wrestling original audio",
    "dog daycare camera footage",
    "dog meeting another dog for the first time",
    "introducing a new dog to my dog",
    "dog body language play analysis",
    "puppy and adult dog playing",
    "dogs playing in the backyard",
    "two dogs playing tug of war",
]
EDITED = re.compile(r"compilation|funny|try not to laugh|music|song|lyrics|#shorts|top \d+|best of|asmr|cartoon|animation",
                    re.IGNORECASE)
# Titles must mention dogs; searches also return unrelated videos (wrestling, music).
DOGS = re.compile(r"\bdogs?\b|pupp|\bpups?\b|doggy|doggo|canine|\bk9\b|husk|retriever|shepherd|terrier|corgi|poodle|"
                  r"beagle|collie|pit ?bulls?|labrador|doodle|malinois|boerboel|sheltie|rottweiler|dachshund|spaniel",
                  re.IGNORECASE)
# Talks and tutorials rarely show long uninterrupted interaction.
TALK = re.compile(r"guide|how to|tips|hacks|explained|mistakes|what it (?:really )?means|signs|podcast|interview",
                  re.IGNORECASE)


def search(query: str, n: int) -> list[dict]:
    with YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    return [{
        "video_id": e.get("id"),
        "youtube_url": f"https://www.youtube.com/watch?v={e.get('id')}",
        "title": e.get("title"),
        "channel_id": e.get("channel_id"),
        "duration_s": e.get("duration"),
        "is_short": "/shorts/" in (e.get("url") or ""),
        "query": query,
    } for e in info.get("entries", []) if e and e.get("id")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-query", type=int, default=50)
    parser.add_argument("--min-duration", type=float, default=120.0)
    parser.add_argument("--max-duration", type=float, default=3600.0)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="CSV files with a video_id column")
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--from-raw", action="store_true",
                        help="Re-filter the unfiltered results saved next to --out instead of searching again")
    args = parser.parse_args()
    raw_path = args.out.with_name(args.out.stem + "_raw.csv")

    if args.from_raw:
        found = pd.read_csv(raw_path)
    else:
        rows = []
        for query in QUERIES:
            try:
                rows += search(query, args.per_query)
            except Exception as exc:  # rate limit, network
                print(f"{query!r}: {str(exc).splitlines()[0][:120]}")
            time.sleep(args.delay)
        found = pd.DataFrame(rows).drop_duplicates("video_id")
        found.to_csv(raw_path, index=False)
    known = set().union(*[set(pd.read_csv(p).video_id) for p in args.exclude]) if args.exclude else set()
    keep = (
        found.duration_s.between(args.min_duration, args.max_duration)
        & ~found.is_short
        & ~found.title.fillna("").str.contains(EDITED)
        & found.title.fillna("").str.contains(DOGS)
        & ~found.title.fillna("").str.contains(TALK)
        & ~found.video_id.isin(known)
    )
    out = found[keep].drop(columns="is_short")
    out.to_csv(args.out, index=False)
    print(f"{len(found)} unique results; kept {len(out)} "
          f"({out.channel_id.nunique()} channels, {out.duration_s.sum() / 3600:.1f} h) -> {args.out}")


if __name__ == "__main__":
    main()
