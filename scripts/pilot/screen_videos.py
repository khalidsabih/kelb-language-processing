"""Screen candidate videos for two or more visible dogs before tracking them.

    .venv-pilot/bin/python scripts/pilot/screen_videos.py --candidates data/external/search_candidates.csv \
        --out data/external/screen/screen.csv

For each video: download the lowest usable resolution (video only), run the YOLO dog
detector on ``--frames`` frames spread evenly through it, then delete the file (unless
``--keep``). Writes one row per video: the share of sampled frames with at least two
dogs, the mean number of dogs, and the share of frames with any dog. Rows are appended
as videos finish, and videos already in ``--out`` are skipped, so an interrupted run
can be resumed.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from yt_dlp import YoutubeDL

ROOT = Path(__file__).resolve().parents[2]
DOG_CLASS = 16  # COCO
FORMAT = "bv*[height<=240][vcodec^=avc1]/bv*[height<=360][vcodec^=avc1]/bv*[height<=360]/wv*"


def download(video_id: str, folder: Path, retry_after_s: float = 30.0) -> Path:
    """Download one video; retry once, since YouTube sometimes answers 403 for a while."""
    try:
        return _download(video_id, folder)
    except Exception:
        time.sleep(retry_after_s)
        return _download(video_id, folder)


def _download(video_id: str, folder: Path) -> Path:
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "format": FORMAT, "noplaylist": True,
            "outtmpl": str(folder / "%(id)s.%(ext)s")}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    return Path(ydl.prepare_filename(info))


def dog_counts(path: Path, model, n_frames: int, conf: float, device: str) -> list[int]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    counts = []
    # Skip the first and last 5%: intros and end screens.
    for index in np.linspace(0.05 * total, 0.95 * total, n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        result = model.predict(frame, classes=[DOG_CLASS], conf=conf, device=device, verbose=False)[0]
        counts.append(len(result.boxes))
    cap.release()
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", type=Path, required=True, help="CSV with a video_id column")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=str(ROOT / "data" / "external" / "pilot" / "models" / "yolo11s.pt"))
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between downloads")
    parser.add_argument("--keep", action="store_true", help="Keep the downloaded low-resolution videos")
    args = parser.parse_args()

    folder = args.out.parent / "videos"
    folder.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(args.out).video_id) if args.out.exists() else set()
    ids = [v for v in pd.read_csv(args.candidates).video_id.drop_duplicates() if v not in done]
    model = YOLO(args.model)
    for i, video_id in enumerate(ids, 1):
        row = {"video_id": video_id}
        try:
            path = download(video_id, folder)
            counts = np.array(dog_counts(path, model, args.frames, args.conf, args.device))
            row.update(frames=len(counts), two_plus_share=float((counts >= 2).mean()),
                       any_dog_share=float((counts >= 1).mean()), mean_dogs=float(counts.mean()), error="")
            if not args.keep:
                path.unlink()
        except Exception as exc:  # unavailable, rate-limited, unreadable
            row.update(frames=0, error=str(exc).splitlines()[0][:200])
        pd.DataFrame([row]).to_csv(args.out, mode="a", header=not args.out.exists(), index=False)
        print(f"[{i}/{len(ids)}] {video_id}: " + (f"2+ dogs in {row['two_plus_share']:.0%} of {row['frames']} frames"
                                                  if row["frames"] else row["error"]), flush=True)
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
