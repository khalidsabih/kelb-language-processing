"""Pilot: remove camera motion from extracted tracks.

    .venv-pilot/bin/python scripts/pilot/stabilize_tracks.py --tracks data/external/pilot/pose/tracks.parquet

Web footage is mostly handheld: when the camera pans, every dog moves in the
image at once, and that shared motion ends up in speeds and targets. For each
sampled frame this estimates the camera's motion since the previous sampled
frame from background corners tracked with optical flow (dog boxes masked out),
as a similarity transform (rotation, uniform scale, translation) fitted with
RANSAC. Chaining those within a shot maps every frame onto the first frame of
its shot, and track positions, body sizes and headings are re-expressed in
that frame.

Writes ``tracks_stabilized.parquet`` beside the input, with the original
columns replaced and ``camera_ok`` (False where the transform could not be
estimated and the previous one was reused). Limits: static overlays (captions,
logos) count as background, and a scene dominated by moving people or dogs can
fool RANSAC.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MAX_CORNERS = 300
MIN_INLIERS = 20
WORK_WIDTH = 640  # frames are downscaled to this width for flow


def _camera_steps(path: Path, fps: float, boxes: dict) -> pd.DataFrame:
    """Similarity transform from each sampled frame to the previous one, in full-frame pixels.

    Every sampled frame gets a row, including frames without dogs, since the camera
    can move between two frames in which the dogs are detected.
    """
    step = max(1, round(fps / 10))  # same sampling as extract_tracks.py
    cap = cv2.VideoCapture(str(path))
    rows, prev_gray, frame_idx, scale = [], None, -1, 1.0
    while True:
        if not cap.grab():
            break
        frame_idx += 1
        if frame_idx % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        scale = WORK_WIDTH / frame.shape[1]
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        time_s = frame_idx / fps
        matrix, good = np.eye(2, 3), False
        if prev_gray is not None:
            mask = np.full(prev_gray.shape, 255, np.uint8)
            for x1, y1, x2, y2 in boxes.get(round(prev_time, 3), []):
                mask[int(y1 * scale):int(y2 * scale) + 1, int(x1 * scale):int(x2 * scale) + 1] = 0
            corners = cv2.goodFeaturesToTrack(prev_gray, MAX_CORNERS, 0.01, 8, mask=mask)
            if corners is not None and len(corners) >= MIN_INLIERS:
                moved, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None)
                found = status.ravel() == 1
                if found.sum() >= MIN_INLIERS:
                    # Maps current-frame points to previous-frame points.
                    m, inliers = cv2.estimateAffinePartial2D(moved[found], corners[found], method=cv2.RANSAC,
                                                             ransacReprojThreshold=2.0)
                    if m is not None and inliers is not None and inliers.sum() >= MIN_INLIERS:
                        m[:, 2] /= scale  # translation back to full-frame pixels
                        matrix, good = m, True
        rows.append({"time_s": time_s, "m": matrix, "camera_ok": good or prev_gray is None})
        prev_gray, prev_time = gray, time_s
    cap.release()
    return pd.DataFrame(rows)


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2x3 affine for applying ``b`` first, then ``a``."""
    return (np.vstack([a, [0, 0, 1]]) @ np.vstack([b, [0, 0, 1]]))[:2]


def stabilize_video(video: pd.DataFrame, path: Path, fps: float) -> pd.DataFrame:
    boxes = {}
    for r in video.itertuples():
        boxes.setdefault(round(r.time_s, 3), []).append(
            (r.x - r.box_w / 2, r.y - r.box_h, r.x + r.box_w / 2, r.y))
    steps = _camera_steps(path, fps, boxes)
    steps["key"] = steps.time_s.round(3)

    out = []
    for shot_id, shot in video.groupby("shot_id"):
        keys = shot.time_s.round(3)
        first, last = keys.min(), keys.max()
        span = steps[(steps.key >= first) & (steps.key <= last)]
        cumulative, total, last_good = {}, np.eye(2, 3), np.eye(2, 3)
        for r in span.itertuples():
            if r.key == first:
                total = np.eye(2, 3)
            else:
                m = r.m if r.camera_ok else last_good
                total = _compose(total, m)  # frame t -> frame t-1 -> ... -> first frame of the shot
                if r.camera_ok:
                    last_good = r.m
            cumulative[r.key] = (total.copy(), r.camera_ok)

        shot = shot.copy()
        ms = [cumulative.get(k, (np.full((2, 3), np.nan), False)) for k in keys]
        mat = np.stack([m for m, _ in ms])
        xy = np.stack([shot.x, shot.y, np.ones(len(shot))], axis=1)
        new_xy = np.einsum("nij,nj->ni", mat, xy)
        s = np.hypot(mat[:, 0, 0], mat[:, 1, 0])
        rotation = np.arctan2(mat[:, 1, 0], mat[:, 0, 0])
        shot["x"], shot["y"] = new_xy[:, 0], new_xy[:, 1]
        shot["body_size"] = shot.body_size * s
        shot["heading"] = shot.heading + rotation
        for col in ("body_heading", "head_heading"):
            if col in shot:
                shot[col] = shot[col] + rotation
        shot["camera_ok"] = [ok for _, ok in ms]
        out.append(shot)
    return pd.concat(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracks", type=Path, required=True, help="tracks.parquet with video_stats.csv beside it")
    parser.add_argument("--videos", type=Path, help="Default: <batch dir>/videos, the batch dir holding the track dir")
    args = parser.parse_args()

    tracks = pd.read_parquet(args.tracks)
    stats = pd.read_csv(args.tracks.with_name("video_stats.csv")).set_index("video_id")
    videos = args.videos or args.tracks.parent.parent / "videos"
    if not videos.exists():
        videos = args.tracks.parent / "videos"

    parts = []
    for i, (video_id, video) in enumerate(tracks.groupby("video_id"), 1):
        part = stabilize_video(video, videos / f"{video_id}.mp4", stats.loc[video_id, "fps"])
        parts.append(part)
        print(f"[{i}/{tracks.video_id.nunique()}] {video_id}: camera estimated in {part.camera_ok.mean():.0%} of rows",
              flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(args.tracks.with_name("tracks_stabilized.parquet"), index=False)
    print(f"Wrote {len(out)} rows; camera estimated in {out.camera_ok.mean():.0%}")


if __name__ == "__main__":
    main()
