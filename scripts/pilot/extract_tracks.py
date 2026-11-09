"""Pilot: extract per-frame dog tracks from downloaded videos with YOLO + ByteTrack.

Run in the pilot environment (see README):

    .venv-pilot/bin/python scripts/pilot/extract_tracks.py

Output follows the KLP track format. Positions are pixel coordinates of the
bounding-box bottom centre (roughly where the dog meets the ground).
``body_size`` is the square root of the box area, so window features can be
expressed in body sizes.

With ``--pose-model`` (a YOLO pose model trained on Dog-Pose, e.g. from
``train_dog_pose.py``), the pose model runs on a crop around each tracked box,
and the keypoints give:

- ``body_heading``: rear (tail start, else rear elbows) to front (front elbows).
- ``head_heading``: ear-base midpoint to nose.
- ``head_height``: nose height above the box bottom, in box heights.
- ``pose_compactness``: spread of confident keypoints around their centroid,
  in body sizes (low = curled or crouched).

Raw keypoints are kept as ``kp_<name>_{x,y,c}``. ``heading`` is
``body_heading`` where available and the direction of motion otherwise
(``heading_source`` says which). All angles are in the image plane, and camera
motion is not compensated, so positions and headings are rough.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

DOG_CLASS = 16  # COCO "dog"
PILOT_DIR = Path(__file__).resolve().parents[2] / "data" / "external" / "pilot"

# Dog-Pose keypoint order. Eyes, withers and throat are never labelled in the
# training data, so the model cannot learn them and they are not used below.
KEYPOINTS = [
    "front_left_paw", "front_left_knee", "front_left_elbow",
    "rear_left_paw", "rear_left_knee", "rear_left_elbow",
    "front_right_paw", "front_right_knee", "front_right_elbow",
    "rear_right_paw", "rear_right_knee", "rear_right_elbow",
    "tail_start", "tail_end", "left_ear_base", "right_ear_base", "nose", "chin",
    "left_ear_tip", "right_ear_tip", "left_eye", "right_eye", "withers", "throat",
]
KP = {name: i for i, name in enumerate(KEYPOINTS)}
UNLABELLED = {"left_eye", "right_eye", "withers", "throat"}
POSE_COLUMNS = ["body_heading", "head_heading", "head_height", "pose_compactness"]
# The detector has already found a dog in each crop; the pose model only has to place it.
POSE_DETECTION_CONF = 0.1


def _mean_point(xy, ok, names):
    """Mean of the confident keypoints among ``names``, or None."""
    idx = [KP[n] for n in names if ok[KP[n]]]
    return xy[idx].mean(axis=0) if idx else None


def _angle(start, end):
    return float(np.arctan2(end[1] - start[1], end[0] - start[0]))


def pose_features(xy: np.ndarray, conf: np.ndarray, box, min_conf: float) -> dict[str, float]:
    """Heading, head direction and posture from one dog's keypoints (NaN where unsupported)."""
    x1, y1, x2, y2 = box
    ok = conf > min_conf
    for name in UNLABELLED:
        ok[KP[name]] = False
    features = dict.fromkeys(POSE_COLUMNS, np.nan)

    front = _mean_point(xy, ok, ["front_left_elbow", "front_right_elbow"])
    rear = _mean_point(xy, ok, ["tail_start"]) if ok[KP["tail_start"]] else (
        _mean_point(xy, ok, ["rear_left_elbow", "rear_right_elbow"]))
    if front is not None and rear is not None:
        features["body_heading"] = _angle(rear, front)

    ears = _mean_point(xy, ok, ["left_ear_base", "right_ear_base"])
    if ears is not None and ok[KP["nose"]]:
        features["head_heading"] = _angle(ears, xy[KP["nose"]])
    if ok[KP["nose"]]:
        features["head_height"] = float((y2 - xy[KP["nose"], 1]) / (y2 - y1))

    if ok.sum() >= 5:
        pts = xy[ok]
        spread = np.sqrt(((pts - pts.mean(axis=0)) ** 2).sum(axis=1).mean())
        features["pose_compactness"] = float(spread / ((x2 - x1) * (y2 - y1)) ** 0.5)
    return features


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy boxes ``a`` (n, 4) and ``b`` (m, 4)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def crop_poses(pose_model, frame, boxes: np.ndarray, device: str, pad: float = 0.15, min_iou: float = 0.3):
    """Keypoints ``(xy, conf)`` in frame coordinates for each tracked box, or None.

    The pose model runs on a padded crop of each box rather than the full frame:
    Dog-Pose images mostly show one large dog, and on full multi-dog frames the
    model finds only about a third of the dogs the detector finds. Within a crop,
    the pose detection overlapping the tracked box most is used, so a
    neighbouring dog in the crop is not picked up.
    """
    h, w = frame.shape[:2]
    crops, origins = [], []
    for x1, y1, x2, y2 in boxes:
        px, py = pad * (x2 - x1), pad * (y2 - y1)
        cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
        cx2, cy2 = int(min(w, x2 + px)), int(min(h, y2 + py))
        crops.append(frame[cy1:cy2, cx1:cx2])
        origins.append(np.array([cx1, cy1], float))
    if not crops:
        return []
    poses = []
    for result, origin, box in zip(
        pose_model.predict(crops, conf=POSE_DETECTION_CONF, device=device, verbose=False), origins, boxes
    ):
        if result.keypoints is None or result.keypoints.conf is None or not len(result.boxes):
            poses.append(None)
            continue
        iou = _iou(box[None] - np.tile(origin, 2), result.boxes.xyxy.cpu().numpy())[0]
        j = int(iou.argmax())
        poses.append(None if iou[j] < min_iou else (
            result.keypoints.xy[j].cpu().numpy() + origin, result.keypoints.conf[j].cpu().numpy()))
    return poses


def _hsv_hist(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist)


def _reset_tracker(model):
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", None) or []:
        tracker.reset()


def extract_video(path: Path, model, sample_fps: float, conf: float, cut_threshold: float, preview_dir: Path,
                  device: str, pose_model=None, kpt_conf: float = 0.3):
    """Detect and track dogs in one video. A new shot starts at each detected cut."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / sample_fps))
    _reset_tracker(model)

    rows, frame_idx, shot, prev_hist, n_sampled, preview_saved = [], -1, 0, None, 0, False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % step:
            continue
        n_sampled += 1

        hist = _hsv_hist(frame)
        if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < cut_threshold:
            shot += 1
            _reset_tracker(model)
        prev_hist = hist

        result = model.track(
            frame, persist=True, classes=[DOG_CLASS], conf=conf, tracker="bytetrack.yaml", device=device,
            verbose=False,
        )[0]
        if result.boxes.id is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        poses = crop_poses(pose_model, frame, boxes, device) if pose_model is not None else [None] * len(boxes)
        if not preview_saved and len(result.boxes) >= 2:
            plot = result.plot()
            for pose in filter(None, poses):
                for (kx, ky), c in zip(*pose):
                    if c > kpt_conf:
                        cv2.circle(plot, (int(kx), int(ky)), 4, (0, 255, 255), -1)
            cv2.imwrite(str(preview_dir / f"{path.stem}.jpg"), plot)
            preview_saved = True
        for (x1, y1, x2, y2), tid, score, pose in zip(
            boxes.tolist(), result.boxes.id.int().tolist(), result.boxes.conf.tolist(), poses
        ):
            keypoints = {}
            if pose_model is not None:
                keypoints = dict.fromkeys(POSE_COLUMNS, np.nan)
                if pose is not None:
                    xy, kc = pose
                    keypoints = pose_features(xy, kc, (x1, y1, x2, y2), kpt_conf)
                    for name, (kx, ky), c in zip(KEYPOINTS, xy.tolist(), kc.tolist()):
                        keypoints.update({f"kp_{name}_x": kx, f"kp_{name}_y": ky, f"kp_{name}_c": c})
            rows.append({
                "shot_id": str(shot),
                "track_id": f"{shot}-{tid}",
                "time_s": frame_idx / fps,
                "x": (x1 + x2) / 2,
                "y": y2,
                "box_w": x2 - x1,
                "box_h": y2 - y1,
                "body_size": ((x2 - x1) * (y2 - y1)) ** 0.5,
                "visibility": score,
                **keypoints,
            })
    cap.release()
    stats = {"fps": fps, "sampled_frames": n_sampled, "shots": shot + 1}
    return pd.DataFrame(rows), stats


def add_heading(tracks: pd.DataFrame, min_speed: float = 0.02) -> pd.DataFrame:
    """Heading = pose body heading where available, else direction of smoothed motion.

    Motion heading is held while the dog is still. A dog counts as moving when it
    shifts more than ``min_speed`` box heights per sampled frame. Frames with
    neither (a track that never moves and has no pose) get NaN heading.
    """
    parts = []
    for _, track in tracks.groupby(["video_id", "track_id"], sort=False):
        track = track.sort_values("time_s").copy()
        x = track.x.rolling(3, center=True, min_periods=1).mean()
        y = track.y.rolling(3, center=True, min_periods=1).mean()
        dx, dy = x.diff(), y.diff()
        moving = np.hypot(dx, dy) > min_speed * track.box_h
        motion = pd.Series(np.where(moving, np.arctan2(dy, dx), np.nan), index=track.index).ffill().bfill()
        body = track["body_heading"] if "body_heading" in track else pd.Series(np.nan, index=track.index)
        track["heading"] = body.fillna(motion)
        track["heading_source"] = np.where(body.notna(), "pose", np.where(motion.notna(), "motion", None))
        parts.append(track)
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", type=Path, default=PILOT_DIR / "videos")
    parser.add_argument("--out", type=Path, default=PILOT_DIR / "tracks.parquet")
    parser.add_argument("--model", default=str(PILOT_DIR / "models" / "yolo11s.pt"))
    parser.add_argument("--device", default="mps", help="mps (Apple GPU), cuda, or cpu")
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--cut-threshold", type=float, default=0.5, help="HSV histogram correlation below this = cut")
    parser.add_argument("--pose-model", help="Dog-Pose YOLO weights, e.g. data/external/pose_train/"
                                             "yolo11s_dogpose/weights/best.pt; omit for motion heading only")
    parser.add_argument("--kpt-conf", type=float, default=0.3, help="Keypoint confidence needed to use a keypoint")
    args = parser.parse_args()

    preview_dir = args.out.parent / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    pose_model = YOLO(args.pose_model) if args.pose_model else None

    tables, video_stats = [], []
    videos = sorted(args.videos.glob("*.mp4"))
    for i, path in enumerate(videos, 1):
        info_path = path.with_suffix(".info.json")
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        tracks, stats = extract_video(path, model, args.sample_fps, args.conf, args.cut_threshold, preview_dir,
                                      args.device, pose_model, args.kpt_conf)
        video_stats.append({"video_id": path.stem, "channel_id": info.get("channel_id"), **stats})
        print(f"[{i}/{len(videos)}] {path.stem}: {stats['sampled_frames']} frames, {stats['shots']} shots, "
              f"{len(tracks)} detections", flush=True)
        if not tracks.empty:
            tables.append(tracks.assign(video_id=path.stem, source="pawgaze", channel_id=info.get("channel_id")))

    tracks = add_heading(pd.concat(tables, ignore_index=True))
    tracks.to_parquet(args.out, index=False)
    if pose_model is not None:
        print("Heading source: " + ", ".join(f"{k} {v:.0%}" for k, v in
                                              tracks.heading_source.value_counts(normalize=True).items()))
    pd.DataFrame(video_stats).to_csv(args.out.with_name("video_stats.csv"), index=False)
    print(f"Wrote {len(tracks)} detections to {args.out}")


if __name__ == "__main__":
    main()
