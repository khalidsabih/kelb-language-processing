"""Pilot: fine-tune a YOLO pose model on the Ultralytics Dog-Pose dataset.

    .venv-pilot/bin/python scripts/pilot/train_dog_pose.py

Dog-Pose (6,773 train / 1,703 val Stanford Dogs images, 24 keypoints) downloads
to ``datasets/`` on first use. Starts from a human-pose checkpoint, so detection,
pose and tracking can later run in one pass. Weights land in
``data/external/pose_train/<name>/weights/best.pt``.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="yolo11s-pose.pt", help="Starting checkpoint")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15, help="Stop after this many epochs without improvement")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--name", default="yolo11s_dogpose")
    args = parser.parse_args()

    model = YOLO(str(ROOT / "data" / "external" / "pilot" / "models" / args.model))
    model.train(
        data="dog-pose.yaml",
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "data" / "external" / "pose_train"),
        name=args.name,
        exist_ok=True,
        seed=42,
    )


if __name__ == "__main__":
    main()
