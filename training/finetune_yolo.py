"""Fine-tune YOLO on the 6-class intersection dataset (AV/BC/HDV/MC/PED/SCO).

Trains from the dataset_yolo split produced by prepare_dataset.py. Defaults
target 1080p frames with many small/distant objects: imgsz=1280, DDP on both
4090s. Best weights land in runs/finetune/yolo26_6class/weights/best.pt.
"""

from __future__ import annotations

import argparse
import os

# Windows DDP rendezvous uses libuv by default, but stock PyTorch Windows
# wheels are built without it. Disable before torch initializes.
os.environ.setdefault("USE_LIBUV", "0")

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--data", default="training/dataset_yolo/data.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0,1")
    ap.add_argument("--project", default="runs/finetune")
    ap.add_argument("--name", default="yolo26_6class")
    args = ap.parse_args()

    device = [int(d) for d in args.device.split(",")] if "," in args.device else args.device

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
