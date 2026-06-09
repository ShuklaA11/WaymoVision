"""Standalone multi-class tracker with persistent (flicker-free) boxes.

Runs the fine-tuned 6-class YOLO + tuned ByteTrack on one video, fills short
detection gaps so boxes never blink out, and writes:
  1. an annotated MP4 (stable per-ID colors + class labels)
  2. a per-frame track CSV (foundation for trajectory smoothing in Phase 3)

Usage:
    python track_video.py --video "output/sjb/night/clip.mp4"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from waymo_pipeline.consolidate import consolidate_track

# Fine-tuned class order (alphabetical, from prepare_dataset.py)
CLASS_NAMES = ["AV", "BC", "HDV", "MC", "PED", "SCO"]


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color per track ID — stable color == stable ID."""
    h = hashlib.md5(str(track_id).encode()).digest()
    return int(h[0]), int(h[1]), int(h[2])


def _interp(p0, p1, t):
    return tuple(a + (b - a) * t for a, b in zip(p0, p1))


def fill_gaps(frames_by_id: dict[int, dict], max_gap: int) -> dict[int, dict]:
    """Linearly interpolate bbox across internal gaps <= max_gap frames.

    Makes a box persist through momentary missed detections instead of
    flickering off. Only fills gaps bounded by real detections on both sides.
    """
    for tid, frames in frames_by_id.items():
        nums = sorted(frames.keys())
        for a, b in zip(nums, nums[1:]):
            gap = b - a
            if 1 < gap <= max_gap:
                box_a, box_b = frames[a]["bbox"], frames[b]["bbox"]
                cls = frames[a]["cls"]
                for k in range(1, gap):
                    t = k / gap
                    frames[a + k] = {
                        "bbox": _interp(box_a, box_b, t),
                        "cls": cls,
                        "conf": 0.0,
                        "interp": True,
                    }
    return frames_by_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="models/yolo26_6class.pt")
    ap.add_argument("--tracker", default="waymo_pipeline/trackers/bytetrack_persistent.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    # Low conf so weak boxes reach ByteTrack's 2nd-stage association (anti-flicker)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-gap", type=int, default=30, help="max frames to hold/interpolate a missing box")
    ap.add_argument("--min-frames", type=int, default=8, help="drop tracks shorter than this (removes ghost/flicker tracks)")
    ap.add_argument("--trail-len", type=int, default=30, help="frames of fading trajectory trail to draw behind each object (0 = off)")
    ap.add_argument("--min-displacement", type=float, default=40.0, help="path movement (px) required for an AV label; stationary objects can't be AV")
    ap.add_argument("--av-min-ratio", type=float, default=0.60, help="min AV-frame fraction to keep an AV label")
    ap.add_argument("--av-min-conf", type=float, default=0.55, help="min mean confidence to keep an AV label")
    ap.add_argument("--drop-stationary", action="store_true", help="also remove non-moving tracks from the video (off by default so stopped/queued cars stay boxed)")
    ap.add_argument("--out", default=None, help="annotated mp4 (default: <video>_tracked.mp4)")
    ap.add_argument("--csv", default=None, help="per-frame track dump (default: <video>_tracks.csv)")
    args = ap.parse_args()

    video = Path(args.video)
    out_mp4 = Path(args.out) if args.out else video.with_name(video.stem + "_tracked.mp4")
    out_csv = Path(args.csv) if args.csv else video.with_name(video.stem + "_tracks.csv")

    model = YOLO(args.model)

    # Pass 1: track, collecting bbox/class/conf per (track_id, frame).
    frames_by_id: dict[int, dict] = defaultdict(dict)
    frame_idx = 0
    for result in model.track(
        source=str(video),
        stream=True,
        tracker=args.tracker,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=False,
        persist=False,
    ):
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                if box.id is None:
                    continue
                tid = int(box.id.item())
                frames_by_id[tid][frame_idx] = {
                    "bbox": tuple(box.xyxy[0].tolist()),
                    "cls": int(box.cls.item()),
                    "conf": float(box.conf.item()),
                    "interp": False,
                }
        frame_idx += 1
    n_frames = frame_idx

    # Drop ephemeral tracks (night reflections / brief false positives) that
    # would visually flicker as one- or two-frame boxes.
    frames_by_id = {tid: fr for tid, fr in frames_by_id.items()
                    if len(fr) >= args.min_frames}

    # Fill short gaps so boxes persist through missed detections.
    frames_by_id = fill_gaps(frames_by_id, args.max_gap)

    # Consolidate: drop stationary/ghost tracks, assign one stable label each.
    label_of: dict[int, str] = {}
    for tid, frames in list(frames_by_id.items()):
        ordered = [frames[fn] for fn in sorted(frames)]
        pts = [((d["bbox"][0] + d["bbox"][2]) / 2, d["bbox"][3]) for d in ordered]
        classes = [CLASS_NAMES[d["cls"]] if 0 <= d["cls"] < len(CLASS_NAMES) else str(d["cls"]) for d in ordered]
        confs = [d["conf"] for d in ordered]
        keep, label = consolidate_track(
            pts, classes, confs,
            min_displacement=args.min_displacement,
            av_min_ratio=args.av_min_ratio,
            av_min_conf=args.av_min_conf,
            drop_stationary=args.drop_stationary,
        )
        if keep:
            label_of[tid] = label
        else:
            del frames_by_id[tid]

    # Reindex by frame for rendering.
    by_frame: dict[int, list] = defaultdict(list)
    for tid, frames in frames_by_id.items():
        for fnum, d in frames.items():
            by_frame[fnum].append((tid, d))

    # Per-track trail points: box CENTER, moving-average smoothed so the line
    # stays on the car and doesn't spike when the box jumps (glare at lights,
    # brief occlusions). Map projection still uses bottom-center separately.
    def _smooth_xy(seq, w=9):
        half = w // 2
        out = []
        for i in range(len(seq)):
            lo, hi = max(0, i - half), min(len(seq), i + half + 1)
            xs = [p[0] for p in seq[lo:hi]]
            ys = [p[1] for p in seq[lo:hi]]
            out.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        return out

    trail_pts: dict[int, dict[int, tuple[int, int]]] = {}
    for tid, frames in frames_by_id.items():
        fns = sorted(frames)
        centers = [((frames[f]["bbox"][0] + frames[f]["bbox"][2]) / 2,
                    (frames[f]["bbox"][1] + frames[f]["bbox"][3]) / 2) for f in fns]
        sm = _smooth_xy(centers)
        trail_pts[tid] = {f: (int(sm[i][0]), int(sm[i][1])) for i, f in enumerate(fns)}

    # Pass 2: re-read video and draw persistent boxes.
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    fnum = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, d in by_frame.get(fnum, []):
            color = _color_for_id(tid)
            # Fading trajectory trail: recent bottom-center points up to now.
            if args.trail_len > 0:
                pts = [trail_pts[tid].get(f) for f in range(fnum - args.trail_len, fnum + 1)]
                pts = [p for p in pts if p is not None]
                for k in range(1, len(pts)):
                    fade = k / len(pts)  # older = fainter/thinner
                    col = tuple(int(c * (0.3 + 0.7 * fade)) for c in color)
                    cv2.line(frame, pts[k - 1], pts[k], col, max(1, int(1 + 3 * fade)), cv2.LINE_AA)

            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cls_name = label_of[tid]
            thickness = 1 if d["interp"] else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"{cls_name}#{tid}"
            cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        writer.write(frame)
        fnum += 1

    cap.release()
    writer.release()

    # Dump per-frame tracks.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "track_id", "class", "x1", "y1", "x2", "y2", "conf", "interpolated"])
        for fnum in sorted(by_frame):
            for tid, d in by_frame[fnum]:
                x1, y1, x2, y2 = d["bbox"]
                cls_name = label_of[tid]
                wr.writerow([fnum, tid, cls_name, f"{x1:.1f}", f"{y1:.1f}",
                             f"{x2:.1f}", f"{y2:.1f}", f"{d['conf']:.3f}", int(d["interp"])])

    n_tracks = len(frames_by_id)
    n_av = sum(1 for lbl in label_of.values() if lbl == "AV")
    print(f"Frames: {n_frames}  tracks: {n_tracks}  (AV: {n_av})")
    print(f"Annotated video: {out_mp4}")
    print(f"Track CSV: {out_csv}")


if __name__ == "__main__":
    main()
