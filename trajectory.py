"""Phase 3: extract and smooth object trajectories from a per-frame track CSV.

Reads the CSV emitted by track_video.py, takes each object's bottom-center
(ground-contact) point per frame, smooths the x(t)/y(t) series with a
Savitzky-Golay filter, and writes:
  1. a trajectory CSV (raw + smoothed pixel coords per frame)
  2. an overlay plot of smoothed paths on a representative frame

Usage:
    python trajectory.py --csv "output/.../clip_tracks.csv" --video "output/.../clip.mp4"
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

from waymo_pipeline.consolidate import consolidate_track

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Per-class plot colors (matplotlib RGB).
CLASS_COLORS = {
    "AV": "#e6194b",   # red — the Waymo
    "HDV": "#3cb44b",  # green
    "BC": "#4363d8",   # blue
    "MC": "#f58231",   # orange
    "PED": "#911eb4",  # purple
    "SCO": "#42d4f4",  # cyan
}


def _smooth(vals: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay smoothing with window adapted to the series length.

    Short tracks can't support the full window; shrink it (keeping it odd and
    > polyorder) or return the raw values when there aren't enough points.
    """
    n = len(vals)
    if n <= poly + 1:
        return vals
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w <= poly:
        return vals
    return savgol_filter(vals, w, poly)


def load_tracks(csv_path: str) -> dict[int, dict]:
    """Group CSV rows by track_id -> frame-sorted trajectory points.

    Bottom-center (x_mid, y2) is the ground-contact anchor; per-frame class and
    confidence are retained for track consolidation.
    """
    by_id: dict[int, list] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            tid = int(r["track_id"])
            x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            conf = float(r["conf"]) if "conf" in r and r["conf"] != "" else 0.0
            by_id[tid].append((int(r["frame"]), (x1 + x2) / 2.0, y2, r["class"], conf))

    tracks = {}
    for tid, pts in by_id.items():
        pts.sort(key=lambda p: p[0])
        tracks[tid] = {
            "frames": np.array([p[0] for p in pts]),
            "x": np.array([p[1] for p in pts]),
            "y": np.array([p[2] for p in pts]),
            "classes": [p[3] for p in pts],
            "confs": [p[4] for p in pts],
        }
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="per-frame track CSV from track_video.py")
    ap.add_argument("--video", default=None, help="source clip, used as plot background")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-plot", default=None)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--window", type=int, default=15, help="Savitzky-Golay window (frames)")
    ap.add_argument("--poly", type=int, default=2, help="Savitzky-Golay polynomial order")
    ap.add_argument("--min-displacement", type=float, default=40.0, help="drop tracks moving less than this (px); 0 = keep stationary")
    ap.add_argument("--av-min-ratio", type=float, default=0.60, help="min AV-frame fraction to keep an AV label")
    ap.add_argument("--av-min-conf", type=float, default=0.55, help="min mean confidence to keep an AV label")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_csv = Path(args.out_csv) if args.out_csv else csv_path.with_name(csv_path.stem.replace("_tracks", "") + "_trajectories.csv")
    out_plot = Path(args.out_plot) if args.out_plot else csv_path.with_name(csv_path.stem.replace("_tracks", "") + "_trajectories.png")

    tracks = load_tracks(str(csv_path))
    tracks = {tid: t for tid, t in tracks.items() if len(t["frames"]) >= args.min_frames}

    # Consolidate: drop stationary/ghost tracks, assign one stable label each.
    kept = {}
    for tid, t in tracks.items():
        pts = list(zip(t["x"].tolist(), t["y"].tolist()))
        keep, label = consolidate_track(
            pts, t["classes"], t["confs"],
            min_displacement=args.min_displacement,
            av_min_ratio=args.av_min_ratio,
            av_min_conf=args.av_min_conf,
        )
        if keep:
            t["class"] = label
            kept[tid] = t
    tracks = kept

    # Smooth each trajectory.
    for t in tracks.values():
        t["sx"] = _smooth(t["x"], args.window, args.poly)
        t["sy"] = _smooth(t["y"], args.window, args.poly)

    # Write trajectory data.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["track_id", "class", "frame", "raw_x", "raw_y", "smooth_x", "smooth_y"])
        for tid in sorted(tracks):
            t = tracks[tid]
            for i in range(len(t["frames"])):
                wr.writerow([tid, t["class"], int(t["frames"][i]),
                             f"{t['x'][i]:.1f}", f"{t['y'][i]:.1f}",
                             f"{t['sx'][i]:.1f}", f"{t['sy'][i]:.1f}"])

    # Plot smoothed paths over a representative frame.
    bg = None
    h = w = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        ok, frame = cap.read()
        cap.release()
        if ok:
            bg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = bg.shape[:2]

    fig, ax = plt.subplots(figsize=(16, 9))
    if bg is not None:
        ax.imshow(bg, alpha=0.55)
    seen_labels = set()
    for tid in sorted(tracks):
        t = tracks[tid]
        color = CLASS_COLORS.get(t["class"], "#a9a9a9")
        lbl = t["class"] if t["class"] not in seen_labels else None
        seen_labels.add(t["class"])
        ax.plot(t["sx"], t["sy"], color=color, linewidth=1.8, label=lbl, alpha=0.9)
        ax.scatter(t["sx"][0], t["sy"][0], color=color, s=18, marker="o")  # start
    if w:
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
    ax.set_title(f"Smoothed trajectories ({len(tracks)} tracks)")
    ax.legend(loc="upper right", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=110)
    plt.close(fig)

    print(f"Tracks: {len(tracks)}  ->  {out_csv}")
    print(f"Plot: {out_plot}")


if __name__ == "__main__":
    main()
