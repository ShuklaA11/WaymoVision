"""Phase 4: project pixel trajectories to a 2D metric map via homography.

Reads a GCP file (pixel <-> lat/lng correspondences) and a trajectory CSV
(smoothed pixel paths from trajectory.py). Builds a pixel->ground homography,
projects every trajectory point to local meters (ENU) and lat/lng, computes
speed, and writes:
  1. a map-coordinate trajectory CSV (east_m, north_m, lat, lng, speed)
  2. a top-down metric plot (meters) colored by class

Usage:
    python homography.py --gcp configs/gcp_sjb.json --traj "output/.../clip_trajectories.csv" --fps 25
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_COLORS = {
    "AV": "#e6194b", "HDV": "#3cb44b", "BC": "#4363d8",
    "MC": "#f58231", "PED": "#911eb4", "SCO": "#42d4f4",
}

M_PER_DEG_LAT = 110574.0


def latlng_to_local(lat, lng, lat0, lng0):
    """Equirectangular approximation: lat/lng -> local east/north meters."""
    east = (lng - lng0) * (111320.0 * math.cos(math.radians(lat0)))
    north = (lat - lat0) * M_PER_DEG_LAT
    return east, north


def local_to_latlng(east, north, lat0, lng0):
    lat = lat0 + north / M_PER_DEG_LAT
    lng = lng0 + east / (111320.0 * math.cos(math.radians(lat0)))
    return lat, lng


def build_homography(gcp: dict):
    """Pixel -> local-meter homography from GCP correspondences."""
    pts = [p for p in gcp["points"] if p.get("lat") is not None and p.get("lng") is not None]
    if len(pts) < 4:
        raise SystemExit(
            f"Need >= 4 points with lat/lng, got {len(pts)}. "
            "Fill the lat/lng fields in the GCP file (see pick_gcp.py)."
        )

    center = gcp.get("intersection_center")
    if center and center.get("lat") is not None:
        lat0, lng0 = center["lat"], center["lng"]
    else:  # fall back to the centroid of the points
        lat0 = sum(p["lat"] for p in pts) / len(pts)
        lng0 = sum(p["lng"] for p in pts) / len(pts)

    src, dst = [], []
    for p in pts:
        src.append(p["px"])
        e, n = latlng_to_local(p["lat"], p["lng"], lat0, lng0)
        dst.append([e, n])
    H, _ = cv2.findHomography(np.array(src, dtype=np.float64),
                              np.array(dst, dtype=np.float64), method=0)
    return H, lat0, lng0


def project(points_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply homography to Nx2 pixel points -> Nx2 meter points."""
    pts = points_px.reshape(-1, 1, 2).astype(np.float64)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def load_trajectories(path: str) -> dict[int, dict]:
    by_id = defaultdict(lambda: {"frames": [], "px": []})
    cls_of = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            tid = int(r["track_id"])
            by_id[tid]["frames"].append(int(r["frame"]))
            by_id[tid]["px"].append([float(r["smooth_x"]), float(r["smooth_y"])])
            cls_of[tid] = r["class"]
    out = {}
    for tid, d in by_id.items():
        out[tid] = {
            "class": cls_of[tid],
            "frames": np.array(d["frames"]),
            "px": np.array(d["px"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcp", required=True)
    ap.add_argument("--traj", required=True, help="trajectory CSV from trajectory.py")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-plot", default=None)
    args = ap.parse_args()

    traj_path = Path(args.traj)
    out_csv = Path(args.out_csv) if args.out_csv else traj_path.with_name(traj_path.stem.replace("_trajectories", "") + "_map.csv")
    out_plot = Path(args.out_plot) if args.out_plot else traj_path.with_name(traj_path.stem.replace("_trajectories", "") + "_map.png")

    gcp = json.loads(Path(args.gcp).read_text())
    H, lat0, lng0 = build_homography(gcp)

    tracks = load_trajectories(str(args.traj))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 12))
    seen = set()
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["track_id", "class", "frame", "px", "py", "east_m", "north_m", "lat", "lng", "speed_mps", "speed_mph"])
        for tid in sorted(tracks):
            t = tracks[tid]
            m = project(t["px"], H)  # Nx2 east,north
            frames = t["frames"]
            # Speed from consecutive smoothed metric points.
            speeds = np.zeros(len(m))
            for i in range(1, len(m)):
                d = np.hypot(*(m[i] - m[i - 1]))
                dt = (frames[i] - frames[i - 1]) / args.fps
                speeds[i] = d / dt if dt > 0 else 0.0
            if len(m) > 1:
                speeds[0] = speeds[1]

            color = CLASS_COLORS.get(t["class"], "#a9a9a9")
            lbl = t["class"] if t["class"] not in seen else None
            seen.add(t["class"])
            ax.plot(m[:, 0], m[:, 1], color=color, linewidth=1.6, label=lbl, alpha=0.9)

            for i in range(len(m)):
                lat, lng = local_to_latlng(m[i, 0], m[i, 1], lat0, lng0)
                wr.writerow([tid, t["class"], int(frames[i]),
                             f"{t['px'][i,0]:.1f}", f"{t['px'][i,1]:.1f}",
                             f"{m[i,0]:.2f}", f"{m[i,1]:.2f}",
                             f"{lat:.7f}", f"{lng:.7f}",
                             f"{speeds[i]:.2f}", f"{speeds[i]*2.23694:.1f}"])

    # Draw GCP box for reference.
    gcp_m = np.array([latlng_to_local(p["lat"], p["lng"], lat0, lng0) for p in gcp["points"]])
    ax.scatter(gcp_m[:, 0], gcp_m[:, 1], c="black", marker="x", s=80, zorder=5, label="GCP")

    ax.set_aspect("equal")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Top-down trajectories (meters from intersection center)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=110)
    plt.close(fig)

    print(f"Tracks: {len(tracks)}  ->  {out_csv}")
    print(f"Top-down plot: {out_plot}")


if __name__ == "__main__":
    main()
