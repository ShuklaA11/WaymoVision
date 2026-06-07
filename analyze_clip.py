"""End-to-end clip analysis with clean, per-video/per-clip output.

Runs the full chain for each clip and writes tidy artifacts under
output/analysis/<video>/<clip>/:

    tracked.mp4         annotated video (persistent boxes, labels, trails)
    tracks.csv          per-frame detections
    trajectories.csv    smoothed pixel trajectories
    trajectories.png    pixel-space trajectory plot
    map.csv             metric map coords (east/north m, lat/lng) + speed
    map.png             top-down metric plot
    overlay.png         trajectories on the satellite 2D map
    summary.txt         per-clip counts and speed stats

The camera (and its GCP file) is inferred from the clip path: SJB vs Speedway.

Usage:
    python analyze_clip.py "output/sjb/night/clip.mp4"
    python analyze_clip.py --glob "output/sjb/night/*.mp4"
    python analyze_clip.py --glob "output/sjb/**/*_clip001_*.mp4"
"""

from __future__ import annotations

import argparse
import csv
import glob as globlib
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLIP_RE = re.compile(r"waymo_(.+)_clip(\d+)_(day|night)", re.IGNORECASE)


def parse_names(clip_path: str) -> tuple[str, str]:
    """(video_stem, clip_label) from a 'waymo_<video>_clipNNN_<tod>' filename."""
    stem = Path(clip_path).stem
    m = CLIP_RE.match(stem)
    if m:
        return m.group(1), f"clip{m.group(2)}_{m.group(3).lower()}"
    return stem, "clip"  # fallback for non-standard names


def infer_gcp(clip_path: str, override: str | None) -> str | None:
    if override:
        return override
    p = clip_path.lower()
    if "speedway" in p:
        return "configs/gcp_speedway.json"
    if "sjb" in p:
        return "configs/gcp_sjb.json"
    return None


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def write_summary(outdir: Path, tracks_csv: Path, map_csv: Path, clip: str,
                  video: str, clip_label: str) -> None:
    rows = list(csv.DictReader(open(tracks_csv, newline="")))
    by_id = defaultdict(list)
    for r in rows:
        by_id[int(r["track_id"])].append(r)
    n_frames = (max(int(r["frame"]) for r in rows) + 1) if rows else 0
    labels = Counter(v[0]["class"] for v in by_id.values())

    speed_lines = []
    if map_csv.exists():
        mrows = list(csv.DictReader(open(map_csv, newline="")))
        sp = defaultdict(list)
        cls = {}
        for r in mrows:
            sp[int(r["track_id"])].append(float(r["speed_mph"]))
            cls[int(r["track_id"])] = r["class"]
        med = sorted(((sorted(s)[len(s) // 2], tid) for tid, s in sp.items()), reverse=True)
        for m, tid in med[:5]:
            speed_lines.append(f"    track {tid} ({cls[tid]}): median {m:.1f} mph")

    lines = [
        f"video:  {video}",
        f"clip:   {clip_label}",
        f"source: {clip}",
        f"frames: {n_frames}",
        f"tracks: {len(by_id)}",
        "labels: " + ", ".join(f"{k}={v}" for k, v in sorted(labels.items())),
        "fastest tracks:",
        *speed_lines,
    ]
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n")


def analyze_one(clip: str, args) -> bool:
    video, clip_label = parse_names(clip)
    gcp = infer_gcp(clip, args.gcp)
    outdir = Path(args.out_root) / video / clip_label
    outdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    tracked, tracks = outdir / "tracked.mp4", outdir / "tracks.csv"
    traj_csv, traj_png = outdir / "trajectories.csv", outdir / "trajectories.png"
    map_csv, map_png = outdir / "map.csv", outdir / "map.png"
    overlay = outdir / "overlay.png"

    print(f"\n=== {video} / {clip_label} -> {outdir} ===")
    run([py, "track_video.py", "--video", clip, "--device", args.device,
         "--imgsz", str(args.imgsz), "--trail-len", str(args.trail_len),
         "--out", str(tracked), "--csv", str(tracks)])
    run([py, "trajectory.py", "--csv", str(tracks), "--video", clip,
         "--out-csv", str(traj_csv), "--out-plot", str(traj_png)])

    if gcp and Path(gcp).exists():
        run([py, "homography.py", "--gcp", gcp, "--traj", str(traj_csv),
             "--fps", str(args.fps), "--out-csv", str(map_csv), "--out-plot", str(map_png)])
        run([py, "map_overlay.py", "--map", str(map_csv), "--out", str(overlay)])
    else:
        print(f"  (skipping map stages — no GCP file for this camera: {gcp})")

    write_summary(outdir, tracks, map_csv, clip, video, clip_label)
    print(f"  done -> {outdir}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*", help="clip file path(s)")
    ap.add_argument("--glob", default=None, help="glob pattern for many clips")
    ap.add_argument("--out-root", default="output/analysis")
    ap.add_argument("--gcp", default=None, help="override GCP file (else inferred SJB/Speedway)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--trail-len", type=int, default=30)
    args = ap.parse_args()

    clips = list(args.clips)
    if args.glob:
        clips += sorted(globlib.glob(args.glob, recursive=True))
    clips = [c for c in clips if not Path(c).stem.endswith("_tracked")]
    if not clips:
        raise SystemExit("No clips given. Pass paths or --glob.")

    ok = 0
    for i, clip in enumerate(clips, 1):
        print(f"\n[{i}/{len(clips)}] {clip}")
        try:
            analyze_one(clip, args)
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {e}")
    print(f"\nDone: {ok}/{len(clips)} clip(s) -> {args.out_root}")


if __name__ == "__main__":
    main()
