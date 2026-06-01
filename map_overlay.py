"""Phase 5: draw geo-referenced trajectories on a 2D satellite map.

Reads the map CSV from homography.py (lat/lng per point), auto-fits an ESRI
World Imagery satellite tile to the trajectory extent, and overlays each
object's path colored by class. This is the "trajectories on a 2D map" view.

Usage:
    python map_overlay.py --map "output/.../clip_map.csv"
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_COLORS = {
    "AV": "#e6194b", "HDV": "#3cb44b", "BC": "#4363d8",
    "MC": "#f58231", "PED": "#911eb4", "SCO": "#42d4f4",
}

ESRI = ("https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
        "MapServer/export?bbox={xmin},{ymin},{xmax},{ymax}"
        "&bboxSR=4326&imageSR=4326&size={w},{h}&format=png&f=image")


def load_map(path: str):
    by_id = defaultdict(lambda: {"lat": [], "lng": []})
    cls_of = {}
    for r in csv.DictReader(open(path, newline="")):
        tid = int(r["track_id"])
        by_id[tid]["lat"].append(float(r["lat"]))
        by_id[tid]["lng"].append(float(r["lng"]))
        cls_of[tid] = r["class"]
    return by_id, cls_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="map CSV from homography.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--margin", type=float, default=0.15, help="extra padding around trajectory extent")
    ap.add_argument("--width", type=int, default=1280, help="satellite image width (px)")
    args = ap.parse_args()

    map_path = Path(args.map)
    out_png = Path(args.out) if args.out else map_path.with_name(map_path.stem.replace("_map", "") + "_mapoverlay.png")

    by_id, cls_of = load_map(str(map_path))
    all_lat = [v for d in by_id.values() for v in d["lat"]]
    all_lng = [v for d in by_id.values() for v in d["lng"]]
    lat_min, lat_max = min(all_lat), max(all_lat)
    lng_min, lng_max = min(all_lng), max(all_lng)

    # Pad the extent.
    dlat = (lat_max - lat_min) or 1e-4
    dlng = (lng_max - lng_min) or 1e-4
    lat_min -= dlat * args.margin; lat_max += dlat * args.margin
    lng_min -= dlng * args.margin; lng_max += dlng * args.margin
    lat_span, lng_span = lat_max - lat_min, lng_max - lng_min

    # Image size with a true metric aspect ratio.
    lat0 = (lat_min + lat_max) / 2
    lng_span_m = lng_span * 111320.0 * math.cos(math.radians(lat0))
    lat_span_m = lat_span * 110574.0
    W = args.width
    H = max(1, int(W * lat_span_m / lng_span_m))

    sat_path = map_path.with_name(map_path.stem + "_sat.png")
    url = ESRI.format(xmin=lng_min, ymin=lat_min, xmax=lng_max, ymax=lat_max, w=W, h=H)
    urlretrieve(url, sat_path)
    sat = cv2.cvtColor(cv2.imread(str(sat_path)), cv2.COLOR_BGR2RGB)

    def to_px(lat, lng):
        x = (lng - lng_min) / lng_span * W
        y = (lat_max - lat) / lat_span * H
        return x, y

    fig, ax = plt.subplots(figsize=(W / 110, H / 110))
    ax.imshow(sat, extent=[0, W, H, 0])
    seen = set()
    for tid, d in by_id.items():
        xs, ys = zip(*[to_px(la, ln) for la, ln in zip(d["lat"], d["lng"])])
        c = CLASS_COLORS.get(cls_of[tid], "#a9a9a9")
        lbl = cls_of[tid] if cls_of[tid] not in seen else None
        seen.add(cls_of[tid])
        ax.plot(xs, ys, color=c, linewidth=1.6, alpha=0.9, label=lbl)
        ax.scatter(xs[0], ys[0], color=c, s=14)  # start marker

    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Trajectories on 2D map (ESRI World Imagery)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    print(f"Tracks: {len(by_id)}  satellite: {sat_path}")
    print(f"Map overlay: {out_png}")


if __name__ == "__main__":
    main()
