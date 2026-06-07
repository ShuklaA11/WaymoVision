"""Track consolidation: drop stationary/ghost tracks and stabilize labels.

Shared by track_video.py and trajectory.py so the video and the trajectory
data agree. Two cleanups, both data-driven:
  1. Stationary filter   - drop tracks whose path barely moves (parked cars,
                           static false positives, night-reflection ghosts).
  2. Class vote + AV gate - one stable label per track by majority vote, but
                           only keep "AV" when the evidence is strong (high AV
                           frame fraction AND mean confidence). Precision over
                           recall, matching the project's philosophy.
"""

from __future__ import annotations

import math
from collections import Counter


def path_extent(points: list[tuple[float, float]]) -> float:
    """Bounding-box diagonal of the path (pixels) — robust movement measure."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def consolidate_track(
    points: list[tuple[float, float]],
    classes: list[str],
    confs: list[float],
    *,
    min_displacement: float = 40.0,
    av_min_ratio: float = 0.60,
    av_min_conf: float = 0.55,
    drop_stationary: bool = True,
) -> tuple[bool, str | None]:
    """Decide whether to keep a track and what single label to give it.

    Returns (keep, label); label is None when the track is dropped.

    Movement (path extent >= min_displacement) is required for an AV label —
    a real Waymo passes through, so a stationary "AV" is almost always a white
    reflection/parked car misread.

    drop_stationary controls what happens to non-moving tracks:
      - True  (analysis): drop them — a parked car has no useful trajectory.
      - False (video): keep them (real stopped/queued cars stay boxed), but
        they can never be labeled AV.
    """
    if not points:
        return False, None

    moved = min_displacement <= 0 or path_extent(points) >= min_displacement
    if drop_stationary and not moved:
        return False, None

    av_frac = sum(1 for c in classes if c == "AV") / len(classes)
    real = [c for c in confs if c > 0]
    mean_conf = sum(real) / len(real) if real else 0.0

    if moved and av_frac >= av_min_ratio and mean_conf >= av_min_conf:
        return True, "AV"

    non_av = [c for c in classes if c != "AV"]
    label = Counter(non_av).most_common(1)[0][0] if non_av else "HDV"
    return True, label
