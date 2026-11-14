"""Stage 1 - motion filter: drop the parts of the raw footage where nothing moves.

Cheap enough to run on the box: every few frames it shrinks the image, compares it with the
previous sample and counts the pixels that changed. Spans with no change are never looked at
again (and, on the box, would never be uploaded).

Usage:
  uv run python -m relentless.motion VIDEO
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import media


@dataclass
class MotionConfig:
    sample_fps: float = 10.0      # how often to look
    width: int = 480              # analysis width in px (the frame is shrunk to this)
    pixel_delta: int = 10         # grey-level change (0-255) that counts as a changed pixel
    min_fraction: float = 0.00035  # share of changed pixels that counts as motion (about 45 px at 480x270)
    hold_s: float = 1.0           # keep this long before and after any motion
    merge_gap_s: float = 2.0      # join kept spans separated by less than this
    min_span_s: float = 1.0       # drop kept spans shorter than this


def spans_from_mask(active: np.ndarray, frames: np.ndarray, fps: float, first: int, last: int,
                    hold_s: float, merge_gap_s: float, min_span_s: float) -> list[tuple[int, int]]:
    """Turn a per-sample on/off mask into padded, merged [start, end) frame spans."""
    spans: list[list[int]] = []
    hold = int(round(hold_s * fps))
    step = int(frames[1] - frames[0]) if len(frames) > 1 else 1
    for f, on in zip(frames, active):
        if not on:
            continue
        a, b = max(first, int(f) - hold), min(last, int(f) + step + hold)
        if spans and a - spans[-1][1] <= merge_gap_s * fps:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return [(a, b) for a, b in spans if b - a >= min_span_s * fps]


def scan(src: Path, first: int = 0, last: int | None = None, cfg: MotionConfig | None = None,
         progress=None) -> dict:
    """Score motion over frames [first, last) and return the spans worth keeping."""
    cfg = cfg or MotionConfig()
    with media.Reader(src) as rd:
        fps = rd.fps
        last = min(last or rd.count, rd.count)
        step = max(1, int(round(fps / cfg.sample_fps)))
        size = (cfg.width, media.even(cfg.width * rd.height / rd.width))
        kernel = np.ones((3, 3), np.uint8)
        idx, scores, prev = [], [], None
        for i, frame in rd.iter(range(first, last, step)):
            small = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            small = cv2.GaussianBlur(small, (5, 5), 0)
            if prev is None:
                score = 0.0
            else:
                changed = (cv2.absdiff(small, prev) > cfg.pixel_delta).astype(np.uint8)
                changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, kernel)  # drop single-pixel sensor noise
                score = float(changed.mean())
            prev = small
            idx.append(i)
            scores.append(score)
            if progress and len(idx) % 100 == 0:
                progress((i - first) / max(1, last - first))
    frames = np.array(idx, dtype=np.int64)
    score = np.array(scores, dtype=np.float32)
    if len(score) > 1:
        score[0] = score[1]  # the first sample has nothing to compare with
    spans = spans_from_mask(score >= cfg.min_fraction, frames, fps, first, last,
                            cfg.hold_s, cfg.merge_gap_s, cfg.min_span_s)
    kept = sum(b - a for a, b in spans)
    return {
        "fps": fps, "first": first, "last": last, "step": step,
        "frames": frames, "score": score, "spans": spans,
        "total_frames": last - first, "kept_frames": kept,
        "kept_fraction": kept / max(1, last - first),
        "config": asdict(cfg),
    }


def summary(result: dict, bins: int = 240) -> dict:
    """JSON-friendly digest for the run report: spans in seconds plus a coarse motion timeline."""
    fps, first, last = result["fps"], result["first"], result["last"]
    score = result["score"]
    edges = np.linspace(0, len(score), min(bins, max(1, len(score))) + 1).astype(int)
    timeline = [float(score[a:b].max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    return {
        "seconds": (last - first) / fps,
        "kept_seconds": result["kept_frames"] / fps,
        "kept_fraction": result["kept_fraction"],
        "spans": [[(a - first) / fps, (b - first) / fps] for a, b in result["spans"]],
        "frame_spans": [[a, b] for a, b in result["spans"]],
        "timeline": timeline,
        "threshold": result["config"]["min_fraction"],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--seconds", type=float)
    a = ap.parse_args(argv)
    info = media.probe(a.video)
    first = int(a.start * info["fps"])
    last = first + int(a.seconds * info["fps"]) if a.seconds else None
    r = scan(a.video, first, last)
    s = summary(r)
    print(f"{a.video.name}: kept {s['kept_seconds']:.1f}s of {s['seconds']:.1f}s "
          f"({s['kept_fraction']:.0%}) in {len(s['spans'])} span(s)")
    for a_, b_ in s["spans"]:
        print(f"  {a_:7.1f}s - {b_:7.1f}s")
    q = np.percentile(r["score"], [5, 25, 50, 75, 95])
    print("  motion score percentiles 5/25/50/75/95: " + " ".join(f"{v:.5f}" for v in q))


if __name__ == "__main__":
    main()
