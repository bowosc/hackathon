"""Stage 2 - people: find and follow every person in the footage that survived the motion filter.

One pose model pass gives, for every sampled frame, each person's box and 17 body keypoints.
ByteTrack links the boxes over time into tracklets (one per person-appearance). The same
keypoints later drive the activity filter (wrists), the hand zoom and the face blur, so the
expensive model runs exactly once per sampled frame.

Usage:
  uv run python -m relentless.people VIDEO --start 60 --seconds 30
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import media
from .config import MODELS, POSE_WEIGHTS, device, setting

# COCO keypoint indices used across the engine.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP = 11, 12
FACE = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR]
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]


@dataclass
class PeopleConfig:
    sample_fps: float = 10.0   # frames per second sent through the pose model
    # model input size; CCTV people are small, so keep it high. POSE_IMGSZ lowers it on a CPU-only machine.
    imgsz: int = field(default_factory=lambda: int(setting("POSE_IMGSZ", "1280")))
    batch: int = 8
    det_conf: float = 0.1      # ByteTrack wants low-confidence boxes too
    track_high: float = 0.3
    track_new: float = 0.4
    track_buffer_s: float = 3.0  # how long a lost person keeps their id
    max_gap_s: float = 1.0     # gaps up to this are interpolated; longer ones split the tracklet
    min_seconds: float = 1.5   # shorter appearances are dropped
    min_height: int = 110      # px; below this hands are only a few pixels wide


@dataclass
class Tracklet:
    """One person-appearance, sampled every `step` source frames."""

    id: int
    frames: np.ndarray   # (n,) source frame index of each sample
    box: np.ndarray      # (n, 4) x1, y1, x2, y2
    kp: np.ndarray       # (n, 17, 3) x, y, confidence (0 where interpolated)
    conf: np.ndarray     # (n,) detection confidence (0 where interpolated)
    step: int
    fps: float

    @property
    def seconds(self) -> float:
        return (int(self.frames[-1]) - int(self.frames[0]) + self.step) / self.fps

    @property
    def height(self) -> float:
        return float(np.median(self.box[:, 3] - self.box[:, 1]))

    def at(self, frame: int) -> int:
        """Sample index closest to a source frame."""
        return int(np.clip(round((frame - int(self.frames[0])) / self.step), 0, len(self.frames) - 1))


_model = None


def pose_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        MODELS.mkdir(parents=True, exist_ok=True)
        _model = YOLO(str(MODELS / POSE_WEIGHTS))
    return _model


def _tracker(cfg: PeopleConfig, sample_fps: float):
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace

    return BYTETracker(IterableSimpleNamespace(
        tracker_type="bytetrack", track_high_thresh=cfg.track_high, track_low_thresh=cfg.det_conf,
        new_track_thresh=cfg.track_new, track_buffer=int(round(cfg.track_buffer_s * sample_fps)),
        match_thresh=0.8, fuse_score=True))


def head_circle(box: np.ndarray, kp: np.ndarray) -> tuple[float, float, float]:
    """Centre and radius of a circle that covers the person's head, from keypoints with fallbacks."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    sh = kp[[L_SHOULDER, R_SHOULDER]]
    sh_ok = bool((sh[:, 2] > 0.35).all())
    sh_w = float(np.linalg.norm(sh[0, :2] - sh[1, :2])) if sh_ok else 0.0
    face = kp[FACE]
    seen = face[face[:, 2] > 0.35]
    if len(seen):
        c = seen[:, :2].mean(axis=0)
        spread = float(np.linalg.norm(seen[:, :2] - c, axis=1).max()) if len(seen) > 1 else 0.0
        r = max(1.5 * spread, 0.45 * sh_w, 0.065 * h)
    elif sh_ok:
        mid = sh[:, :2].mean(axis=0)
        c = np.array([mid[0], mid[1] - 0.6 * sh_w])
        r = max(0.5 * sh_w, 0.07 * h)
    else:
        c = np.array([(x1 + x2) / 2, y1 + 0.09 * h])
        r = max(0.09 * h, 0.2 * (x2 - x1))
    return float(c[0]), float(c[1]), float(min(r, 0.3 * max(h, x2 - x1)) * 1.15)


def track(src: Path, spans: list[tuple[int, int]], cfg: PeopleConfig | None = None, progress=None) -> dict:
    """Run pose + tracking over the frame spans. Returns raw per-sample rows and timing."""
    cfg = cfg or PeopleConfig()
    model, dev = pose_model(), device()
    rows_frame, rows_id, rows_box, rows_conf, rows_kp = [], [], [], [], []
    loose: dict[int, list[tuple[float, float, float]]] = {}  # head circles of detections the tracker did not keep
    total = sum(b - a for a, b in spans)
    done, n_inferred = 0, 0
    ids: dict[tuple[int, int], int] = {}   # (span number, tracker id) -> id unique across the whole video

    def consume(span: int, tracker, frames: list[int], images: list) -> None:
        """Run the pose model on a batch and feed its detections to the tracker, frame by frame."""
        results = model.predict(images, imgsz=cfg.imgsz, device=dev, conf=cfg.det_conf, classes=[0], verbose=False)
        for f, r in zip(frames, results):
            det = r.boxes.cpu().numpy()
            kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 17, 3))
            used = set()
            for x1, y1, x2, y2, tid, score, _cls, di in tracker.update(det, r.orig_img):
                di = int(di)
                used.add(di)
                rows_frame.append(f)
                rows_id.append(ids.setdefault((span, int(tid)), len(ids)))
                rows_box.append((x1, y1, x2, y2))
                rows_conf.append(score)
                rows_kp.append(kps[di])
            for di in range(len(det)):  # anything else that looks like a person still gets its face blurred
                if di not in used and det.conf[di] >= 0.2:
                    loose.setdefault(f, []).append(head_circle(det.xyxy[di], kps[di]))

    t0 = time.time()
    with media.Reader(src) as rd:
        step = max(1, int(round(rd.fps / cfg.sample_fps)))
        sample_fps = rd.fps / step
        for span, (a, b) in enumerate(spans):
            tracker = _tracker(cfg, sample_fps)   # a fresh tracker per span: nobody is carried across a gap
            frames, images = [], []
            for f, frame in rd.iter(range(a, b, step)):
                frames.append(f)
                images.append(frame)
                if len(images) == cfg.batch:
                    consume(span, tracker, frames, images)
                    n_inferred += len(images)
                    frames, images = [], []
                    if progress:
                        progress((done + f - a) / max(1, total))
            if images:
                consume(span, tracker, frames, images)
                n_inferred += len(images)
            done += b - a
        fps, size = rd.fps, (rd.width, rd.height)
    return {
        "frame": np.array(rows_frame, dtype=np.int64), "id": np.array(rows_id, dtype=np.int64),
        "box": np.array(rows_box, dtype=np.float32).reshape(-1, 4),
        "conf": np.array(rows_conf, dtype=np.float32),
        "kp": np.array(rows_kp, dtype=np.float32).reshape(-1, 17, 3),
        "loose": loose, "step": step, "fps": fps, "size": size,
        "samples": n_inferred, "seconds": time.time() - t0, "config": asdict(cfg),
    }


def build_tracklets(raw: dict, cfg: PeopleConfig | None = None) -> tuple[list[Tracklet], dict]:
    """Clean raw tracker rows into regular-grid tracklets. Returns (kept, counts of what was dropped)."""
    cfg = cfg or PeopleConfig()
    step, fps = raw["step"], raw["fps"]
    max_gap = int(round(cfg.max_gap_s * fps))
    pieces: list[Tracklet] = []
    for tid in np.unique(raw["id"]):
        sel = np.flatnonzero(raw["id"] == tid)
        sel = sel[np.argsort(raw["frame"][sel], kind="stable")]
        f, box, kp, conf = raw["frame"][sel], raw["box"][sel], raw["kp"][sel], raw["conf"][sel]
        # split at long gaps and at jumps (the tracker swapped to another person)
        centre = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], axis=1)
        height = box[:, 3] - box[:, 1]
        cuts = [0]
        for i in range(1, len(f)):
            gap = f[i] - f[i - 1]
            jump = np.linalg.norm(centre[i] - centre[i - 1]) > 0.6 * max(height[i], height[i - 1]) * max(1, gap / step)
            if gap > max_gap or jump:
                cuts.append(i)
        cuts.append(len(f))
        for a, b in zip(cuts[:-1], cuts[1:]):
            grid = np.arange(f[a], f[b - 1] + 1, step)
            pos = ((f[a:b] - f[a]) // step).astype(int)
            known = np.zeros(len(grid), dtype=bool)
            known[pos] = True
            gbox = np.zeros((len(grid), 4), np.float32)
            gkp = np.zeros((len(grid), 17, 3), np.float32)
            gconf = np.zeros(len(grid), np.float32)
            gbox[pos], gkp[pos], gconf[pos] = box[a:b], kp[a:b], conf[a:b]
            if not known.all():  # fill short gaps by linear interpolation; confidence stays 0 there
                xs = np.flatnonzero(known)
                miss = np.flatnonzero(~known)
                for c in range(4):
                    gbox[miss, c] = np.interp(miss, xs, gbox[xs, c])
                for k in range(17):
                    for c in range(2):
                        gkp[miss, k, c] = np.interp(miss, xs, gkp[xs, k, c])
            pieces.append(Tracklet(0, grid, gbox, gkp, gconf, step, fps))
    kept, dropped = [], {"short": 0, "small": 0}
    for t in sorted(pieces, key=lambda t: int(t.frames[0])):
        if t.seconds < cfg.min_seconds:
            dropped["short"] += 1
        elif t.height < cfg.min_height:
            dropped["small"] += 1
        else:
            t.id = len(kept)
            kept.append(t)
    return kept, dropped


def head_circles(raw: dict) -> dict[int, np.ndarray]:
    """Head circles (cx, cy, r) for every source frame, interpolated between samples.

    Uses every raw tracker row, including people later dropped as too small or too brief,
    plus loose detections, so that no visible face is left sharp in any output.
    """
    step = raw["step"]
    out: dict[int, list] = {}
    for tid in np.unique(raw["id"]):
        sel = np.flatnonzero(raw["id"] == tid)
        sel = sel[np.argsort(raw["frame"][sel], kind="stable")]
        f = raw["frame"][sel]
        circ = np.array([head_circle(raw["box"][i], raw["kp"][i]) for i in sel], dtype=np.float32)
        for i in range(len(f)):
            out.setdefault(int(f[i]), []).append(circ[i])
            if i + 1 < len(f) and f[i + 1] - f[i] <= 6 * step:
                for g in range(int(f[i]) + 1, int(f[i + 1])):
                    w = (g - f[i]) / (f[i + 1] - f[i])
                    c = circ[i] * (1 - w) + circ[i + 1] * w
                    c[2] = max(circ[i][2], circ[i + 1][2])
                    out.setdefault(g, []).append(c)
            else:  # last sample of a run: hold it until the next sample would have come
                for g in range(int(f[i]) + 1, int(f[i]) + step):
                    out.setdefault(g, []).append(circ[i])
    for f, circles in raw["loose"].items():
        for cx, cy, r in circles:
            for g in range(f - step + 1, f + step):
                out.setdefault(g, []).append(np.array([cx, cy, r * 1.25], dtype=np.float32))
    return {f: np.stack(v) for f, v in out.items()}


def save(path: Path, raw: dict) -> None:
    loose_f = np.array([f for f, v in raw["loose"].items() for _ in v], dtype=np.int64)
    loose_c = np.array([c for v in raw["loose"].values() for c in v], dtype=np.float32).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frame=raw["frame"], id=raw["id"], box=raw["box"], conf=raw["conf"], kp=raw["kp"],
                        loose_f=loose_f, loose_c=loose_c, step=raw["step"], fps=raw["fps"],
                        size=np.array(raw["size"]), samples=raw["samples"], seconds=raw["seconds"])


def load(path: Path) -> dict:
    z = np.load(path)
    loose: dict[int, list] = {}
    for f, c in zip(z["loose_f"], z["loose_c"]):
        loose.setdefault(int(f), []).append(tuple(float(v) for v in c))
    return {"frame": z["frame"], "id": z["id"], "box": z["box"], "conf": z["conf"], "kp": z["kp"], "loose": loose,
            "step": int(z["step"]), "fps": float(z["fps"]), "size": tuple(int(v) for v in z["size"]),
            "samples": int(z["samples"]), "seconds": float(z["seconds"])}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--save", type=Path, help="write the raw tracks to this .npz")
    a = ap.parse_args(argv)
    info = media.probe(a.video)
    first = int(a.start * info["fps"])
    last = first + int(a.seconds * info["fps"]) if a.seconds else int(info["seconds"] * info["fps"])
    raw = track(a.video, [(first, last)])
    tracklets, dropped = build_tracklets(raw)
    span_s = (last - first) / info["fps"]
    print(f"{a.video.name}: {raw['samples']} frames through the pose model in {raw['seconds']:.1f}s "
          f"({span_s / raw['seconds']:.1f}x real time)")
    print(f"  {len(tracklets)} tracklet(s) kept, dropped {dropped}")
    for t in tracklets:
        print(f"  #{t.id:<3} {t.frames[0] / t.fps:6.1f}s +{t.seconds:5.1f}s  height {t.height:4.0f}px  "
              f"wrists seen {float((t.kp[:, [L_WRIST, R_WRIST], 2] > 0.4).mean()):.0%}")
    if a.save:
        save(a.save, raw)
        print(f"  saved {a.save}")


if __name__ == "__main__":
    main()
