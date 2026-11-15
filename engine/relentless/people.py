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
    batch: int = field(default_factory=lambda: int(setting("POSE_BATCH", "8")))   # frames per model call; fewer needs less memory
    det_conf: float = 0.1      # ByteTrack wants low-confidence boxes too
    track_high: float = 0.3
    track_new: float = 0.4
    track_buffer_s: float = 3.0  # how long a lost person keeps their id
    max_gap_s: float = 1.0     # gaps up to this are interpolated; longer ones split the tracklet
    min_seconds: float = 1.5   # shorter appearances are dropped
    min_height: int = 110      # px; below this hands are only a few pixels wide
    # Linking the pieces of one person's track (the tracker loses someone behind a counter or another person
    # and gives them a new id when they are back): see link().
    link_near_s: float = 3.0   # pieces this close in time are linked on position (and clothes, where known)
    link_far_s: float = 20.0   # longer gaps only for someone back in the same spot, in the same clothes
    link_look: float = 0.5     # clothing difference (Bhattacharyya distance) above which pieces are not one person
    link_look_far: float = 0.35


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
    person: int = -1     # tracklets of one person, split where the tracker lost them, share this number

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


APP_BINS = (8, 4, 4)   # hue x saturation x value bins of the clothing colour histogram


def appearance(frame: np.ndarray, box, kp: np.ndarray) -> np.ndarray:
    """What a person's clothes look like: a normalised HSV histogram of the torso (shoulders to hips where the
    model sees them, else the middle of the upper half of the box). Lets two pieces of one person's track,
    split where the tracker lost them, be told apart from two people who stood in the same spot."""
    import cv2

    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    sh, hip = kp[[L_SHOULDER, R_SHOULDER]], kp[[L_HIP, R_HIP]]
    if (sh[:, 2] > 0.5).all() and (hip[:, 2] > 0.5).all():
        pts = np.concatenate([sh[:, :2], hip[:, :2]])
        ax1, ay1 = pts.min(axis=0)
        ax2, ay2 = pts.max(axis=0)
        pad = 0.15 * max(ax2 - ax1, 1.0)
        ax1, ax2 = ax1 + pad, ax2 - pad
    else:
        ax1, ax2, ay1, ay2 = x1 + 0.25 * w, x2 - 0.25 * w, y1 + 0.2 * h, y1 + 0.5 * h
    H, W = frame.shape[:2]
    ax1, ax2 = int(max(0, ax1)), int(min(W, ax2))
    ay1, ay2 = int(max(0, ay1)), int(min(H, ay2))
    if ax2 - ax1 < 4 or ay2 - ay1 < 4:
        return np.zeros(int(np.prod(APP_BINS)), np.float32)
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[ay1:ay2, ax1:ax2]), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(APP_BINS), [0, 180, 0, 256, 0, 256]).ravel()
    return (hist / max(float(hist.sum()), 1.0)).astype(np.float32)


def track(src: Path, spans: list[tuple[int, int]], cfg: PeopleConfig | None = None, progress=None, watch=None) -> dict:
    """Run pose + tracking over the frame spans. Returns raw per-sample rows and timing.

    `watch`, if given, is called for every sampled frame with (frame index, the frame, the people the tracker
    follows in it as (id, box, keypoints), everything else that looks like a person as (box, keypoints),
    how many frames have been through the model): it is how a job page follows the tracking."""
    cfg = cfg or PeopleConfig()
    model, dev = pose_model(), device()
    rows_frame, rows_id, rows_box, rows_conf, rows_kp, rows_app = [], [], [], [], [], []
    loose: dict[int, list[tuple[float, float, float]]] = {}  # head circles of detections the tracker did not keep
    total = sum(b - a for a, b in spans)
    done, n_inferred = 0, 0
    ids: dict[tuple[int, int], int] = {}   # (span number, tracker id) -> id unique across the whole video

    def consume(span: int, tracker, frames: list[int], images: list) -> None:
        """Run the pose model on a batch and feed its detections to the tracker, frame by frame."""
        results = model.predict(images, imgsz=cfg.imgsz, device=dev, conf=cfg.det_conf, classes=[0], verbose=False)
        for n, (f, r) in enumerate(zip(frames, results)):
            det = r.boxes.cpu().numpy()
            kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 17, 3))
            used, followed, others = set(), [], []
            for x1, y1, x2, y2, tid, score, _cls, di in tracker.update(det, r.orig_img):
                di = int(di)
                used.add(di)
                rows_frame.append(f)
                rows_id.append(ids.setdefault((span, int(tid)), len(ids)))
                rows_box.append((x1, y1, x2, y2))
                rows_conf.append(score)
                rows_kp.append(kps[di])
                rows_app.append(appearance(images[n], (x1, y1, x2, y2), kps[di]))
                followed.append((rows_id[-1], (x1, y1, x2, y2), kps[di]))
            for di in range(len(det)):  # anything else that looks like a person still gets its face blurred
                if di not in used and det.conf[di] >= 0.2:
                    loose.setdefault(f, []).append(head_circle(det.xyxy[di], kps[di]))
                    others.append((det.xyxy[di], kps[di]))
            if watch:
                watch(f, images[n], followed, others, n_inferred + n + 1)

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
        "app": np.array(rows_app, dtype=np.float32).reshape(-1, int(np.prod(APP_BINS))),
        "loose": loose, "step": step, "fps": fps, "size": size,
        "samples": n_inferred, "seconds": time.time() - t0, "config": asdict(cfg),
    }


def build_tracklets(raw: dict, cfg: PeopleConfig | None = None) -> tuple[list[Tracklet], dict]:
    """Clean raw tracker rows into regular-grid tracklets. Returns (kept, counts of what was dropped)."""
    cfg = cfg or PeopleConfig()
    step, fps = raw["step"], raw["fps"]
    max_gap = int(round(cfg.max_gap_s * fps))
    pieces: list[Tracklet] = []
    rows: list[np.ndarray] = []      # the raw rows of each piece
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
            rows.append(sel[a:b])
    # Linked before the short and small pieces are dropped, so that a chain can run through a brief one.
    for t, who in zip(pieces, link(raw, rows, cfg)):
        t.person = who
    kept, dropped = [], {"short": 0, "small": 0}
    for t in sorted(pieces, key=lambda t: int(t.frames[0])):
        if t.seconds < cfg.min_seconds:
            dropped["short"] += 1
        elif t.height < cfg.min_height:
            dropped["small"] += 1
        else:
            t.id = len(kept)
            kept.append(t)
    number: dict[int, int] = {}      # people numbered 0, 1, ... in the order they first appear
    for t in kept:
        t.person = number.setdefault(t.person, len(number))
    return kept, dropped


def _look(app: np.ndarray, rows: np.ndarray) -> np.ndarray:
    d = app[rows].mean(axis=0)
    return d / max(float(d.sum()), 1e-9)


def _clothes(a: np.ndarray, b: np.ndarray) -> float:
    """How different two clothing histograms are: 0 the same, 1 nothing in common (Bhattacharyya distance)."""
    return float(np.sqrt(max(0.0, 1.0 - float(np.sqrt(a * b).sum()))))


def link(raw: dict, pieces: list[np.ndarray], cfg: PeopleConfig | None = None) -> list[int]:
    """Which pieces of track are one person. `pieces` are the raw rows of each piece, in time order within
    each. Returns a person number per piece (the index of the chain's first piece).

    The tracker gives a person a new id when it loses them for more than a moment: behind a counter, behind
    someone walking past, bent over. A piece is continued by one that starts after it ends, close by, about
    as big, and (where the clothes were measured) dressed alike. Gaps of up to `link_near_s` are judged
    on that; longer ones, up to `link_far_s`, only for someone who is back in the same spot in clothes that
    match closely. A gap of more than a second is not bridged if anyone else stood in that spot meanwhile,
    and a piece with two similar candidates is left alone: an unlinked track is better than a wrong one.
    """
    cfg = cfg or PeopleConfig()
    if not pieces:
        return []
    fps, F, B = raw["fps"], raw["frame"], raw["box"]
    app = raw.get("app")
    if app is not None and len(app) != len(F):
        app = None
    n_end = 5
    start = [int(F[p[0]]) for p in pieces]
    end = [int(F[p[-1]]) for p in pieces]
    first = [np.median(B[p[:n_end]], axis=0) for p in pieces]
    last = [np.median(B[p[-n_end:]], axis=0) for p in pieces]
    look_in = [_look(app, p[:2 * n_end]) for p in pieces] if app is not None else None
    look_out = [_look(app, p[-2 * n_end:]) for p in pieces] if app is not None else None

    def top(b) -> np.ndarray:      # the head end of a box: stays put when a counter hides the legs
        return np.array([(b[0] + b[2]) / 2, b[1]])

    def spot_taken(i: int, j: int, h: float) -> bool:
        spot = (top(last[i]) + top(first[j])) / 2
        for k, p in enumerate(pieces):
            if k in (i, j) or end[k] <= end[i] or start[k] >= start[j]:
                continue
            during = p[(F[p] > end[i]) & (F[p] < start[j])]
            if len(during):
                b = B[during]
                heads = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 1]], axis=1)
                if (np.linalg.norm(heads - spot, axis=1) / h < 0.3).any():
                    return True
        return False

    options = []
    for i in range(len(pieces)):
        for j in range(len(pieces)):
            gap = (start[j] - end[i]) / fps
            if i == j or not 0 < gap <= cfg.link_far_s:
                continue
            a, b = last[i], first[j]
            ha, hb, wa, wb = a[3] - a[1], b[3] - b[1], a[2] - a[0], b[2] - b[0]
            if min(ha, hb, wa, wb) <= 0 or not (0.6 <= wb / wa <= 1 / 0.6 and 0.5 <= hb / ha <= 2.0):
                continue
            h = max(ha, hb)
            dist = float(np.linalg.norm(top(a) - top(b)) / h)
            look = _clothes(look_out[i], look_in[j]) if app is not None else None
            if gap <= cfg.link_near_s:
                if dist > 0.2 + 0.1 * gap or (look is not None and look > cfg.link_look):
                    continue
            elif look is None or dist > 0.15 or look > cfg.link_look_far:
                continue
            if gap > 1.0 and spot_taken(i, j, h):
                continue
            options.append((dist + 0.02 * gap + 0.5 * (look or 0.0), i, j))
    options.sort()
    costs_i: dict[int, list[float]] = {}
    costs_j: dict[int, list[float]] = {}
    for cost, i, j in options:
        costs_i.setdefault(i, []).append(cost)
        costs_j.setdefault(j, []).append(cost)

    def clear(costs: list[float], cost: float) -> bool:   # no second option nearly as good
        return len(costs) < 2 or costs[1] >= 1.3 * cost + 0.03

    after: dict[int, int] = {}
    before: dict[int, int] = {}
    for cost, i, j in options:
        if i in after or j in before or not (clear(costs_i[i], cost) and clear(costs_j[j], cost)):
            continue
        after[i], before[j] = j, i
    who = list(range(len(pieces)))
    for i in sorted(range(len(pieces)), key=lambda k: start[k]):
        if i in before:
            who[i] = who[before[i]]
    return who


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


def _track_child(src: Path, spans: list, out: Path, updates, live: dict | None = None) -> None:
    watch = None
    if live:
        from .live import TrackingView     # here, not at the top: live.py builds on this module

        watch = TrackingView(**live)
    save(out, track(src, spans, progress=updates.put, watch=watch))


def track_apart(src: Path, spans: list[tuple[int, int]], out: Path, progress=None, live: dict | None = None) -> dict:
    """track() in a child process that writes its result to `out` and exits. `live` is what the child
    needs to show a job page what it is tracking (see live.Kit.spec).

    The pose model and everything torch allocated for it are returned to the system before the
    next stage loads its own model. That halves the engine's peak memory on a small server.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    updates = ctx.Queue()
    child = ctx.Process(target=_track_child, args=(src, spans, out, updates, live), daemon=True)
    child.start()
    while child.is_alive():
        try:
            frac = updates.get(timeout=0.5)
        except Exception:   # nothing new yet
            continue
        if progress:
            progress(frac)
    child.join()
    if child.exitcode != 0:
        raise RuntimeError(f"person tracking stopped with exit code {child.exitcode}")
    return load(out)


def save(path: Path, raw: dict) -> None:
    loose_f = np.array([f for f, v in raw["loose"].items() for _ in v], dtype=np.int64)
    loose_c = np.array([c for v in raw["loose"].values() for c in v], dtype=np.float32).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"app": raw["app"]} if raw.get("app") is not None else {}
    np.savez_compressed(path, frame=raw["frame"], id=raw["id"], box=raw["box"], conf=raw["conf"], kp=raw["kp"], **extra,
                        loose_f=loose_f, loose_c=loose_c, step=raw["step"], fps=raw["fps"],
                        size=np.array(raw["size"]), samples=raw["samples"], seconds=raw["seconds"])


def load(path: Path) -> dict:
    z = np.load(path)
    loose: dict[int, list] = {}
    for f, c in zip(z["loose_f"], z["loose_c"]):
        loose.setdefault(int(f), []).append(tuple(float(v) for v in c))
    return {"frame": z["frame"], "id": z["id"], "box": z["box"], "conf": z["conf"], "kp": z["kp"], "loose": loose,
            "app": z["app"] if "app" in z.files else None,
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
