"""Previews for a job page: what each stage is looking at, written while the engine runs.

With --live the engine also fills <run>/live/:

  now.json      the frame a running stage is on, a few times a second (motion filter, person tracking):
                the picture itself and, for tracking, the people found in it
  kit.json      which of the files below exist, and how to read them
  scan.mp4      the whole stretch, small and at a few frames a second
  preview.mp4   the busiest seconds of it at full frame rate
  tracks.json   the people in preview.mp4: box, keypoints and whether their hands are busy, per sample
  cands.jpg     one still per hand moment, tiled
  screen.jpg    the stills the quick screen showed Claude, one row per candidate

The engine never reads these back. Faces are blurred in all of them by the rule the engine's outputs
use (people.head_circles): every head the pose model found, and where a person was between two
sightings a moment apart. For the frame person tracking is on, that takes knowing the samples after
it, so the frame shown is the one from a few samples ago. The one picture made before any face has
been located (the motion filter's) is coarser than a face: 64 pixels across the whole frame, with
the changed pixels drawn over it.

  uv run python -m relentless.live data/runs/NAME     # write them for a run made without --live
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import media, render
from .people import head_circle, head_circles

BINS = 240                 # how finely the live charts divide the stretch
MOTION_BLUE = (229, 135, 57)   # BGR
TILE = (160, 90)
SPRITE_COLS = 16
MAX_THUMBS = 480           # hand moments that get a still (the most active ones, when there are more)
MAX_SCREEN_ROWS = 120
REACH = 6                  # samples: people.head_circles joins two sightings of a person up to this far apart


def write_json(obj, path: Path) -> None:
    """Compact and atomic: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
    os.replace(tmp, path)


def jpeg(img: np.ndarray, quality: int = 76) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def data_url(img: np.ndarray, quality: int = 76) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg(img, quality)).decode()


def motion_picture(small: np.ndarray, changed: np.ndarray | None) -> np.ndarray:
    """What the motion filter works with: a dim, very coarse view of the frame with the changed pixels lit.

    `small` is the filter's own grey frame. No face has been found yet at this stage, so the frame is
    reduced to 64 pixels across before it is shown: enough to tell a room, far too little for a face."""
    h, w = small.shape[:2]
    coarse = cv2.resize(small, (64, max(2, round(64 * h / w))), interpolation=cv2.INTER_AREA)
    dim = (cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC) * 0.42).astype(np.uint8)
    img = cv2.cvtColor(dim, cv2.COLOR_GRAY2BGR)
    if changed is not None:
        lit = cv2.dilate(changed, np.ones((3, 3), np.uint8)) > 0
        img[lit] = (0.2 * img[lit] + 0.8 * np.array(MOTION_BLUE)).astype(np.uint8)
    return img


class MotionView:
    """Follows the motion scan: a picture now and then, and the motion found so far along the stretch."""

    def __init__(self, folder: Path, first: int, last: int, fps: float, every: float = 0.4):
        self.folder, self.first, self.last, self.fps, self.every = Path(folder), first, last, fps, every
        self.timeline: list[float | None] = [None] * BINS
        self._at, self._seq = 0.0, 0

    def __call__(self, f: int, small: np.ndarray, changed: np.ndarray | None, score: float) -> None:
        b = min(BINS - 1, int((f - self.first) * BINS / max(1, self.last - self.first)))
        self.timeline[b] = max(self.timeline[b] or 0.0, round(float(score), 5))
        now = time.time()
        if now - self._at < self.every:
            return
        self._at, self._seq = now, self._seq + 1
        write_json({"stage": "motion", "seq": self._seq, "t": round((f - self.first) / self.fps, 2),
                    "size": [int(small.shape[1]), int(small.shape[0])], "timeline": self.timeline,
                    "image": data_url(motion_picture(small, changed), 70)}, self.folder / "now.json")


class TrackingView:
    """Follows person tracking: a frame the pose model has just been through, with the people found in it.

    Heads are blurred by the outputs' own rule, which also covers a person the model missed for a few samples
    (from where they were before and after). That needs the samples after a frame as well as those before it,
    so the frame shown is the one from REACH samples ago, and the last REACH of a stretch are never shown."""

    def __init__(self, folder: Path, first: int, last: int, fps: float, every: float = 0.6, width: int = 768):
        self.folder, self.first, self.last, self.fps = Path(folder), first, last, fps
        self.every, self.width = every, width
        self.counts: list[int | None] = [None] * BINS
        self.ids: set[int] = set()
        self._at, self._seq = 0.0, 0
        # the last samples: (frame index, small picture, people followed, other detections, source size)
        self._window: deque = deque(maxlen=2 * REACH + 1)

    def __call__(self, f: int, frame: np.ndarray, tracked: list, others: list, samples: int) -> None:
        """`tracked` is [(id, box, keypoints)] for the people the tracker follows in this frame, `others` is
        [(box, keypoints)] for everything else that looks like a person (their faces are blurred too)."""
        b = min(BINS - 1, int((f - self.first) * BINS / max(1, self.last - self.first)))
        self.counts[b] = max(self.counts[b] or 0, len(tracked))
        self.ids.update(pid for pid, _box, _kp in tracked)
        h, w = frame.shape[:2]
        out = (media.even(self.width), media.even(self.width * h / w))
        self._window.append((int(f), cv2.resize(frame, out, interpolation=cv2.INTER_AREA),
                             [(int(pid), np.asarray(box, np.float32), np.asarray(kp, np.float32)) for pid, box, kp in tracked],
                             [(np.asarray(box, np.float32), np.asarray(kp, np.float32)) for box, kp in others], (w, h)))
        if len(self._window) <= REACH:      # the frame REACH samples back does not exist yet
            return
        now = time.time()
        if now - self._at < self.every:
            return
        self._at, self._seq = now, self._seq + 1
        index = len(self._window) - 1 - REACH
        g, small, people, _others, (w, h) = self._window[index]
        shown, scale = small.copy(), small.shape[1] / w
        circles = self.circles(index) * scale
        circles[:, 2] += 1.0      # the picture is small: a pixel more, so that rounding never uncovers a rim
        render.blur_heads(shown, circles if len(circles) else None)
        write_json({"stage": "people", "seq": self._seq, "t": round((g - self.first) / self.fps, 2), "size": [w, h],
                    "people": [person_row(pid, box, kp) for pid, box, kp in people],
                    "seen": len(self.ids), "samples": int(samples), "counts": self.counts,
                    "image": data_url(shown, 72)}, self.folder / "now.json")

    def circles(self, index: int) -> np.ndarray:
        """Head circles (cx, cy, r) in source pixels for the window's sample `index`, by the outputs' rule.

        The window reaches REACH samples to either side of the frame that is shown, which is as far as that rule
        looks, so these are the circles the outputs will blur in that frame."""
        frames = [entry[0] for entry in self._window]
        rows = [(f, pid, box, kp) for f, _small, people, _others, _size in self._window for pid, box, kp in people]
        raw = {"frame": np.array([row[0] for row in rows], np.int64), "id": np.array([row[1] for row in rows], np.int64),
               "box": np.array([row[2] for row in rows], np.float32).reshape(-1, 4),
               "kp": np.array([row[3] for row in rows], np.float32).reshape(-1, 17, 3),
               "step": max(1, min((b - a for a, b in zip(frames, frames[1:])), default=1)),
               "loose": {f: [head_circle(box, kp) for box, kp in others] for f, _small, _people, others, _size in self._window if others}}
        return head_circles(raw).get(frames[index], np.zeros((0, 3), np.float32))


def person_row(pid: int, box, kp, **more) -> dict:
    return {"id": int(pid), "box": [int(round(float(v))) for v in box],
            "kp": [[int(round(float(x))), int(round(float(y))), round(float(c), 2)] for x, y, c in kp], **more}


# ---------- jobs for the cut pass: the source is decoded once, these take what they need from it ----------

@dataclass
class Small(render.Job):
    """The frames [start, end), every `every`-th one, shrunk to `width`: a small video for a web page."""

    path: Path
    start: int
    end: int
    every: int
    width: int
    fps: float            # of the source; the video plays at fps / every, so its clock is the footage's
    crf: int = 25
    gop: int = 10
    size: tuple[int, int] | None = None
    _w: media.Writer | None = field(default=None, repr=False)

    def wants(self, f: int) -> bool:
        return self.start <= f < self.end and (f - self.start) % self.every == 0

    def emit(self, f: int, frame: np.ndarray) -> None:
        if self._w is None:
            h, w = frame.shape[:2]
            self.size = (media.even(self.width), media.even(self.width * h / w))
            self._w = media.Writer(self.path, *self.size, self.fps / self.every, crf=self.crf, gop=self.gop)
        self._w.write(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA))

    def close(self) -> None:
        if self._w:
            self._w.close()


@dataclass
class Thumbs(render.Job):
    """One still per hand moment, taken from the middle of it."""

    moments: list[tuple[int, int, tuple[int, int, int, int]]]     # (candidate number, frame, crop x y w h)
    tiles: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._at: dict[int, list] = {}
        for k, f, crop in self.moments:
            self._at.setdefault(f, []).append((k, crop))
        self.start = min(self._at, default=0)
        self.end = max(self._at, default=-1) + 1

    def wants(self, f: int) -> bool:
        return f in self._at

    def emit(self, f: int, frame: np.ndarray) -> None:
        for k, (x, y, w, h) in self._at[f]:
            self.tiles[k] = cv2.resize(frame[y:y + h, x:x + w], TILE, interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        pass


def sprite(rows: list[np.ndarray], cols: int) -> np.ndarray:
    """Same-sized pictures tiled left to right, top to bottom."""
    h, w = rows[0].shape[:2]
    sheet = np.zeros((h * -(-len(rows) // cols), w * cols, 3), np.uint8)
    for i, img in enumerate(rows):
        y, x = (i // cols) * h, (i % cols) * w
        sheet[y:y + h, x:x + w] = img
    return sheet


def busiest(first: int, last: int, fps: float, moments: list[tuple[int, int]], people: list[tuple[int, int]],
            seconds: float = 14.0) -> tuple[int, int]:
    """The `seconds`-long stretch [a, b) with the most hand moments in it (people in view break a tie)."""
    n = last - first
    length = min(n, int(round(seconds * fps)))
    if length >= n:
        return first, last
    cover = np.zeros(n + 1, np.float64)
    for spans, weight in ((moments, 1.0), (people, 0.05)):
        for a, b in spans:
            a, b = max(a, first) - first, min(b, last) - first
            if b > a:
                cover[a] += weight
                cover[b] -= weight
    total = np.concatenate([[0.0], np.cumsum(np.cumsum(cover)[:n])])      # total[i] = cover summed over frames < i
    step = max(1, int(round(fps / 2)))
    starts = np.arange(0, n - length + 1, step)
    best = int(starts[int(np.argmax(total[starts + length] - total[starts]))])
    return first + best, first + best + length


class Kit:
    """Writes the previews of one run."""

    def __init__(self, folder: Path, first: int, last: int, fps: float, size: tuple[int, int]):
        self.folder, self.first, self.last, self.fps, self.size = Path(folder), first, last, fps, tuple(size)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.parts: dict = {}
        self._cut: dict | None = None

    # -- while the first stages run
    def spec(self) -> dict:
        """What another process needs to follow tracking for this run."""
        return {"folder": str(self.folder), "first": self.first, "last": self.last, "fps": self.fps}

    def motion(self) -> MotionView:
        return MotionView(self.folder, self.first, self.last, self.fps)

    def tracking(self) -> TrackingView:
        return TrackingView(**self.spec())

    def quiet(self) -> None:
        """No stage is showing a frame any more."""
        (self.folder / "now.json").unlink(missing_ok=True)

    def note(self, **parts) -> None:
        self.parts.update(parts)
        write_json(self.parts, self.folder / "kit.json")

    def rel(self, name: str) -> str:
        return f"{self.folder.name}/{name}"

    # -- the quick screen
    def screen_sheet(self, stills: dict[int, list[np.ndarray]]) -> None:
        """The frames Claude is shown for each screened candidate, one row per candidate."""
        index = sorted(stills)[:MAX_SCREEN_ROWS]
        if not index:
            return
        n = len(stills[index[0]])
        rows = [np.hstack([cv2.resize(f, TILE, interpolation=cv2.INTER_AREA) for f in stills[k]]) for k in index]
        cv2.imwrite(str(self.folder / "screen.jpg"), sprite(rows, 1), [cv2.IMWRITE_JPEG_QUALITY, 80])
        self.note(screen={"file": self.rel("screen.jpg"), "tile": list(TILE), "frames": n, "index": index})

    # -- the cut pass
    def cut_jobs(self, tracklets: list, active: dict[int, list[tuple[int, int]]], moments: list[dict]) -> list[render.Job]:
        """Jobs to add to the pass that cuts the clips. `moments` are the hand moments, as dicts with
        k (their number), track, start, end (frames), crop, hands, score and clip (the clip made of it, if any)."""
        n = self.last - self.first
        # at most five pictures a second, and at most 1,800 in all: a long stretch gets a slower video, not a bigger one
        every = max(1, int(round(self.fps / 5)), -(-n // 1800))
        p0, p1 = busiest(self.first, self.last, self.fps, [(m["start"], m["end"]) for m in moments],
                         [(int(t.frames[0]), int(t.frames[-1]) + t.step) for t in tracklets])
        skip = 2 if self.fps > 40 else 1
        shown = sorted(moments, key=lambda m: -m.get("score", 0.0))[:MAX_THUMBS]
        thumbs = Thumbs([(m["k"], m["start"] + (m["end"] - m["start"]) // 2, tuple(m["crop"])) for m in shown])
        # a keyframe every few seconds keeps these small; a browser that plays them has no need for more
        scan = Small(self.folder / "scan.mp4", self.first, self.last, every, 640, self.fps, crf=30, gop=25)
        preview = Small(self.folder / "preview.mp4", p0, p1, skip, 960, self.fps, crf=24, gop=30)
        self._cut = dict(tracklets=tracklets, active=active, moments=moments, thumbs=thumbs, scan=scan,
                         preview=preview, stretch=(p0, p1), every=every, skip=skip)
        return [scan, preview] + ([thumbs] if shown else [])

    def tracks(self, tracklets: list, active: dict[int, list[tuple[int, int]]], moments: list[dict],
               p0: int, p1: int) -> dict:
        """The people of the stretch [p0, p1): one entry per sampled frame, as the page draws them."""
        frames: dict[int, list] = {}
        for t in tracklets:
            spans = active.get(t.id, [])
            for i, fr in enumerate(t.frames):
                fr = int(fr)
                if not (p0 - 3 * t.step <= fr < p1 + 3 * t.step):
                    continue
                frames.setdefault(fr, []).append(person_row(
                    t.id, t.box[i], t.kp[i], busy=any(a <= fr < b for a, b in spans)))
        t0, t1 = (p0 - self.first) / self.fps, (p1 - self.first) / self.fps
        return {"start_s": round(t0, 3), "seconds": round(t1 - t0, 3), "fps": self.fps, "size": list(self.size),
                "samples": [{"t": round((fr - self.first) / self.fps, 4), "people": frames[fr]} for fr in sorted(frames)],
                "candidates": [{"k": m["k"], "track": m["track"], "start_s": round((m["start"] - self.first) / self.fps, 2),
                                "seconds": round((m["end"] - m["start"]) / self.fps, 2), "crop": list(m["crop"]),
                                "hands": m.get("hands"), "clip": m.get("clip")}
                               for m in moments if m["start"] < p1 and m["end"] > p0]}

    def after_cut(self) -> None:
        """Call when the pass is over: writes the still sheet and the tracks, and says what is there."""
        c = self._cut
        if not c:
            return
        p0, p1 = c["stretch"]
        write_json(self.tracks(c["tracklets"], c["active"], c["moments"], p0, p1), self.folder / "tracks.json")
        parts = {}
        if c["scan"].size:
            parts["scan"] = {"file": self.rel("scan.mp4"), "fps": round(self.fps / c["every"], 3),
                             "seconds": round((self.last - self.first) / self.fps, 2), "size": list(c["scan"].size)}
        if c["preview"].size:
            parts["preview"] = {"file": self.rel("preview.mp4"), "tracks": self.rel("tracks.json"),
                                "start_s": round((p0 - self.first) / self.fps, 3),
                                "seconds": round((p1 - p0) / self.fps, 3), "fps": round(self.fps / c["skip"], 3),
                                "size": list(c["preview"].size)}
        tiles = c["thumbs"].tiles
        if tiles:
            index = sorted(tiles)
            cv2.imwrite(str(self.folder / "cands.jpg"), sprite([tiles[k] for k in index], SPRITE_COLS),
                        [cv2.IMWRITE_JPEG_QUALITY, 78])
            parts["cands"] = {"file": self.rel("cands.jpg"), "tile": list(TILE), "cols": SPRITE_COLS, "index": index}
        self.note(**parts)
        self._cut = None


# ---------- for a run that was made without --live ----------

def rebuild(run_dir: Path) -> dict:
    """Write the previews of a finished run from its state and the tracking cache. One pass over the source."""
    from . import people
    from .config import POSE_WEIGHTS
    from .pipeline import CACHE, file_sig, signature

    state = json.loads((run_dir / "state.json").read_text())
    src, fps, size = Path(state["source"]["path"]), state["source"]["fps"], tuple(state["source"]["size"])
    first, last = state["source"]["frames"]
    spans = [tuple(s) for s in state["motion"]["frame_spans"]]
    cache = CACHE / f"tracks-{signature(file_sig(src), spans, asdict(people.PeopleConfig()), POSE_WEIGHTS)}.npz"
    if not cache.exists():
        raise SystemExit(f"the tracking cache for this run ({cache.name}) is gone; run the engine again with --live")
    raw = people.load(cache)
    tracklets, _ = people.build_tracklets(raw)
    heads = people.head_circles(raw)
    frame = lambda s: first + int(round(s * fps))    # noqa: E731
    active = {p["id"]: [(frame(a), frame(b)) for a, b in p["active"]] for p in state["people"]}
    moments = [{"k": k, "track": c["track"], "start": frame(c["start_s"]), "end": frame(c["start_s"] + c["seconds"]),
                "crop": c["crop"], "hands": c["hands"], "score": c.get("score", 0.0), "clip": c.get("clip")}
               for k, c in enumerate(state["candidates"])]
    kit = Kit(run_dir / "live", first, last, fps, size)
    screened = {m["k"]: render.Stills(m["start"], m["end"], tuple(m["crop"])) for m in moments
                if state["candidates"][m["k"]].get("rating") is not None}
    render.run(src, kit.cut_jobs(tracklets, active, moments) + list(screened.values()), heads)
    kit.screen_sheet({k: j.frames for k, j in screened.items() if len(j.frames) == j.n})
    kit.after_cut()
    return kit.parts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="a finished run folder, e.g. data/runs/NAME")
    a = ap.parse_args(argv)
    parts = rebuild(a.run)
    print(f"{a.run.name}: wrote {', '.join(sorted(parts))} to {a.run / 'live'}")


if __name__ == "__main__":
    main()
