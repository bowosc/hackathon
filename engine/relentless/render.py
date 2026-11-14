"""Cut every output video in one pass over the source.

The source is decoded once. Each needed frame first gets its faces blurred (so nothing
downstream, including the upscaler and the labeler, ever sees a face), then every job that
wants the frame takes its crop. Jobs:

  FixedCrop   a hand-zoom clip, or its wider context clip (optionally with the zoom box drawn)
  Stills      a few evenly spaced frames of a planned clip, kept in memory for the screening step
  FollowCrop  a preview that follows one person through an active span
  Overview    the whole frame, small, with boxes and skeletons, to show what the engine saw
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import media
from .people import SKELETON, Tracklet

ACTIVE, IDLE, ZOOM = (60, 200, 60), (150, 150, 150), (0, 190, 255)  # BGR


def blur_heads(frame: np.ndarray, circles: np.ndarray | None) -> None:
    """Replace each head circle with a heavy, irreversible blur (in place)."""
    if circles is None:
        return
    fh, fw = frame.shape[:2]
    for cx, cy, r in circles:
        x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
        x2, y2 = min(fw, int(cx + r) + 1), min(fh, int(cy + r) + 1)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = frame[y1:y2, x1:x2]
        tiny = cv2.resize(roi, (3, 3), interpolation=cv2.INTER_AREA)  # 9 values per head: nothing to recover
        soft = cv2.resize(tiny, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
        mask = np.zeros(roi.shape[:2], np.uint8)
        cv2.ellipse(mask, (int(cx) - x1, int(cy) - y1), (int(r), int(r * 1.1)), 0, 0, 360, 255, -1)
        roi[mask > 0] = soft[mask > 0]


class Job:
    start: int
    end: int

    def wants(self, f: int) -> bool:
        return self.start <= f < self.end

    def emit(self, f: int, frame: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


@dataclass
class FixedCrop(Job):
    path: Path
    start: int
    end: int
    crop: tuple[int, int, int, int]                      # x, y, w, h in source pixels
    fps: float
    outline: tuple[int, int, int, int] | None = None     # box to draw, in source pixels
    _w: media.Writer | None = field(default=None, repr=False)

    def emit(self, f: int, frame: np.ndarray) -> None:
        x, y, w, h = self.crop
        if self._w is None:
            self._w = media.Writer(self.path, w, h, self.fps)
        out = frame[y:y + h, x:x + w]
        if self.outline:
            out = out.copy()
            ox, oy, ow, oh = self.outline
            cv2.rectangle(out, (ox - x, oy - y), (ox - x + ow, oy - y + oh), ZOOM, max(2, h // 160))
        self._w.write(out)

    def close(self) -> None:
        if self._w:
            self._w.close()


@dataclass
class Stills(Job):
    """`n` evenly spaced frames of a crop, kept in memory (no file)."""

    start: int
    end: int
    crop: tuple[int, int, int, int]
    n: int = 4
    frames: list = field(default_factory=list)

    def __post_init__(self) -> None:
        span = self.end - self.start
        self._take = {self.start + int((i + 0.5) * span / self.n) for i in range(self.n)}

    def wants(self, f: int) -> bool:
        return f in self._take

    def emit(self, f: int, frame: np.ndarray) -> None:
        x, y, w, h = self.crop
        self.frames.append(frame[y:y + h, x:x + w].copy())

    def close(self) -> None:
        pass


@dataclass
class FollowCrop(Job):
    """Preview that keeps one person centred. The centre is a smoothed path, so the crop glides instead of jittering."""

    path: Path
    tracklet: Tracklet
    start: int
    end: int
    frame_size: tuple[int, int]
    every: int = 2          # write every n-th source frame
    max_h: int = 360
    _w: media.Writer | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        t = self.tracklet
        a, b = t.at(self.start), t.at(self.end - 1) + 1
        box = t.box[a:b]
        bw, bh = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
        h = float(np.percentile(bh, 95)) * 1.2
        w = max(float(np.percentile(bw, 95)) * 1.35, h * 0.75)
        k = max(1, int(round(1.0 * t.fps / t.step))) // 2 * 2 + 1
        centre = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], axis=1)
        pad = k // 2
        cp = np.concatenate([np.repeat(centre[:1], pad, 0), centre, np.repeat(centre[-1:], pad, 0)])
        self._centre = np.stack([np.convolve(cp[:, c], np.ones(k) / k, mode="valid") for c in (0, 1)], axis=1)
        self._frames = t.frames[a:b]
        _, _, self._cw, self._ch = media.fit_box(0, 0, w, h, *self.frame_size)
        s = min(1.0, self.max_h / self._ch)
        self._out = (media.even(self._cw * s), media.even(self._ch * s))

    def wants(self, f: int) -> bool:
        return self.start <= f < self.end and (f - self.start) % self.every == 0

    def emit(self, f: int, frame: np.ndarray) -> None:
        cx = float(np.interp(f, self._frames, self._centre[:, 0]))
        cy = float(np.interp(f, self._frames, self._centre[:, 1]))
        x, y, w, h = media.fit_box(cx, cy, self._cw, self._ch, *self.frame_size)
        if self._w is None:
            self._w = media.Writer(self.path, *self._out, self.tracklet.fps / self.every, crf=24)
        self._w.write(cv2.resize(frame[y:y + h, x:x + w], self._out, interpolation=cv2.INTER_AREA))

    def close(self) -> None:
        if self._w:
            self._w.close()


@dataclass
class Overview(Job):
    """Small full-frame video with what the engine saw: people (green = hands busy), skeletons, zoom boxes."""

    path: Path
    start: int
    end: int
    frame_size: tuple[int, int]
    fps: float
    step: int
    tracklets: list[Tracklet]
    active: dict[int, list[tuple[int, int]]]          # tracklet id -> active spans
    zooms: list[tuple[int, int, tuple[int, int, int, int]]]  # (start, end, crop) of every planned clip
    width: int = 960
    _w: media.Writer | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._s = self.width / self.frame_size[0]
        self._out = (media.even(self.width), media.even(self.frame_size[1] * self._s))

    def wants(self, f: int) -> bool:
        return self.start <= f < self.end and (f - self.start) % self.step == 0

    def emit(self, f: int, frame: np.ndarray) -> None:
        if self._w is None:
            self._w = media.Writer(self.path, *self._out, self.fps / self.step, crf=26)
        img = cv2.resize(frame, self._out, interpolation=cv2.INTER_AREA)
        s = self._s
        for t in self.tracklets:
            if not (t.frames[0] <= f <= t.frames[-1]):
                continue
            i = t.at(f)
            busy = any(a <= f < b for a, b in self.active.get(t.id, []))
            col = ACTIVE if busy else IDLE
            x1, y1, x2, y2 = (t.box[i] * s).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if busy else 1)
            cv2.putText(img, f"#{t.id}", (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            if busy:
                kp = t.kp[i]
                for a, b in SKELETON:
                    if kp[a, 2] > 0.4 and kp[b, 2] > 0.4:
                        cv2.line(img, tuple((kp[a, :2] * s).astype(int)), tuple((kp[b, :2] * s).astype(int)), col, 1,
                                 cv2.LINE_AA)
        for a, b, (x, y, w, h) in self.zooms:
            if a <= f < b:
                cv2.rectangle(img, (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s)), ZOOM, 2)
        self._w.write(img)

    def close(self) -> None:
        if self._w:
            self._w.close()


def run(src: Path, jobs: list[Job], heads: dict[int, np.ndarray], progress=None) -> int:
    """Decode the source once and feed every job. Returns the number of frames decoded."""
    if not jobs:
        return 0
    lo, hi = min(j.start for j in jobs), max(j.end for j in jobs)
    pending = sorted(jobs, key=lambda j: j.start)
    live: list[Job] = []
    n = 0
    try:
        with media.Reader(src) as rd:
            for f in range(lo, hi):
                while pending and pending[0].start <= f:
                    live.append(pending.pop(0))
                takers = [j for j in live if j.wants(f)]
                if takers:
                    frame = rd.read(f)
                    if frame is None:
                        break
                    blur_heads(frame, heads.get(f))
                    for j in takers:
                        j.emit(f, frame)
                    n += 1
                done = [j for j in live if j.end <= f + 1]
                for j in done:
                    j.close()
                    live.remove(j)
                if progress and (f - lo) % 150 == 0:
                    progress((f - lo) / max(1, hi - lo))
    finally:
        for j in live:
            j.close()
    return n
