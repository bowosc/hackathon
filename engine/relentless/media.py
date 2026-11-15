"""Video I/O helpers. Uses the arm64 ffmpeg bundled with imageio-ffmpeg and OpenCV for decoding."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# x264 starts about one thread per core, each with its own buffers. THREADS caps that too.
X264 = ["-c:v", "libx264"] + (["-threads", os.environ["RELENTLESS_THREADS"]] if os.environ.get("RELENTLESS_THREADS") else [])


def _run(args: list[str]) -> str:
    proc = subprocess.run([FFMPEG, "-hide_banner", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(args)}\n{proc.stderr[-2000:]}")
    return proc.stderr


def probe(path: Path) -> dict:
    """Duration (s), width, height, fps. ffmpeg -i prints these to stderr."""
    proc = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    err = proc.stderr
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    vid = re.search(r"Video: .*?, (\d{2,5})x(\d{2,5})", err)
    fps = re.search(r"([\d.]+) fps", err)
    if not (dur and vid):
        raise RuntimeError(f"could not probe {path}")
    h, m, s = dur.groups()
    return {
        "seconds": int(h) * 3600 + int(m) * 60 + float(s),
        "width": int(vid.group(1)),
        "height": int(vid.group(2)),
        "fps": float(fps.group(1)) if fps else 30.0,
    }


def cut(src: Path, dst: Path, start: float = 0.0, seconds: float | None = None,
        crop: tuple[int, int, int, int] | None = None, max_height: int | None = None) -> Path:
    """Trim (and optionally crop x,y,w,h / downscale) to an H.264 mp4 without audio."""
    filters = []
    if crop:
        x, y, w, h = crop
        filters.append(f"crop={w}:{h}:{x}:{y}")
    if max_height:
        filters.append(f"scale=-2:'min({max_height},ih)'")
    args = ["-y", "-ss", f"{start:.3f}", "-i", str(src)]
    if seconds:
        args += ["-t", f"{seconds:.3f}"]
    if filters:
        args += ["-vf", ",".join(filters)]
    args += ["-an", *X264, "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(dst)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(args)
    return dst


def frames(src: Path, out_dir: Path, n: int = 6, width: int = 640) -> list[Path]:
    """n evenly spaced JPEG frames, resized to `width` px wide."""
    info = probe(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        t = info["seconds"] * (i + 0.5) / n
        p = out_dir / f"f{i:02d}.jpg"
        _run(["-y", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(p)])
        paths.append(p)
    return paths


def poster(src: Path, dst: Path, at: float = 0.5, width: int = 960) -> Path:
    """One JPEG still from `at` (0-1) of the way through a clip, `width` px wide."""
    t = probe(src)["seconds"] * at
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["-y", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(dst)])
    return dst


def web_copy(src: Path, dst: Path, crf: int = 26, max_height: int = 1080, preset: str = "slow") -> Path:
    """A smaller H.264 copy for streaming from a website: no audio, starts playing before it has loaded."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["-y", "-i", str(src), "-an", "-vf", f"scale=-2:'min({max_height},ih)'", *X264,
          "-preset", preset, "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart", str(dst)])
    return dst


def side_by_side(paths: list[Path], dst: Path, height: int = 360) -> Path:
    """Stack clips horizontally (same duration assumed) for demo/comparison."""
    args = ["-y"]
    for p in paths:
        args += ["-i", str(p)]
    scaled = "".join(f"[{i}:v]scale=-2:{height},setsar=1,fps=24[v{i}];" for i in range(len(paths)))
    stack = "".join(f"[v{i}]" for i in range(len(paths)))
    args += ["-filter_complex", f"{scaled}{stack}hstack=inputs={len(paths)}:shortest=1[out]",
             "-map", "[out]", "-an", *X264, "-pix_fmt", "yuv420p", "-crf", "20", str(dst)]
    _run(args)
    return dst


def dump(obj, path: Path) -> None:
    """Write JSON atomically, so a reader polling the file never sees a half-written one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


# ---------- frame-accurate reading and writing ----------

class Reader:
    """Sequential decoder that hands out frames by index.

    Small gaps are skipped with grab() (decode only); larger ones use a seek, which is
    frame-accurate on the H.264 AVI/MP4 files we handle (checked against sequential reads).
    """

    SEEK_GAP = 120  # frames; below this, grabbing is faster than a keyframe seek

    def __init__(self, path: Path):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = 0  # index of the frame the next read() returns

    def read(self, index: int) -> np.ndarray | None:
        """Frame `index`, or None past the end. Any order works; going backwards or far ahead costs a seek."""
        if index < self.pos or index - self.pos > self.SEEK_GAP:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.pos = index
        while self.pos < index:
            if not self.cap.grab():
                return None
            self.pos += 1
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.pos += 1
        return frame

    def iter(self, indices: Iterable[int]) -> Iterator[tuple[int, np.ndarray]]:
        for i in indices:
            frame = self.read(i)
            if frame is None:
                return
            yield i, frame

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "Reader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Writer:
    """Encode BGR frames to a browser-playable H.264 mp4 through an ffmpeg pipe."""

    def __init__(self, dst: Path, width: int, height: int, fps: float, crf: int = 18, preset: str = "veryfast",
                 gop: int | None = None):
        """`gop` is the distance between keyframes; a short one lets a browser seek to any frame quickly."""
        if width % 2 or height % 2:
            raise ValueError(f"H.264 needs even dimensions, got {width}x{height}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.dst, self.size, self.n = dst, (width, height), 0
        self._err = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-", "-an", *X264, "-preset", preset,
             *(["-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"] if gop else []),
             "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart", str(dst)],
            stdin=subprocess.PIPE, stderr=self._err)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError(f"frame is {frame.shape[1]}x{frame.shape[0]}, writer expects {self.size[0]}x{self.size[1]}")
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            self.close()
        self.n += 1

    def close(self) -> Path:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        code = self._proc.wait()
        self._err.seek(0)
        err = self._err.read().decode(errors="replace")
        self._err.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg encode failed for {self.dst}: {err[-1500:]}")
        return self.dst

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, exc_type, *exc) -> None:
        if exc_type is None:
            self.close()
        else:  # don't mask the original error with an encoder error
            self._proc.kill()
            self._err.close()


def even(v: float) -> int:
    """Round down to an even integer (H.264 with yuv420p needs even sizes)."""
    return int(v) // 2 * 2


def fit_box(cx: float, cy: float, w: float, h: float, frame_w: int, frame_h: int,
            aspect: float | None = None) -> tuple[int, int, int, int]:
    """Box of about w x h centred on (cx, cy), optionally grown to `aspect` (w/h), clamped inside the frame.

    Returns x, y, w, h with even sizes.
    """
    if aspect:
        if w / h < aspect:
            w = h * aspect
        else:
            h = w / aspect
        if w > frame_w:
            w, h = frame_w, frame_w / aspect
        if h > frame_h:
            h, w = frame_h, frame_h * aspect
    w, h = min(even(w), even(frame_w)), min(even(h), even(frame_h))
    x = int(min(max(cx - w / 2, 0), frame_w - w))
    y = int(min(max(cy - h / 2, 0), frame_h - h))
    return x, y, w, h
