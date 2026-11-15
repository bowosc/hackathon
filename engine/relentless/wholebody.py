"""Whole-body pose: a second, sharper look at one tracked person, hands included.

The pose model of stage 2 looks at the whole frame, where a person is one or two hundred pixels tall
and a wrist is a guess between a sleeve and a coat. It is good enough to follow people and to notice
that their hands are busy. It is not good enough to say where a hand is.

This model (RTMW, from OpenMMLab) looks at one person at a time, cut out of the frame and enlarged,
and returns 133 keypoints: the same 17 body joints, plus feet, face and 21 keypoints for each hand.
The engine uses it where the answer is shown or sold: on the clips it cuts, and on the stretch of
footage the job page replays.

Scores are the model's own (the height of its response, roughly 0 to 10; the smaller model's are
brought onto the larger one's scale when they come out, see config.WHOLEBODY_MODELS). `confidence` turns them
into the 0-1 scale the rest of the engine uses, with 0.5 at the score below which a keypoint was
more often wrong than right on the footage this was measured on: 4,266 hand and 4,266 wrist
predictions from cafe, hallway and night-time truck footage, looked at in bins of score. Hands
scoring under 2 were right about one time in six, between 2.5 and 3 about four in five, over 4.5
every time; wrists turn reliable a little earlier.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .config import MODELS, WHOLEBODY_MODELS, setting

BODY, LEFT_HAND, RIGHT_HAND = slice(0, 17), slice(91, 112), slice(112, 133)
HANDS = (LEFT_HAND, RIGHT_HAND)             # index 0 = left, 1 = right, as everywhere in the engine
MEAN = np.array([123.675, 116.28, 103.53], np.float32)     # RGB
STD = np.array([58.395, 57.12, 57.375], np.float32)
PADDING = 1.25          # the person box is grown by this before it is cut out, as the model was trained
SIMCC_SPLIT = 2.0       # the model answers in half pixels of its input
# Score at which a keypoint is as likely right as wrong, and how fast confidence changes around it.
BODY_HALF, HAND_HALF, SLOPE = 2.3, 2.8, 0.5


def confidence(score: np.ndarray, half: float) -> np.ndarray:
    """Model score -> 0-1, with 0.5 at `half` (a logistic curve, so thresholds keep their order)."""
    return (1.0 / (1.0 + np.exp(-(np.asarray(score, np.float32) - half) / SLOPE))).astype(np.float32)


def accelerated() -> bool:
    """Is there something faster than the CPU to run this model on?"""
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    if setting("DEVICE", "") == "cpu":
        return False
    have = ort.get_available_providers()
    return "CoreMLExecutionProvider" in have or "CUDAExecutionProvider" in have


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return setting("WHOLEBODY", "") != "off"


def model_name() -> str:
    """Which model to use: WHOLEBODY in the settings, else the large one on a GPU and the small one on a CPU."""
    name = setting("WHOLEBODY", "")
    if name in WHOLEBODY_MODELS:
        return name
    return "x" if accelerated() else "m"


def label(name: str | None = None) -> str:
    """The model's name as it is written into a run and a label card, e.g. "RTMW-x 384x288"."""
    name = name or model_name()
    w, h = WHOLEBODY_MODELS[name]["size"]
    return f"RTMW-{name} {h}x{w}"


def weights(name: str) -> Path:
    """The model file, fetched into the models folder on first use."""
    spec = WHOLEBODY_MODELS[name]
    path = MODELS / spec["file"]
    if not path.exists():
        MODELS.mkdir(parents=True, exist_ok=True)
        print(f"downloading {spec['file']} ...", flush=True)
        archive = path.with_suffix(".zip.part")
        urllib.request.urlretrieve(spec["url"], archive)
        with zipfile.ZipFile(archive) as z:
            member = next(m for m in z.namelist() if m.endswith(".onnx"))
            tmp = path.with_suffix(".onnx.part")
            tmp.write_bytes(z.read(member))
        os.replace(tmp, path)
        archive.unlink(missing_ok=True)
    return path


def crop_box(box: np.ndarray, size: tuple[int, int], padding: float = PADDING) -> tuple[np.ndarray, np.ndarray]:
    """Centre and size (w, h) of the area cut out for a person box (x1, y1, x2, y2): padded, at the model's shape."""
    w, h = size
    x1, y1, x2, y2 = (float(v) for v in box)
    centre = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    bw, bh = (x2 - x1) * padding, (y2 - y1) * padding
    if bw > bh * w / h:
        bh = bw * h / w
    else:
        bw = bh * w / h
    return centre, np.array([bw, bh], np.float32)


def cut(frame: np.ndarray, centre: np.ndarray, scale: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The area around `centre`, `scale` wide and high, resized to the model's input (w, h). Black outside the frame."""
    w, h = size
    src = np.float32([[centre[0] - scale[0] / 2, centre[1] - scale[1] / 2],
                      [centre[0] + scale[0] / 2, centre[1] - scale[1] / 2],
                      [centre[0] - scale[0] / 2, centre[1] + scale[1] / 2]])
    dst = np.float32([[0, 0], [w, 0], [0, h]])
    return cv2.warpAffine(frame, cv2.getAffineTransform(src, dst), (w, h), flags=cv2.INTER_LINEAR)


def decode(simcc_x: np.ndarray, simcc_y: np.ndarray, centres: np.ndarray, scales: np.ndarray,
           size: tuple[int, int]) -> np.ndarray:
    """The model's answer -> (n, K, 3): x, y in the frame and the score."""
    x = simcc_x.argmax(axis=2).astype(np.float32) / SIMCC_SPLIT
    y = simcc_y.argmax(axis=2).astype(np.float32) / SIMCC_SPLIT
    score = 0.5 * (simcc_x.max(axis=2) + simcc_y.max(axis=2))
    out = np.empty((len(x), x.shape[1], 3), np.float32)
    out[:, :, 0] = x / size[0] * scales[:, None, 0] + centres[:, None, 0] - scales[:, None, 0] / 2
    out[:, :, 1] = y / size[1] * scales[:, None, 1] + centres[:, None, 1] - scales[:, None, 1] / 2
    out[:, :, 2] = score
    return out


class WholeBody:
    """The model, loaded once. Call it with a frame and the boxes of the people to look at."""

    def __init__(self, name: str | None = None, batch: int | None = None):
        import onnxruntime as ort

        self.name = name or model_name()
        spec = WHOLEBODY_MODELS[self.name]
        self.size = tuple(spec["size"])                 # (w, h) of the model's input
        self.scores = tuple(spec.get("scores", (1.0, 1.0)))     # this model's scores -> the scale the thresholds are on
        # people per model call: POSE_BATCH, as for stage 2's model (a small server sets it low to save memory)
        self.batch = batch or int(setting("POSE_BATCH", "8"))
        options = ort.SessionOptions()
        options.log_severity_level = 3
        threads = int(setting("THREADS", "0"))
        if threads > 0:
            options.intra_op_num_threads = threads
        have = ort.get_available_providers()
        want: list = []
        if setting("DEVICE", "") != "cpu":
            if "CUDAExecutionProvider" in have:
                want.append("CUDAExecutionProvider")
            if "CoreMLExecutionProvider" in have:
                # As an ML program the model is ready in about 2 s and runs at 16 ms a person on an M1 Pro;
                # in Core ML's older format the first call alone took 10 s.
                want.append(("CoreMLExecutionProvider", {"ModelFormat": "MLProgram"}))
        path = str(weights(self.name))
        try:
            self.session = ort.InferenceSession(path, options, providers=want + ["CPUExecutionProvider"])
        except Exception:       # a GPU provider that will not take the model: the CPU always does
            self.session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        self.provider = self.session.get_providers()[0].replace("ExecutionProvider", "")
        self._input = self.session.get_inputs()[0].name

    def __call__(self, frame: np.ndarray, boxes) -> np.ndarray:
        """Keypoints of the people in `boxes` (n, 4: x1, y1, x2, y2) of one frame: (n, 133, 3) x, y, score."""
        return self.many([(frame, box) for box in boxes])

    def many(self, items: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Keypoints for (frame, box) pairs, which may come from different frames."""
        out = np.zeros((len(items), 133, 3), np.float32)
        w, h = self.size
        for a in range(0, len(items), self.batch):
            part = items[a:a + self.batch]
            placed = [crop_box(box, self.size) for _, box in part]
            centres = np.stack([c for c, _ in placed])
            scales = np.stack([s for _, s in placed])
            # A GPU provider prepares the model for one batch size and starts over when the size changes,
            # which costs seconds: so a short last batch is filled up to the usual size there.
            n = self.batch if self.provider != "CPU" else len(part)
            tensor = np.zeros((n, 3, h, w), np.float32)
            for i, ((frame, _), (c, s)) in enumerate(zip(part, placed)):
                rgb = cut(frame, c, s, self.size)[:, :, ::-1].astype(np.float32)     # the model was trained on RGB
                tensor[i] = ((rgb - MEAN) / STD).transpose(2, 0, 1)
            simcc_x, simcc_y = self.session.run(None, {self._input: tensor})
            out[a:a + len(part)] = decode(simcc_x[:len(part)], simcc_y[:len(part)], centres, scales, self.size)
        body, hand = getattr(self, "scores", (1.0, 1.0))
        out[:, :LEFT_HAND.start, 2] *= body
        out[:, LEFT_HAND.start:, 2] *= hand
        return out


_shared: WholeBody | None = None


def shared() -> WholeBody:
    """The model, loaded once per process and used by whoever needs it (the live view, the second look)."""
    global _shared
    if _shared is None:
        _shared = WholeBody()
    return _shared


def body17(kp: np.ndarray) -> np.ndarray:
    """(…, 133, 3) -> (…, 17, 3): the body joints, with the score turned into a 0-1 confidence."""
    out = np.array(kp[..., BODY, :], np.float32)
    out[..., 2] = confidence(out[..., 2], BODY_HALF)
    return out


def hand_boxes(kp: np.ndarray, scale: float | np.ndarray, grow: float = 0.3, floor: float = 0.3,
               ceiling: float = 1.1) -> tuple[np.ndarray, np.ndarray]:
    """Where each hand is, from its 21 keypoints.

    kp is (…, 133, 3). Returns boxes (…, 2, 4) as cx, cy, w, h in frame pixels (left hand, then right)
    and confidences (…, 2) in 0-1. The box holds the keypoints the model is surest of, grown a little
    so that fingertips and whatever they hold are inside; `scale` (torso length in px) bounds its size,
    so a single stray keypoint cannot make a hand the size of a person.
    """
    kp = np.asarray(kp, np.float32)
    lead = kp.shape[:-2]
    flat = kp.reshape(-1, 133, 3)
    scales = np.broadcast_to(np.asarray(scale, np.float32).reshape(-1), (len(flat),)) if np.ndim(scale) else \
        np.full(len(flat), float(scale), np.float32)
    boxes = np.zeros((len(flat), 2, 4), np.float32)
    conf = np.zeros((len(flat), 2), np.float32)
    for i, person in enumerate(flat):
        for s, part in enumerate(HANDS):
            pts = person[part]
            order = np.argsort(-pts[:, 2])
            best = pts[order[:14]]                                   # two thirds of the hand: the fingers that are seen
            conf[i, s] = float(confidence(np.median(best[:, 2]), HAND_HALF))
            lo, hi = best[:, :2].min(axis=0), best[:, :2].max(axis=0)
            centre = (lo + hi) / 2
            side = float(np.clip(max(hi[0] - lo[0], hi[1] - lo[1]) * (1 + grow), floor * scales[i], ceiling * scales[i]))
            w = float(np.clip((hi[0] - lo[0]) * (1 + grow), 0.8 * side, side))
            h = float(np.clip((hi[1] - lo[1]) * (1 + grow), 0.8 * side, side))
            boxes[i, s] = (centre[0], centre[1], w, h)
    return boxes.reshape(*lead, 2, 4), conf.reshape(*lead, 2)
