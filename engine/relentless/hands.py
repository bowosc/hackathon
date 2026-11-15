"""Hand tracking: where each hand of a person is, for the clips the engine cuts and the stretch it replays.

Stage 2's pose model follows people through the whole footage, and its wrists are good enough to notice
that hands are busy. They are not good enough to draw: on a person 200 pixels tall, a wrist in a dark
sleeve against a dark coat lands anywhere along the arm. So once the engine knows which moments it will
cut, it looks at those people again with the whole-body model (wholebody.py), one person at a time and
enlarged, and takes from it

  the body joints    for the skeleton the job page draws, and to decide again which hand is the busy one
  both hands         a box around each, from 21 keypoints per hand

What the model is not sure of is not kept: a hand counts as located only where its keypoints are confident,
sit together, hang on the wrist they belong to, at the end of an arm the model can read, and stay located
for more than a moment. A person the model cannot read at all (fog, glare, a blur next to the lens) gets no
hands and no skeleton, rather than hands in the wrong place.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import activity, media, wholebody
from .people import L_ELBOW, L_SHOULDER, L_WRIST, R_ELBOW, R_SHOULDER, R_WRIST, Tracklet

SIDES = ("left", "right")


@dataclass
class HandsConfig:
    hand_conf: float = 0.5        # a hand is located from this confidence of its keypoints
    wrist_conf: float = 0.3       # and its wrist must at least be plausible
    readable: float = 0.7         # mean confidence of an arm's shoulder, elbow and wrist below which that arm is not readable
    readable_s: float = 1.5       # ... judged over this long, so one bad sample does not blank a good stretch
    clip_readable: float = 0.75   # share of a clip in which the person must be readable for it to be a hand clip
    span_readable: float = 0.3    # share of a busy stretch below which the activity filter's verdict is not trusted
    span_every_s: float = 1.0     # the activity check looks at a busy stretch about this often
    span_samples: int = 5         # and at least this many times (3 when the budget is tight)
    max_spread: float = 1.0       # torso lengths: keypoints of one hand further apart than this are not one hand
    max_off_wrist: float = 0.6    # torso lengths between a hand and the wrist it should hang on
    max_jump: float = 1.0         # torso lengths: a hand that is this far from both its neighbours, which agree, is a slip
    min_run_s: float = 0.3        # a hand located for less than this, with nothing around it, is dropped
    max_gap_s: float = 0.35       # shorter holes in a located hand are bridged
    pad_s: float = 1.0            # the clip's window is widened by this for the body measurements, which use windows


@dataclass
class Track:
    """One person over a stretch of footage, as the whole-body model sees them. Arrays share `frames`."""

    frames: np.ndarray        # (n,) source frame of each sample
    box: np.ndarray           # (n, 4) the person box the model was given
    kp: np.ndarray            # (n, 17, 3) body joints: x, y, confidence 0-1
    hand: np.ndarray          # (n, 2, 4) left and right hand box: cx, cy, w, h in source pixels
    hand_conf: np.ndarray     # (n, 2) confidence 0-1 of each hand's keypoints
    seen: np.ndarray          # (n, 2) the hand is located here
    scale: np.ndarray         # (n,) torso length in px
    readable: np.ndarray      # (n,) the person can be read here: at least one arm, shoulder to wrist
    step: int
    fps: float

    def tracklet(self, tid: int = 0) -> Tracklet:
        """The same person as a stage 2 tracklet, with these keypoints: what stages 3 and 4 work on."""
        return Tracklet(tid, self.frames, self.box, self.kp, np.ones(len(self.frames), np.float32), self.step, self.fps)

    def at(self, frame: int) -> int:
        return int(np.clip(round((frame - int(self.frames[0])) / self.step), 0, len(self.frames) - 1))

    def on(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Body joints (m, 17, 3), hand boxes (m, 2, 4), located flags (m, 2) and whether the person is readable
        (m,) at any source frames, between the samples by interpolation. A joint is as sure as the less sure
        of the two samples around it."""
        frames = np.asarray(frames, np.float64)
        pos = np.clip(np.interp(frames, self.frames, np.arange(len(self.frames))), 0, len(self.frames) - 1)
        lo = np.floor(pos).astype(int)
        w = (pos - lo).astype(np.float32)
        hi = np.where(w > 0, np.minimum(lo + 1, len(self.frames) - 1), lo)     # on a sample: that sample alone
        kp = self.kp[lo] * (1 - w)[:, None, None] + self.kp[hi] * w[:, None, None]
        kp[:, :, 2] = np.minimum(self.kp[lo, :, 2], self.kp[hi, :, 2])
        hand = self.hand[lo] * (1 - w)[:, None, None] + self.hand[hi] * w[:, None, None]
        seen = self.seen[lo] & self.seen[hi]
        return kp, hand, seen, self.readable[lo] & self.readable[hi]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[a, b) index ranges where mask is on."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _median(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < 2:
        return x
    pad = k // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)])
    return np.median(np.stack([xp[i:i + len(x)] for i in range(k)]), axis=0)


def assemble(frames: np.ndarray, box: np.ndarray, kp133: np.ndarray, step: int, fps: float,
             cfg: HandsConfig | None = None) -> Track:
    """The model's raw answers for one person -> a track: confidences on the engine's scale, hand boxes,
    and the hands cleaned up over time."""
    cfg = cfg or HandsConfig()
    n = len(frames)
    dt = step / fps
    kp = wholebody.body17(kp133)
    shell = Tracklet(0, frames, box, kp, np.ones(n, np.float32), step, fps)
    scale = activity.body_scale(shell, 0.45)

    # Is the person readable? Judged arm by arm (hips are behind a counter half the time, and one arm is
    # often behind the body or out of the frame: that must not cost the hand that is in plain view).
    k = max(1, int(round(cfg.readable_s / dt))) // 2 * 2 + 1
    arm_ok = np.stack([_median(kp[:, [L_SHOULDER + s, L_ELBOW + s, L_WRIST + s], 2].mean(axis=1), k) >= cfg.readable
                       for s in (0, 1)], axis=1)
    readable = arm_ok.any(axis=1)

    hand, conf = wholebody.hand_boxes(kp133, scale)
    seen = np.zeros((n, 2), bool)
    for s, part in enumerate(wholebody.HANDS):
        pts = kp133[:, part]
        order = np.argsort(-pts[:, :, 2], axis=1)[:, :14]
        best = np.take_along_axis(pts[:, :, :2], order[:, :, None], axis=1)
        spread = np.linalg.norm(best.max(axis=1) - best.min(axis=1), axis=1) / scale
        wrist = kp[:, (L_WRIST, R_WRIST)[s]]
        off = np.linalg.norm(hand[:, s, :2] - wrist[:, :2], axis=1) / scale
        seen[:, s] = (arm_ok[:, s] & (conf[:, s] >= cfg.hand_conf) & (wrist[:, 2] >= cfg.wrist_conf)
                      & (spread <= cfg.max_spread) & (off <= cfg.max_off_wrist))

    hand, seen = tidy(hand, seen, scale, dt, cfg)
    return Track(frames, box, kp, hand, conf, seen, scale, readable, step, fps)


def glance(kp133: np.ndarray, box: np.ndarray, cfg: HandsConfig | None = None) -> tuple[np.ndarray, dict, bool]:
    """One person in one frame, with nothing before or after to go by: the live view of person tracking.
    Returns the body joints (17, 3), the located hands as {"l": [x1, y1, x2, y2], "r": [...]}, and whether
    the person is readable (if not, the joints carry no confidence and there are no hands)."""
    cfg = cfg or HandsConfig()
    kp = wholebody.body17(kp133)
    sh, hip = kp[[L_SHOULDER, R_SHOULDER]], kp[[11, 12]]
    torso = float(np.linalg.norm(sh[:, :2].mean(axis=0) - hip[:, :2].mean(axis=0)))
    scale = torso if (sh[:, 2] > 0.45).all() and (hip[:, 2] > 0.45).all() and torso > 8 else 0.3 * float(box[3] - box[1])
    scale = max(scale, 8.0)
    out = {}
    arm_ok = [bool(kp[[L_SHOULDER + s, L_ELBOW + s, L_WRIST + s], 2].mean() >= cfg.readable) for s in (0, 1)]
    clear = any(arm_ok)
    if not clear:
        kp[:, 2] = 0.0
    else:
        boxes, conf = wholebody.hand_boxes(kp133, scale)
        for s, name in enumerate(("l", "r")):
            wrist = kp[(L_WRIST, R_WRIST)[s]]
            off = float(np.linalg.norm(boxes[s, :2] - wrist[:2])) / scale
            if arm_ok[s] and conf[s] >= cfg.hand_conf and wrist[2] >= cfg.wrist_conf and off <= cfg.max_off_wrist:
                cx, cy, w, h = boxes[s]
                out[name] = [int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2))]
    return kp, out, clear


def tidy(hand: np.ndarray, seen: np.ndarray, scale: np.ndarray, dt: float, cfg: HandsConfig) -> tuple[np.ndarray, np.ndarray]:
    """Clean each hand's path over time: single slips out, short holes bridged, lone blips dropped, jitter smoothed."""
    hand, seen = hand.copy(), seen.copy()
    n = len(hand)
    gap, run = int(round(cfg.max_gap_s / dt)), max(1, int(round(cfg.min_run_s / dt)))
    for s in (0, 1):
        c = hand[:, s, :2]
        # a sample far from both neighbours, while the neighbours agree with each other, is the model slipping
        for i in range(1, n - 1):
            if seen[i - 1, s] and seen[i, s] and seen[i + 1, s]:
                mid = (c[i - 1] + c[i + 1]) / 2
                if (np.linalg.norm(c[i] - mid) > cfg.max_jump * scale[i]
                        and np.linalg.norm(c[i - 1] - c[i + 1]) < 0.5 * cfg.max_jump * scale[i]):
                    hand[i, s] = (hand[i - 1, s] + hand[i + 1, s]) / 2
        # bridge short holes between two sightings
        on = seen[:, s].copy()
        runs = _runs(on)
        for (_, b), (a2, _) in zip(runs[:-1], runs[1:]):
            if a2 - b <= gap:
                for i in range(b, a2):
                    w = (i - b + 1) / (a2 - b + 1)
                    hand[i, s] = hand[b - 1, s] * (1 - w) + hand[a2, s] * w
                on[b:a2] = True
        # a sighting shorter than a moment, on its own, proves nothing
        for a, b in _runs(on):
            if b - a < run and not (a == 0 or b == n):
                on[a:b] = False
        seen[:, s] = on
        # steady the box: the centre a little, the size more (a hand does not change size from sample to sample)
        for a, b in _runs(on):
            if b - a >= 3:
                seg = hand[a:b, s].copy()
                seg[1:-1, :2] = 0.25 * seg[:-2, :2] + 0.5 * seg[1:-1, :2] + 0.25 * seg[2:, :2]
                seg[:, 2:] = _median(seg[:, 2:], 5)
                hand[a:b, s] = seg
        # where the hand is not located, carry the nearest located box, so the numbers stay finite and near the truth
        idx = np.flatnonzero(on)
        if len(idx):
            for i in np.flatnonzero(~on):
                hand[i, s] = hand[idx[np.abs(idx - i).argmin()], s]
    return hand, seen


# ---------- running the model over the source ----------

@dataclass
class Request:
    """Look at one tracklet again, on samples lo..hi (indices into the tracklet), every `stride`-th of them."""

    key: object
    tracklet: Tracklet
    lo: int
    hi: int
    stride: int = 1

    def samples(self) -> np.ndarray:
        return np.arange(self.lo, self.hi, self.stride)


def clip_request(key, t: Tracklet, start: int, end: int, cfg: HandsConfig | None = None) -> Request:
    """The samples of a tracklet that cover a clip [start, end), widened a little on both sides."""
    cfg = cfg or HandsConfig()
    pad = int(round(cfg.pad_s * t.fps / t.step))
    return Request(key, t, max(0, t.at(start) - pad), min(len(t.frames), t.at(end - 1) + 1 + pad))


def stretch_request(key, t: Tracklet, first: int, last: int, stride: int = 1) -> Request | None:
    """The samples of a tracklet inside [first, last), or None if it is not in view then."""
    inside = np.flatnonzero((t.frames >= first) & (t.frames < last))
    return Request(key, t, int(inside[0]), int(inside[-1]) + 1, stride) if len(inside) else None


def span_requests(tracklets: list[Tracklet], active: dict[int, list[tuple[int, int]]], budget: int,
                  cfg: HandsConfig | None = None) -> list[Request]:
    """A sparse second look at every busy stretch the activity filter found: about once a second, at least a few
    times, `budget` samples in all (but never fewer than 3 a stretch). Keys are ("span", tracklet id, index)."""
    cfg = cfg or HandsConfig()
    spans = []
    for t in tracklets:
        for k, (a, b) in enumerate(active.get(t.id, [])):
            inside = np.flatnonzero((t.frames >= a) & (t.frames < b))
            if len(inside) >= 2:
                spans.append((t, k, inside, max(cfg.span_samples, int(round((b - a) / t.fps / cfg.span_every_s)))))
    total = sum(want for *_, want in spans)
    thin = min(1.0, budget / max(1, total))
    out = []
    for t, k, inside, want in spans:
        want = min(len(inside), max(3, int(round(want * thin))))
        stride = max(1, len(inside) // want)
        out.append(Request(("span", t.id, k), t, int(inside[0]), int(inside[-1]) + 1, stride))
    return out


def unreadable_spans(looked: dict, cfg: HandsConfig | None = None) -> dict[int, dict[int, float]]:
    """{tracklet id: {span index: share readable}} for the busy stretches in which the person could be read in
    less than `span_readable` of the samples (a span the model returned nothing for is not judged)."""
    cfg = cfg or HandsConfig()
    out: dict[int, dict[int, float]] = {}
    for key, track in looked.items():
        if isinstance(key, tuple) and len(key) == 3 and key[0] == "span":
            share = float(track.readable.mean()) if len(track.readable) else 1.0
            if share < cfg.span_readable:
                out.setdefault(key[1], {})[key[2]] = share
    return out


def look(src: Path, requests: list[Request], model: wholebody.WholeBody | None = None, progress=None,
         cfg: HandsConfig | None = None) -> dict:
    """Run the whole-body model for every request, in one pass over the source. Returns {key: Track}.

    A person-sample asked for twice (a clip inside the replayed stretch) goes through the model once."""
    if not requests:
        return {}
    model = model or wholebody.shared()
    need: dict[int, dict[int, np.ndarray]] = {}        # frame -> {tracklet id: box}
    for r in requests:
        for i in r.samples():
            need.setdefault(int(r.tracklet.frames[i]), {})[r.tracklet.id] = r.tracklet.box[i]
    answers: dict[tuple[int, int], np.ndarray] = {}
    total, done = sum(len(v) for v in need.values()), 0
    pending: list[tuple[np.ndarray, np.ndarray]] = []
    owners: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal done
        for owner, kp in zip(owners, model.many(pending)):
            answers[owner] = kp
        done += len(pending)
        pending.clear()
        owners.clear()
        if progress:
            progress(done / max(1, total))

    with media.Reader(src) as rd:
        for f in sorted(need):
            frame = rd.read(f)
            if frame is None:
                break
            for tid, box in need[f].items():
                pending.append((frame, box))
                owners.append((tid, f))
            if len(pending) >= model.batch:
                flush()
        if pending:
            flush()

    out = {}
    for r in requests:
        idx = [i for i in r.samples() if (r.tracklet.id, int(r.tracklet.frames[i])) in answers]
        if len(idx) < 2:
            continue
        t = r.tracklet
        kp133 = np.stack([answers[(t.id, int(t.frames[i]))] for i in idx])
        out[r.key] = assemble(t.frames[idx], t.box[idx], kp133, t.step * r.stride, t.fps, cfg)
    return out


def _look_child(src: Path, requests: list[Request], out: Path, updates, cfg) -> None:
    result = look(src, requests, progress=updates.put, cfg=cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(result, f)


def look_apart(src: Path, requests: list[Request], out: Path, progress=None, cfg: HandsConfig | None = None) -> dict:
    """look() in a child process that exits when it is done, so the model's memory is back before the
    upscaler loads its own (the low-memory mode of a small server; see people.track_apart)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    updates = ctx.Queue()
    child = ctx.Process(target=_look_child, args=(src, requests, out, updates, cfg), daemon=True)
    child.start()
    while child.is_alive():
        try:
            frac = updates.get(timeout=0.5)
        except Exception:   # nothing new yet
            continue
        if progress:
            progress(frac)
    child.join()
    try:
        if child.exitcode != 0:
            raise RuntimeError(f"hand tracking stopped with exit code {child.exitcode}")
        with open(out, "rb") as f:
            return pickle.load(f)
    finally:        # whatever happened, the child's file does not stay behind
        out.unlink(missing_ok=True)


# ---------- what a clip takes from a track ----------

@dataclass
class ClipHands:
    """What the second look says about one clip."""

    hands: list[str]                          # the busy hands, as "left" / "right"; empty if no hand could be located
    crop: tuple[int, int, int, int] | None    # a better hand-zoom crop, or None to keep the planned one
    located: dict[str, float]                 # share of the clip in which each hand is located
    readable: float                           # share of the clip in which the person can be read
    notes: list[str] = field(default_factory=list)


def busy_sides(track: Track, start: int, end: int, min_busy_s: float = 0.2, min_located: float = 0.5,
               min_travel: float = 0.25) -> tuple[list[int], np.ndarray]:
    """Which hands work in [start, end): the ones that move while raised or reaching, as stage 3 measures it,
    now on keypoints that are right. Returns (sides, share of the clip each hand is located)."""
    a, b = track.at(start), track.at(end - 1) + 1
    dt = track.step / track.fps
    sig = activity.signals(track.tracklet())
    located = track.seen[a:b].mean(axis=0) if b > a else np.zeros(2)
    busy = (sig.busy[a:b] & track.seen[a:b]).sum(axis=0) * dt
    sides = [s for s in (0, 1) if busy[s] >= min_busy_s and located[s] >= min_located]
    if not sides:
        # nothing passes the wrist-speed bar: take the located hand that travels, if one does
        travel = []
        for s in (0, 1):
            pts = track.hand[a:b, s, :2][track.seen[a:b, s]]
            travel.append(float(np.linalg.norm(np.ptp(pts, axis=0))) / float(np.median(track.scale[a:b])) if len(pts) > 1 else 0.0)
        sides = [s for s in (0, 1) if located[s] >= min_located and travel[s] >= min_travel]
        if len(sides) == 2 and min(travel) < 0.5 * max(travel):
            sides = [int(np.argmax(travel))]
    return sides, located


def for_clip(track: Track, start: int, end: int, planned_hands: list[str], frame_size: tuple[int, int],
             zoom=None, cfg: HandsConfig | None = None) -> ClipHands:
    """Decide a clip's busy hands and crop from its track. No hands means: not a hand clip.

    On the clips this was tuned on, every person who can be seen is readable in 98% of the clip or more, and
    the one who cannot (a blur next to the lens, at night, in fog) in half of it at most: there the model's
    arm crosses the bar every other moment, and whatever hangs on the end of it is a guess."""
    from . import interactions      # here, not at the top: interactions does not need this module

    zoom, cfg = zoom or interactions.ZoomConfig(), cfg or HandsConfig()
    a, b = track.at(start), track.at(end - 1) + 1
    sides, located = busy_sides(track, start, end, min_busy_s=zoom.min_busy_s)
    readable = float(track.readable[a:b].mean()) if b > a else 0.0
    notes = []
    if readable < cfg.clip_readable:
        sides = []
        notes.append("the person cannot be read for most of this clip" if readable > 0
                     else "no hand could be located in this clip")
    elif not sides:
        # The person is readable but no hand passes the motion test on the careful keypoints (stage 2's wrist
        # was jitter, or the motion is slow). Keep the hand the motion filter flagged if it is in view at
        # least half the time, else a hand that is in view nearly throughout: the labeler decides the rest.
        sides = ([s for s, name in enumerate(SIDES) if name in planned_hands and located[s] >= 0.5]
                 or [s for s in (0, 1) if located[s] >= 0.75])
        notes.append("no working hand located; kept the hand the motion filter flagged" if sides
                     else "no hand could be located in this clip")
    crop = None
    if sides:
        pts = np.concatenate([track.hand[a:b, s, :2][track.seen[a:b, s]] for s in sides])
        scale = float(np.median(track.scale[a:b]))
        crop = interactions.plan_crop(pts, scale, frame_size, zoom)
    return ClipHands([SIDES[s] for s in sides], crop, {SIDES[s]: round(float(located[s]), 2) for s in (0, 1)},
                     round(readable, 2), notes)


def tracks_json(track: Track, start: int, end: int, crop: tuple[int, int, int, int], busy: list[str],
                model: str) -> dict:
    """hands.json of a clip: both hands for every frame of [start, end), in coordinates of the crop (0-1).

    x, y is the centre of the hand's box, w and h its size. Where `seen` is false the hand is not located
    in that frame of the clip (hidden, too unclear, or outside the crop) and the numbers are only the
    nearest place it was located: do not draw or train on them."""
    x, y, w, h = crop
    frames = np.arange(start, end)
    out = {"fps": track.fps, "frames": int(len(frames)),
           "source": f"whole-body pose model ({model}), 21 keypoints per hand", "busy": list(busy)}
    sizes = []
    for s, name in enumerate(SIDES):
        cx = np.interp(frames, track.frames, track.hand[:, s, 0])
        cy = np.interp(frames, track.frames, track.hand[:, s, 1])
        bw = np.interp(frames, track.frames, track.hand[:, s, 2])
        bh = np.interp(frames, track.frames, track.hand[:, s, 3])
        conf = np.interp(frames, track.frames, track.hand_conf[:, s])
        seen = np.interp(frames, track.frames, track.seen[:, s].astype(float)) > 0.5
        seen &= (cx >= x) & (cx <= x + w) & (cy >= y) & (cy <= y + h)      # located, but outside this clip's frame
        out[name] = {"x": [round(float(v), 4) for v in (cx - x) / w], "y": [round(float(v), 4) for v in (cy - y) / h],
                     "w": [round(float(v), 4) for v in bw / w], "h": [round(float(v), 4) for v in bh / h],
                     "conf": [round(float(v), 2) for v in conf], "seen": [bool(v) for v in seen]}
        if seen.any():
            sizes.append((float(np.median(bw[seen])), float(np.median(bh[seen]))))
    side = sizes or [(0.4 * float(np.median(track.scale)),) * 2]
    # one size for readers that draw a fixed box (older pages): the usual size of a located hand
    out["box_size"] = [round(float(np.median([s[0] for s in side])) / w, 4), round(float(np.median([s[1] for s in side])) / h, 4)]
    return out


def same_path(video: Path, track: Track, crop: tuple[int, int, int, int], clip_start: int, start_s: float,
              seconds: float, model: wholebody.WholeBody | None = None) -> dict:
    """Are the hands in `video` where the clip's track says they are? A measurement, for the variant check.

    `video` shows the clip's crop from `start_s` seconds into the clip, for `seconds` (the stretch that was
    sent to the edit model, or what came back). The whole-body model looks at it with the person's box
    carried over from the source, and for every sample in which the track has a hand located, the hand it
    finds is compared with the track's. Returns {"left": {...}, "right": {...}} with, per hand: `n` samples,
    `found` (share in which a hand was found at all) and `same` (share in which it is within one hand size of
    the track's); None for a hand the track does not locate. Run on the untouched stretch first: that is the
    score a perfect edit would get, since a crop shows the model less of the person than the full frame did.
    """
    model = model or wholebody.shared()
    x, y, w, h = crop
    f0 = clip_start + int(round(start_s * track.fps))
    idx = [i for i, f in enumerate(track.frames) if f0 <= f < f0 + int(round(seconds * track.fps))]
    items, kept = [], []
    with media.Reader(video) as rd:
        sx, sy = rd.width / w, rd.height / h
        for i in idx:
            frame = rd.read(min(rd.count - 1, int(round((int(track.frames[i]) - f0) / track.fps * rd.fps))))
            if frame is None:
                break
            b = track.box[i]
            items.append((frame, np.array([(b[0] - x) * sx, (b[1] - y) * sy, (b[2] - x) * sx, (b[3] - y) * sy], np.float32)))
            kept.append(i)
    out: dict = {"left": None, "right": None}
    if len(kept) < 3:
        return out
    kp = model.many(items)
    boxes, conf = wholebody.hand_boxes(kp, track.scale[kept] * sx)
    for s, name in enumerate(SIDES):
        there = track.hand[kept, s].copy()
        there[:, 0], there[:, 1] = (there[:, 0] - x) * sx, (there[:, 1] - y) * sy
        with np.errstate(invalid="ignore"):     # only hands that are inside the video can be found in it
            inside = (there[:, 0] >= 0) & (there[:, 0] <= w * sx) & (there[:, 1] >= 0) & (there[:, 1] <= h * sy)
        located = track.seen[kept, s] & inside
        if located.sum() < 5:
            continue
        size = np.maximum(there[:, 2:].max(axis=1) * sx, 1.0)
        off = np.linalg.norm(boxes[:, s, :2] - there[:, :2], axis=1) / size
        found = conf[:, s] >= 0.5
        out[name] = {"n": int(located.sum()), "found": round(float(found[located].mean()), 2),
                     "same": round(float((found & (off < 1.0))[located].mean()), 2)}
    return out


def person_rows(track: Track, frames: np.ndarray) -> dict[int, tuple[np.ndarray, dict, bool]]:
    """A track as the job page draws it, at the given source frames: {frame: (body joints, located hands,
    readable)}. Hands are {"l": [x1, y1, x2, y2], "r": [...]} in source pixels, with only the hands that are
    located. Where the person cannot be read, the joints carry no confidence: a few limbs the model happens
    to be sure of, on a person it cannot make out, say nothing true."""
    kp, hand, seen, readable = track.on(frames)
    rows = {}
    for i, f in enumerate(frames):
        hands = {}
        for s, name in enumerate(("l", "r")):
            if seen[i, s]:
                cx, cy, w, h = hand[i, s]
                hands[name] = [int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2))]
        joints = kp[i].copy()
        if not readable[i]:
            joints[:, 2] = 0.0
        rows[int(f)] = (joints, hands, bool(readable[i]))
    return rows
