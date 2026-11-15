"""Stage 4 - hand zoom: find the moments where a hand works on something and plan a tight crop around it.

A burst of wrist motion with the hand raised or reaching out, while the person stays in place,
is a candidate interaction. Bursts close together in time become one clip of 2-8 s. The crop is
fixed for the whole clip (so objects stay still in frame), covers everywhere the busy hand went
plus room for the object, and is widened to 16:9 because the upscaler, the catalog and the
multiply models all work in 16:9.

Whether the hand really touched an object is decided later by the labeler; this stage only has
to be cheap and not miss much.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import media
from .activity import Signals
from .people import Tracklet


@dataclass
class ZoomConfig:
    event_gap_s: float = 1.0     # bursts closer than this belong to one clip
    min_busy_s: float = 0.2      # a clip needs at least this much busy time
    pad_s: float = 0.7           # lead-in and lead-out around the motion
    min_s: float = 2.0
    max_s: float = 8.0
    min_scale: float = 38.0      # torso length in px; smaller people give hands of only a few pixels
    min_seen: float = 0.5        # share of the clip in which the busy hand must be detected
    margin: float = 0.6          # room around the hand path, in torso lengths (space for the object)
    min_side: float = 1.4        # smallest crop height, in torso lengths
    min_px: int = 96             # smallest crop height in source pixels
    aspect: float = 16 / 9
    max_walk_share: float = 0.65  # skip clips where the person is mostly walking
    max_drift: float = 0.8       # skip clips where the person ends up this many body heights from where they started
    max_reach_h: float = 3.6     # torso lengths: a taller hand path than this means the person moved, not the hand
    max_reach_w: float = 4.6


@dataclass
class Candidate:
    """One planned hand-zoom clip. Frames are source frame indices, [start, end)."""

    track: int
    start: int
    end: int
    crop: tuple[int, int, int, int]   # x, y, w, h in source pixels
    hands: list[str]                  # which hands are busy: "left", "right"
    busy_s: float
    peak_speed: float                 # torso lengths per second
    extent: float                     # how far the busy hand travelled, in torso lengths
    scale: float                      # torso length in px (how big the person is)
    seen: float                       # share of samples where the busy hand is detected
    walk: float = 0.0                 # share of the clip in which the person is walking
    drift: float = 0.0                # how far the person ends from where they started, in body heights
    score: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)


def _runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Index ranges [a, b) where mask is on, joining runs separated by at most max_gap samples."""
    out: list[list[int]] = []
    for i in np.flatnonzero(mask):
        if out and i - out[-1][1] <= max_gap:
            out[-1][1] = int(i) + 1
        else:
            out.append([int(i), int(i) + 1])
    return [(a, b) for a, b in out]


def find(t: Tracklet, sig: Signals, spans: list[tuple[int, int]], frame_size: tuple[int, int],
         cfg: ZoomConfig | None = None) -> list[Candidate]:
    """Plan hand-zoom clips for one tracklet, restricted to its active spans."""
    cfg = cfg or ZoomConfig()
    dt = t.step / t.fps
    n = len(t.frames)
    inside = np.zeros(n, bool)
    for a, b in spans:
        inside[t.at(a):t.at(b - 1) + 1] = True
    busy = sig.busy & inside[:, None]
    events = _runs(busy.any(axis=1), max(1, int(round(cfg.event_gap_s / dt))))
    pad = int(round(cfg.pad_s / dt))
    lo_n, hi_n = int(round(cfg.min_s / dt)), int(round(cfg.max_s / dt))
    out = []
    for a, b in events:
        # cut very long activity into pieces of at most max_s
        pieces = max(1, int(np.ceil((b - a + 2 * pad) / hi_n)))
        edges = np.linspace(a, b, pieces + 1).round().astype(int)
        for pa, pb in zip(edges[:-1], edges[1:]):
            if pb <= pa:
                continue
            wa, wb = pa - pad, pb + pad
            if wb - wa < lo_n:  # grow short clips symmetrically to the minimum length
                grow = lo_n - (wb - wa)
                wa, wb = wa - grow // 2, wb + (grow - grow // 2)
            wa, wb = max(0, wa), min(n, wb)
            if wb - wa < lo_n:  # at the edge of the tracklet: shift instead of shrinking
                wa, wb = (max(0, wb - lo_n), wb) if wb == n else (wa, min(n, wa + lo_n))
            if wb - wa < lo_n * 0.75:
                continue
            core = busy[pa:pb]
            busy_s = float(core.any(axis=1).sum() * dt)
            scale = float(np.median(sig.scale[wa:wb]))
            if busy_s < cfg.min_busy_s or scale < cfg.min_scale:
                continue
            walk = float(sig.walking[wa:wb].mean())
            ba, bb = t.box[wa], t.box[wb - 1]
            drift = float(np.hypot((bb[0] + bb[2] - ba[0] - ba[2]) / 2, (bb[1] + bb[3] - ba[1] - ba[3]) / 2)
                          / np.median(t.box[wa:wb, 3] - t.box[wa:wb, 1]))
            if walk > cfg.max_walk_share or drift > cfg.max_drift:
                continue
            sides = [s for s in (0, 1) if core[:, s].sum() * dt >= min(cfg.min_busy_s, busy_s)]
            if not sides:
                sides = [int(core.sum(axis=0).argmax())]
            seen = float(np.mean([sig.seen[wa:wb, s].mean() for s in sides]))
            pts = np.concatenate([sig.hand[wa:wb, s][sig.seen[wa:wb, s]] for s in sides])
            if seen < cfg.min_seen or len(pts) < 3:
                continue
            crop = plan_crop(pts, scale, frame_size, cfg)
            if crop is None:
                continue
            extent = max(float(np.linalg.norm(np.ptp(sig.hand[pa:pb, s][sig.seen[pa:pb, s]], axis=0)))
                         if sig.seen[pa:pb, s].sum() > 1 else 0.0 for s in sides) / scale
            c = Candidate(track=t.id, start=int(t.frames[wa]), end=int(t.frames[wb - 1]) + t.step, crop=crop,
                          hands=[("left", "right")[s] for s in sides], busy_s=round(busy_s, 2),
                          peak_speed=round(float(sig.speed[pa:pb][:, sides].max()), 2), extent=round(extent, 2),
                          scale=round(scale, 1), seen=round(seen, 2), walk=round(walk, 2), drift=round(drift, 2))
            c.score = score(c)
            out.append(c)
    return out


def plan_crop(pts: np.ndarray, scale: float, frame_size: tuple[int, int],
              cfg: ZoomConfig | None = None) -> tuple[int, int, int, int] | None:
    """The crop for a clip, from where the busy hand was (n, 2 points in source pixels) and how big the
    person is (torso length in px): the hand path plus room for the object, widened to 16:9.
    None if the path is too large to be a hand's (the person moved, not the hand) or too short to place."""
    cfg = cfg or ZoomConfig()
    if len(pts) < 3:
        return None
    x1, y1 = np.percentile(pts, 2, axis=0) - cfg.margin * scale
    x2, y2 = np.percentile(pts, 98, axis=0) + cfg.margin * scale
    if y2 - y1 > cfg.max_reach_h * scale or x2 - x1 > cfg.max_reach_w * scale:
        return None
    h = max(y2 - y1, cfg.min_side * scale, cfg.min_px)
    w = max(x2 - x1, h)  # never narrower than tall before widening to 16:9
    return media.fit_box((x1 + x2) / 2, (y1 + y2) / 2, w, h, frame_size[0], frame_size[1], cfg.aspect)


def score(c: Candidate) -> float:
    """0-1 priority used to decide what to refine first: sustained hand motion that covers some
    distance, on a person big enough to see, with the hand in view, who stays where they are."""
    sustained = min(c.busy_s / 1.5, 1.0)
    travel = min(c.extent / 1.0, 1.0)
    size = min(c.scale / 110.0, 1.0)
    planted = (1.0 - 0.7 * c.walk) * (1.0 - 0.5 * min(c.drift / 0.5, 1.0))
    return round((0.3 * sustained + 0.25 * travel + 0.3 * size + 0.15 * c.seen) * planted, 4)


def select(cands: list[Candidate], limit: int | None, rating: dict[int, int] | None = None,
           exclude: set[int] | None = None) -> list[int]:
    """Indices of the candidates to refine, best first.

    Without ratings: highest motion score first. With screening ratings (0-3, by candidate index):
    anything rated 0 is out, the rest go by rating and then by motion score; candidates that were
    not screened or whose screening failed count as rating 1. No single person may take more than
    a third of the picks while other people still have good candidates. `exclude` are candidates
    that turned out unusable (no hand could be located in them): the next best take their place.
    """
    def rank(i: int) -> float:
        return (rating.get(i, 1) if rating is not None else 0) + cands[i].score

    pool = [i for i in range(len(cands)) if (rating is None or rating.get(i, 1) > 0) and not (exclude and i in exclude)]
    pool.sort(key=lambda i: -rank(i))
    if limit is None or len(pool) <= limit:
        return pool
    cap = max(2, -(-limit // 3))
    picked, per_track, spare = [], {}, []
    for i in pool:
        if per_track.get(cands[i].track, 0) < cap:
            picked.append(i)
            per_track[cands[i].track] = per_track.get(cands[i].track, 0) + 1
        else:
            spare.append(i)
        if len(picked) == limit:
            return picked
    return picked + spare[:limit - len(picked)]
