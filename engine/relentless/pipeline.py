"""Run the whole engine on one raw clip.

  raw video
    1. motion filter     drop the parts where nothing moves
    2. person tracking   one tracklet per person-appearance, faces located for blurring
    3. activity filter   keep the parts where a person's hands are busy
    4. hand zoom         plan a 2-8 s, 16:9 crop around each burst of hand activity
    5. quick screen      Claude rates contact sheets of the candidates, so only real hand-object work goes on
    6. cut + upscale     cut the chosen clips (faces blurred) and upscale each 4x; the raw crop is kept
    7. label             Claude writes the label card and rejects clips with no real interaction
    8. multiply          optional: verified object-swap variants with labels carried over

Everything lands in data/runs/<name>/ and state.json there is rewritten as the run progresses,
which is what the live dashboard reads. With --live the run also writes previews of what each stage is
looking at into live/ (see live.py), for the job page of the website.

Usage:
  uv run python -m relentless.pipeline VIDEO --start 60 --seconds 90
  uv run python -m relentless.pipeline VIDEO --max-clips 8 --multiply 2
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import shutil
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import activity, evaluate, hands, interactions, ledger, live, media, meva, motion, people, render, screen
from . import label as labeler
from . import wholebody
from .config import CLAUDE_MODEL, DATA, FAL_BUDGET_USD, FAL_MODELS, POSE_WEIGHTS, RUNS, limit_threads, setting

CACHE = DATA / "cache"
STAGES = [("motion", "Motion filter"), ("people", "Person tracking"), ("activity", "Activity filter"),
          ("zoom", "Hand zoom"), ("screen", "Quick screen"), ("upscale", "Cut + upscale"), ("label", "Label"),
          ("multiply", "Multiply")]
LABEL_VERSION = "v2"  # bump when the labeling prompt or schema changes, so cached labels are not reused
LOOK_SHARE = 0.10     # of the "cut + upscale" stage's progress: the second look that locates the hands
CUT_SHARE = 0.15      # cutting the clips; the rest is upscaling them
DONE_BEFORE = LOOK_SHARE + CUT_SHARE
LOOK_ROUNDS = 3       # times a moment with no locatable hand is replaced by the next best before the engine settles


@dataclass
class Options:
    start: float = 0.0
    seconds: float | None = None
    max_clips: int = 12           # refine at most this many hand clips (0 = all of them)
    screen: bool = True           # let Claude rate contact sheets of the candidates before refining
    screen_factor: int = 4        # screen this many times max_clips candidates (best motion score first)
    upscaler: str = "general"
    max_height: int = 1080        # upscaled clips are capped at this height (720 is about twice as fast)
    min_quality: int = 4          # labeler quality (0-10) a clip needs to enter the catalog
    label: bool = True
    label_effort: str = "medium"
    label_workers: int = 4
    multiply: int = 0             # make variants for this many accepted clips
    variants: int = 1             # object swaps per multiplied clip
    multiply_model: str = "gemini"
    multiply_seconds: float = 4.0
    multiply_min_quality: int = 5      # label quality a clip needs to be considered for a swap (Haiku's top mark)
    previews: int = 24            # person preview videos to write (most active people first)
    overview: bool = True
    cache: bool = True
    live: bool = False            # also write the previews a job page shows while the run goes on (live.py)


class Run:
    """Holds the run's state and writes state.json for the dashboard."""

    def __init__(self, out: Path, state: dict):
        self.dir, self.state = out, state
        self.lock = threading.RLock()
        self._saved = 0.0
        self.t0 = time.time()
        self._started: dict[str, float] = {}
        # what this run spends = ledger now minus ledger at the start (covers labeling, checks and fal)
        self._base = {s: ledger.spent(s) for s in ("anthropic", "fal")}
        self._costs_at = 0.0

    def save(self, throttle: float = 0.0) -> None:
        with self.lock:
            now = time.time()
            if throttle and now - self._saved < throttle:
                return
            self.state["elapsed"] = round(now - self.t0, 1)
            if now - self._costs_at > 1.0 or not throttle:
                fal = ledger.spent("fal")
                self.state["costs"] = {"anthropic_usd": round(ledger.spent("anthropic") - self._base["anthropic"], 4),
                                       "fal_usd": round(fal - self._base["fal"], 4),
                                       "fal_total_usd": round(fal, 3), "fal_cap_usd": FAL_BUDGET_USD}
                self._costs_at = now
            media.dump({k: v for k, v in self.state.items() if not k.startswith("_")}, self.dir / "state.json")
            self._saved = now

    def say(self, msg: str) -> None:
        print(msg, flush=True)
        with self.lock:
            self.state["log"].append({"t": round(time.time() - self.t0, 1), "msg": msg})
            del self.state["log"][:-300]
        self.save()

    def _stage(self, key: str) -> dict:
        return next(s for s in self.state["stages"] if s["key"] == key)

    def begin(self, key: str) -> None:
        with self.lock:
            st = self._stage(key)
            if st["status"] == "pending":
                st["status"] = "running"
                self._started[key] = time.time()
        self.save()

    def end(self, key: str, detail: str = "", status: str = "done") -> None:
        with self.lock:
            st = self._stage(key)
            st.update(status=status, seconds=round(time.time() - self._started.get(key, time.time()), 1),
                      progress=1.0)
            if detail:
                st["detail"] = detail
        self.save()

    @contextmanager
    def stage(self, key: str):
        self.begin(key)
        try:
            yield self._stage(key)
        except BaseException:
            self.end(key, status="error")
            raise
        else:
            if self._stage(key)["status"] == "running":
                self.end(key)

    def progress(self, key: str):
        def cb(frac: float) -> None:
            with self.lock:
                self._stage(key)["progress"] = round(float(frac), 3)
            self.save(throttle=0.4)
        return cb

    def clip(self, cid: str) -> dict:
        return next(c for c in self.state["clips"] if c["id"] == cid)

    def set_clip(self, cid: str, **fields) -> None:
        with self.lock:
            self.clip(cid).update(fields)
        self.save()


# ---------- small helpers ----------

def run_name(src: Path, first_s: float, last_s: float) -> str:
    m = re.match(r"(\d{4})-(\d\d)-(\d\d)\.(\d\d)-(\d\d)-\d\d\..*\.(G\d+)$", src.stem)
    base = f"{m.group(6)}-{m.group(2)}{m.group(3)}-{m.group(4)}{m.group(5)}" if m else re.sub(r"[^A-Za-z0-9]+", "-", src.stem)[:40]
    return f"{base}_{int(first_s)}-{int(round(last_s))}s"


def unique_dir(root: Path, name: str) -> Path:
    out, n = root / name, 1
    while out.exists():
        n += 1
        out = root / f"{name}-{n}"
    return out


def source_info(src: Path) -> dict:
    m = re.match(r"(\d{4}-\d\d-\d\d)\.(\d\d-\d\d-\d\d)\..*\.(G\d+)$", src.stem)
    info = {"file": src.name, "path": str(src)}
    if m:
        info.update(dataset="MEVA", camera=m.group(3), recorded=f"{m.group(1)} {m.group(2).replace('-', ':')}",
                    license="CC BY 4.0",
                    attribution="Multiview Extended Video with Activities (MEVA) dataset by Kitware Inc. and IARPA, "
                                "CC BY 4.0, mevadata.org")
    return info


def signature(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_sig(path: Path) -> list:
    st = path.stat()
    return [str(path.resolve()), st.st_size, int(st.st_mtime)]


def rel(run: Run, path: Path) -> str:
    return str(path.relative_to(run.dir))


def hand_tracks(t: people.Tracklet, sig: activity.Signals, c: interactions.Candidate) -> dict:
    """Per-frame hand positions for one clip, in coordinates normalised to the crop (0-1), guessed from the
    wrist and forearm of stage 2's keypoints. Only used where the whole-body model cannot run: these are a
    rough estimate (see hands.py for the real thing)."""
    x, y, w, h = c.crop
    frames = np.arange(c.start, c.end)
    side = 0.5 * float(np.median(sig.scale[t.at(c.start):t.at(c.end - 1) + 1]))  # a hand is about half a torso long
    out = {"fps": t.fps, "frames": int(len(frames)), "source": "estimated from body-pose keypoints",
           "busy": list(c.hands), "box_size": [round(side / w, 4), round(side / h, 4)]}
    for s, name in enumerate(("left", "right")):
        hx = np.interp(frames, t.frames, sig.hand[:, s, 0])
        hy = np.interp(frames, t.frames, sig.hand[:, s, 1])
        seen = np.interp(frames, t.frames, sig.seen[:, s].astype(float)) > 0.5
        out[name] = {"x": [round(float(v), 4) for v in (hx - x) / w], "y": [round(float(v), 4) for v in (hy - y) / h],
                     "seen": [bool(v) for v in seen]}
    return out


def locate_hands(run: "Run", src: Path, cands: list, by_id: dict, limit: int | None, rating: dict | None,
                 size: tuple[int, int], stretch: tuple[int, int] | None, tracklets: list, progress=None
                 ) -> tuple[list[int], dict, dict, dict]:
    """Choose the clips and take a second look at the people in them.

    The whole-body model (hands.py) says where each chosen person's hands are. A moment in which it can
    locate no hand is not a hand clip, whatever the motion filter thought: it is dropped and the next best
    moment takes its place, a few rounds at most. For the ones that stay, the busy hands and the crop are
    decided again from keypoints that are right, in place on the candidates.

    Returns (chosen candidate indices, best first; {index: hands.Track}; {index: hands.ClipHands};
    {tracklet id: hands.Track} for everyone in `stretch`, the seconds the job page replays).
    """
    chosen = interactions.select(cands, limit, rating)
    if not wholebody.available():
        why = "is switched off (WHOLEBODY=off)" if setting("WHOLEBODY", "") == "off" else "cannot run here (onnxruntime is not installed)"
        run.say(f"hands: the whole-body model {why}; hand tracks are estimates from the wrist")
        return chosen, {}, {}, {}
    tracks: dict[int, hands.Track] = {}
    found: dict[int, hands.ClipHands] = {}
    poses: dict[int, hands.Track] = {}
    dropped: set[int] = set()
    low_memory = bool(setting("LOW_MEMORY", ""))
    try:
        for round_ in range(LOOK_ROUNDS + 1):
            new = [i for i in chosen if i not in found]
            requests = [hands.clip_request(("clip", i), by_id[cands[i].track], cands[i].start, cands[i].end) for i in new]
            if stretch and round_ == 0:
                for r in live.stretch_requests(tracklets, stretch):
                    r.key = ("person", r.key)
                    requests.append(r)
            if not requests:
                break
            if low_memory:     # a child process holds the model and exits, giving its memory back
                looked = hands.look_apart(src, requests, CACHE / f"hands-{run.dir.name}.pkl", progress=progress)
            else:
                looked = hands.look(src, requests, progress=progress)
            for key, track in looked.items():
                if key[0] == "person":
                    poses[key[1]] = track
            for i in new:
                c = cands[i]
                track = looked.get(("clip", i))
                if track is None:       # too short a stretch to judge: leave the moment as planned
                    found[i] = hands.ClipHands(list(c.hands), None, {}, 0.0, ["too short for a second look"])
                    continue
                tracks[i] = track
                found[i] = hands.for_clip(track, c.start, c.end, c.hands, size)
                if not found[i].hands:
                    dropped.add(i)
            if not (dropped & set(new)) or round_ == LOOK_ROUNDS:
                break
            chosen = interactions.select(cands, limit, rating, exclude=dropped)
    except Exception as e:      # the second look must not sink a run: without it the engine works as it did before
        run.say(f"hands: the whole-body model failed ({type(e).__name__}: {e}); hand tracks are estimates from the wrist")
        return interactions.select(cands, limit, rating), {}, {}, {}
    chosen = [i for i in interactions.select(cands, limit, rating, exclude=dropped) if i in found]
    for i in chosen:
        if i in tracks:
            cands[i].hands = list(found[i].hands)
            if found[i].crop:
                cands[i].crop = tuple(found[i].crop)
    return chosen, tracks, found, poses


def context_crop(t: people.Tracklet, c: interactions.Candidate, frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """A wider, square-ish crop holding the whole person and the zoom area for the clip's duration."""
    a, b = t.at(c.start), t.at(c.end - 1) + 1
    x1 = min(float(t.box[a:b, 0].min()), c.crop[0])
    y1 = min(float(t.box[a:b, 1].min()), c.crop[1])
    x2 = max(float(t.box[a:b, 2].max()), c.crop[0] + c.crop[2])
    y2 = max(float(t.box[a:b, 3].max()), c.crop[1] + c.crop[3])
    side = max(x2 - x1, y2 - y1) * 1.12
    return media.fit_box((x1 + x2) / 2, (y1 + y2) / 2, side, side, *frame_size)


def window_card(card: dict, start: float, seconds: float) -> dict:
    """The label card restricted to [start, start + seconds) with times shifted to the new zero."""
    out = {k: v for k, v in card.items() if k != "labels"}
    out["labels"] = []
    for l in card.get("labels", []):
        a, b = max(l["start"] - start, 0.0), min(l["end"] - start, seconds)
        if b - a > 0.05:
            out["labels"].append({**l, "start": round(a, 2), "end": round(b, 2)})
    return out


def best_window(card: dict, duration: float, seconds: float) -> float:
    """Start time of the `seconds`-long window that holds the most action on the main object."""
    if duration <= seconds:
        return 0.0
    main = card.get("main_object")
    spans = [(l["start"], l["end"]) for l in card.get("labels", []) if not main or l.get("object") == main] \
        or [(l["start"], l["end"]) for l in card.get("labels", [])]
    best, best_t = -1.0, 0.0
    for t in np.arange(0.0, duration - seconds + 1e-6, 0.25):
        cover = sum(max(0.0, min(b, t + seconds) - max(a, t)) for a, b in spans)
        if cover > best + 1e-9:
            best, best_t = cover, float(t)
    return round(best_t, 2)


VAGUE = ("item", "object", "thing", "something", "unclear", "unknown", "stuff", "material")
# Actions in which the hand does not have the object: nothing to swap in the hand.
NOT_HELD = ("reach", "press", "tap_screen", "type", "push", "pull", "wipe", "search_bag", "other")


def held_share(label: dict, seconds: float) -> float:
    """Share of the clip in which the main object is in a hand (picked up, held, carried, set down ...)."""
    main = label.get("main_object")
    held = sum(l["end"] - l["start"] for l in label.get("labels", [])
               if main and l.get("object") == main and l.get("verb") not in NOT_HELD)
    return min(held / max(seconds, 1e-6), 1.0)


def swap_score(label: dict, seconds: float) -> float:
    """How good a clip is for an object swap: a clearly named object that a hand holds for much of the clip.
    Negative if the clip is no candidate at all (no named object, nothing to replace it with, never in hand)."""
    main = (label.get("main_object") or "").lower()
    if not main or not label.get("swap_candidates") or any(w in main.split() for w in VAGUE):
        return -1.0
    held = held_share(label, seconds)
    if held < 0.2:
        return -1.0
    return label["quality"] + 2.0 + held


def swap_candidate(clip: dict, min_quality: int) -> tuple[bool, str]:
    """May this accepted clip be considered for an object swap? Returns (yes/no, why not). These are the
    checks that cost nothing; a clip that passes is then looked at by the planner (multiply.plan_swap)."""
    lab = clip.get("label") or {}
    if lab.get("quality", 0) < min_quality:
        return False, f"label quality {lab.get('quality', 0)} is below {min_quality}"
    if swap_score(lab, clip["seconds"]) < 0:
        return False, "no clearly named object that the hand holds"
    tracking = clip.get("hand_tracking") or {}
    if tracking.get("model"):       # the hands were located by the whole-body model: the working hand must be in view
        located = max((tracking.get("located", {}).get(h, 0.0) for h in clip.get("hands", [])), default=0.0)
        if located < 0.6:
            return False, "the working hand is not located for most of the clip"
    return True, ""


def tiers(clip: dict) -> list[str]:
    """Which catalog tiers a clip qualifies for (see the market report: Bulk, Labeled+, Studio)."""
    lab = clip.get("label") or {}
    out = ["bulk"]
    if lab.get("quality", 0) >= 6 and len(lab.get("labels", [])) >= 2 and clip["files"].get("hands"):
        out.append("labeled+")
    if any(v.get("passed") for v in clip.get("variants", [])):
        out.append("studio")
    return out


def funnel(state: dict) -> list[dict]:
    rows = _funnel_rows(state["_numbers"])
    kept = state["_numbers"].get("kept") or {}
    # No share on the hand-zoom row: those crops are only planned, and being 16:9 with room for the
    # object they are wider than the person boxes the rows before it count.
    share = {"raw": 1.0, "motion": kept.get("motion"), "people": kept.get("people"), "activity": kept.get("activity"),
             "refined": kept.get("refined"), "accepted": kept.get("accepted")}
    for r in rows:
        r["share"] = share.get(r["key"])
    return rows


def _funnel_rows(n: dict) -> list[dict]:
    rows = [{"key": "raw", "label": "Raw footage", "value": n.get("raw_s"), "unit": "s"}]
    if "motion_s" in n:
        rows.append({"key": "motion", "label": "With motion", "value": n["motion_s"], "unit": "s",
                     "sub": f"{n['motion_s'] / max(n['raw_s'], 1e-9):.0%} of raw"})
    if "tracklets" in n:
        rows.append({"key": "people", "label": "People tracked", "value": n["tracklets"], "unit": "appearances",
                     "sub": f"{n['person_s']:.0f} s of person-time"})
    if "active_s" in n:
        rows.append({"key": "activity", "label": "Hands busy", "value": n["active_s"], "unit": "s",
                     "sub": f"{n['active_s'] / max(n['person_s'], 1e-9):.0%} of person-time"})
    if "candidates" in n:
        rows.append({"key": "zoom", "label": "Hand moments found", "value": n["candidates"], "unit": "clips",
                     "sub": f"{n['candidate_s']:.0f} s"})
    if "screened" in n:
        rows.append({"key": "screen", "label": "Passed the quick screen", "value": n["screen_ok"], "unit": "clips",
                     "sub": f"of {n['screened']} screened"})
    if "refined" in n:
        rows.append({"key": "refined", "label": "Cut and upscaled", "value": n.get("upscaled", 0), "unit": "clips",
                     "sub": f"best {n['refined']} of {n['candidates']}" if n["refined"] < n["candidates"] else "all of them"})
    if "labeled" in n:
        rows.append({"key": "accepted", "label": "Labeled and accepted", "value": n.get("accepted", 0), "unit": "clips",
                     "sub": f"{n.get('rejected', 0)} rejected by the labeler"})
    if n.get("variants_made"):
        rows.append({"key": "variants", "label": "Verified variants", "value": n.get("variants_ok", 0), "unit": "clips",
                     "sub": f"{n['variants_made'] - n.get('variants_ok', 0)} rejected by the checker"})
    return rows


# ---------- the run ----------

def run_pipeline(src: Path, opts: Options, out: Path | None = None, name: str | None = None) -> Path:
    src = src.resolve()
    with media.Reader(src) as rd:
        fps, size, count = rd.fps, (rd.width, rd.height), rd.count
    first = int(round(opts.start * fps))
    last = min(count, first + int(round(opts.seconds * fps))) if opts.seconds else count
    if last - first < fps:
        raise SystemExit(f"window too short: {src.name} has {count / fps:.1f}s, asked for {opts.start}s onwards")
    out = out or unique_dir(RUNS, name or run_name(src, first / fps, last / fps))
    out.mkdir(parents=True, exist_ok=True)
    state = {
        "name": out.name, "status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {**source_info(src), "fps": fps, "size": list(size), "start_s": round(first / fps, 2),
                   "seconds": round((last - first) / fps, 2), "frames": [first, last]},
        "options": asdict(opts),
        "engine": {"pose_model": POSE_WEIGHTS, "hand_model": wholebody.label() if wholebody.available() else None,
                   "upscaler": f"Real-ESRGAN {opts.upscaler} x4 + temporal blend",
                   "labeler": CLAUDE_MODEL, "multiply_model": FAL_MODELS[opts.multiply_model]["endpoint"]},
        "stages": [{"key": k, "title": t, "status": "pending", "progress": 0.0} for k, t in STAGES],
        "funnel": [], "motion": None, "people": [], "candidates": [], "clips": [], "eval": None, "log": [],
        "files": {},
        "_numbers": {"raw_s": round((last - first) / fps, 1)},
    }
    run = Run(out, state)
    numbers = state["_numbers"]

    kit = live.Kit(out / "live", first, last, fps, size) if opts.live else None
    seen: dict = {"motion_spans": [], "tracklets": [], "active": {}, "candidates": []}

    def measure(**found) -> dict:
        """Recompute how much of the raw footage each stage still holds, with whatever exists so far."""
        with run.lock:
            seen.update(found)
            clips = [{"id": c["id"], "track": c["track"], "start": c["frames"][0], "end": c["frames"][1],
                      "crop": c["crop"], "status": c["status"], "label": c["label"]} for c in state["clips"]]
            common = dict(fps=fps, first=first, last=last, clips=clips, **seen)
            kept = evaluate.footage(frame_size=size, **common)
            numbers["kept"] = kept
        return {"kept": kept, **common}

    def refresh() -> None:
        measure()
        with run.lock:
            state["funnel"] = funnel(state)
        run.save()

    refresh()
    run.say(f"run {out.name}: {src.name} {first / fps:.0f}s-{last / fps:.0f}s -> {out}")
    gt_box: dict = {}
    gt_thread = threading.Thread(target=lambda: gt_box.update(gt=meva.load(src)), daemon=True)
    gt_thread.start()

    try:
        # 1. motion filter
        with run.stage("motion") as st:
            m = motion.scan(src, first, last, progress=run.progress("motion"), watch=kit.motion() if kit else None)
            state["motion"] = motion.summary(m)
            numbers["motion_s"] = round(m["kept_frames"] / fps, 1)
            st["detail"] = f"kept {numbers['motion_s']:.0f} s of {numbers['raw_s']:.0f} s"
            st["more"] = f"{len(m['spans'])} span(s) with motion; the rest is never looked at again"
        measure(motion_spans=m["spans"])
        refresh()
        run.say(f"motion: {st['detail']}")
        if not m["spans"]:
            raise RuntimeError("no motion found in this footage, nothing to process")

        # 2. person tracking
        with run.stage("people") as st:
            key = signature(file_sig(src), m["spans"], asdict(people.PeopleConfig()), POSE_WEIGHTS)
            cached = CACHE / f"tracks-{key}.npz"
            if opts.cache and cached.exists():
                raw = people.load(cached)
                note = "from cache"
            else:
                if setting("LOW_MEMORY", ""):   # a child process tracks and exits, giving the pose model's memory back
                    raw = people.track_apart(src, m["spans"], cached, progress=run.progress("people"),
                                             live=kit.spec() if kit else None)
                else:
                    raw = people.track(src, m["spans"], progress=run.progress("people"),
                                       watch=kit.tracking() if kit else None)
                    people.save(cached, raw)
                note = f"{sum(b - a for a, b in m['spans']) / fps / max(raw['seconds'], 1e-6):.1f}x real time"
            tracklets, dropped = people.build_tracklets(raw)
            heads = people.head_circles(raw)
            numbers["tracklets"] = len(tracklets)
            numbers["person_s"] = round(sum(t.seconds for t in tracklets), 1)
            st["detail"] = f"{len(tracklets)} people-tracks, {numbers['person_s']:.0f} s of person-time"
            st["more"] = (f"{raw['samples']} frames through the pose model ({note}); dropped "
                          f"{dropped['short']} appearances as too brief and {dropped['small']} as too small")
            st["stats"] = {"samples": raw["samples"], "model_seconds": round(raw["seconds"], 1),
                           "cached": note == "from cache", "person_s": numbers["person_s"],
                           "imgsz": people.PeopleConfig().imgsz,
                           "dropped_short": dropped["short"], "dropped_small": dropped["small"]}
            # Everyone found, right away: when their hands are busy and what was made of it is filled in below.
            state["people"] = [{
                "id": t.id, "start_s": round((int(t.frames[0]) - first) / fps, 2), "seconds": round(t.seconds, 2),
                "height_px": round(t.height), "active_s": 0.0, "active": [], "candidates": 0, "preview": None}
                for t in tracklets]
            if kit:
                kit.quiet()
        measure(tracklets=tracklets)
        refresh()
        run.say(f"people: {st['detail']} ({st['more']})")

        # 3. activity filter
        with run.stage("activity") as st:
            sigs = {t.id: activity.signals(t) for t in tracklets}
            active = {t.id: activity.active_spans(t, sigs[t.id]) for t in tracklets}
            numbers["active_s"] = round(sum(b - a for sp in active.values() for a, b in sp) / fps, 1)
            st["detail"] = (f"hands busy {numbers['active_s']:.0f} s of {numbers['person_s']:.0f} s "
                            f"({numbers['active_s'] / max(numbers['person_s'], 1e-9):.0%})")
            st["more"] = "wrist speed relative to the body, from the pose keypoints; idle and walking time is dropped"
            st["stats"] = {"active_s": numbers["active_s"], "person_s": numbers["person_s"]}
            for person, t in zip(state["people"], tracklets):
                person.update(active_s=round(sum(b - a for a, b in active[t.id]) / fps, 1),
                              active=[[round((a - first) / fps, 2), round((b - first) / fps, 2)] for a, b in active[t.id]])
        measure(active=active)
        refresh()
        run.say(f"activity: {st['detail']}")

        # 4. hand zoom: plan a crop around every burst of hand activity
        with run.stage("zoom") as st:
            by_id = {t.id: t for t in tracklets}
            cands = [c for t in tracklets for c in interactions.find(t, sigs[t.id], active[t.id], size)]
            numbers["candidates"] = len(cands)
            numbers["candidate_s"] = round(sum(c.end - c.start for c in cands) / fps, 1)
            state["candidates"] = [{
                "track": c.track, "start_s": round((c.start - first) / fps, 2),
                "seconds": round((c.end - c.start) / fps, 2), "crop": list(c.crop), "hands": c.hands,
                "score": c.score, "rating": None, "what": None, "clip": None} for c in cands]
            st["detail"] = f"{len(cands)} hand moments, {numbers['candidate_s']:.0f} s of clips"
            st["more"] = "each is a 2-8 s, 16:9 crop around a burst of hand activity by one person"
            st["stats"] = {"candidates": len(cands), "candidate_s": numbers["candidate_s"]}
            for person, t in zip(state["people"], tracklets):
                person["candidates"] = sum(1 for c in cands if c.track == t.id)
        measure(candidates=cands)
        refresh()
        run.say(f"zoom: {st['detail']}")

        # 5. quick screen: Claude rates contact sheets, so the expensive stages only see real hand-object work
        limit = opts.max_clips or None
        rating = None
        if opts.screen and opts.label and cands:
            with run.stage("screen") as st:
                by_score = sorted(range(len(cands)), key=lambda i: -cands[i].score)
                pool = by_score if limit is None else by_score[:limit * opts.screen_factor]
                still_jobs = {i: render.Stills(cands[i].start, cands[i].end, cands[i].crop) for i in pool}
                render.run(src, list(still_jobs.values()), heads)
                stills = {i: j.frames for i, j in still_jobs.items() if len(j.frames) == j.n}
                if kit:
                    kit.screen_sheet(stills)

                def rated_so_far(sheet: dict) -> None:      # each sheet shows up as soon as Claude has rated it
                    with run.lock:
                        for i, r in sheet.items():
                            state["candidates"][i].update(rating=r["rating"], what=r["what"])
                    run.save(throttle=0.3)

                skey = signature(file_sig(src), [(cands[i].start, cands[i].end, cands[i].crop) for i in sorted(stills)],
                                 screen.SCREEN_MODEL or CLAUDE_MODEL, screen.VERSION)
                scache = CACHE / "screen" / f"{skey}.json"
                if opts.cache and scache.exists():
                    saved = json.loads(scache.read_text())
                    rated = {int(k): v for k, v in saved["rated"].items()}
                    cost = {**saved["cost"], "usd": 0.0, "cached": True}
                else:
                    rated, cost = screen.screen(stills, purpose=f"screen:{out.name}", progress=run.progress("screen"),
                                                on_rated=rated_so_far)
                    if not cost["failed"]:
                        media.dump({"rated": rated, "cost": cost}, scache)
                rating = {i: r["rating"] for i, r in rated.items()}
                for i, r in rated.items():
                    state["candidates"][i].update(rating=r["rating"], what=r["what"])
                numbers["screened"] = len(stills)
                numbers["screen_ok"] = sum(1 for r in rated.values() if r["rating"] >= 2)
                st["detail"] = (f"{numbers['screen_ok']} of {len(stills)} show real hand-object work" if rated
                                else "screening unavailable, picking by motion score instead")
                st["more"] = (f"Claude rated contact sheets of the {len(stills)} most active candidates: "
                              + ("reused from an earlier identical run" if cost.get("cached")
                                 else f"{cost['calls']} calls, ${cost['usd']:.3f}")
                              + (f"; {cost['failed']} call(s) failed" if cost["failed"] else ""))
                st["stats"] = {"screened": len(stills), "passed": numbers["screen_ok"], "calls": cost.get("calls", 0),
                               "usd": cost.get("usd", 0.0), "cached": bool(cost.get("cached")),
                               "failed": cost.get("failed", 0)}
            run.say(f"screen: {st['detail']} ({st['more']})")
        else:
            run._stage("screen")["status"] = "skipped"
        # 6a. a second look: the whole-body model locates the hands of the people in the chosen moments
        #     (and of everyone in the stretch the job page replays). Moments with no locatable hand are replaced.
        run.begin("upscale")
        stage_progress = run.progress("upscale")
        stretch = kit.stretch(tracklets, [(c.start, c.end) for c in cands]) if kit else None
        looked_at = time.time()
        first_pick = set(interactions.select(cands, limit, rating))
        chosen, hand_tracks_of, hands_of, poses = locate_hands(
            run, src, cands, by_id, limit, rating, size, stretch, tracklets,
            progress=lambda frac: stage_progress(LOOK_SHARE * frac))
        unlocated = sorted(i for i in first_pick if i not in chosen)
        for i, found in hands_of.items():      # what the second look made of each moment, kept or not
            state["candidates"][i].update(located=bool(found.hands), crop=list(cands[i].crop), hands=list(cands[i].hands))
        numbers["unlocated"] = sum(1 for found in hands_of.values() if not found.hands)
        hand_model = wholebody.label() if hand_tracks_of else None
        if hand_tracks_of:
            run.say(f"hands: located in {len(hand_tracks_of) - numbers['unlocated']} of {len(hand_tracks_of)} moments looked at"
                    + (f"; {numbers['unlocated']} dropped, no hand could be located" if numbers["unlocated"] else "")
                    + f" ({hand_model}, {time.time() - looked_at:.0f} s)")
        rank = {i: n for n, i in enumerate(chosen)}                     # 0 = best
        picked = sorted(chosen, key=lambda i: (cands[i].start, cands[i].track))  # clip ids follow time order
        numbers["refined"] = len(picked)
        refresh()

        # 6b. cut: one pass over the source writes every chosen clip, its context, the previews and the overview
        jobs: list[render.Job] = []
        plan = []
        for n, i in enumerate(picked):
            c = cands[i]
            cid = f"c{n:03d}"
            cdir = out / "clips" / cid
            t = by_id[c.track]
            ctx = context_crop(t, c, size)
            jobs.append(render.FixedCrop(cdir / "raw.mp4", c.start, c.end, c.crop, fps))
            jobs.append(render.FixedCrop(cdir / "context.mp4", c.start, c.end, ctx, fps, outline=c.crop))
            found = hands_of.get(i)
            if i in hand_tracks_of:
                media.dump(hands.tracks_json(hand_tracks_of[i], c.start, c.end, c.crop, c.hands, hand_model),
                           cdir / "hands.json")
            else:
                media.dump(hand_tracks(t, sigs[t.id], c), cdir / "hands.json")
            plan.append((cid, c, rank[i]))
            state["candidates"][i]["clip"] = cid
            state["clips"].append({
                "id": cid, "track": c.track, "start_s": round((c.start - first) / fps, 2),
                "seconds": round((c.end - c.start) / fps, 2), "frames": [c.start, c.end], "crop": list(c.crop),
                "context_crop": list(ctx), "hands": c.hands, "score": c.score,
                "hand_tracking": ({"model": hand_model, "located": found.located, "readable": found.readable}
                                  if i in hand_tracks_of else {"model": None}),
                "screen": {"rating": state["candidates"][i]["rating"], "what": state["candidates"][i]["what"]},
                "motion": {"busy_s": c.busy_s, "peak_speed": c.peak_speed, "extent": c.extent, "torso_px": c.scale},
                "status": "planned", "files": {"hands": f"clips/{cid}/hands.json"}, "label": None,
                "variants": []})
        ranked = sorted(tracklets, key=lambda t: -sum(b - a for a, b in active[t.id]))
        previewed = {t.id for t in ranked[:opts.previews] if active[t.id]}
        for t in ranked[:opts.previews]:
            if t.id not in previewed:
                continue
            a, b = max(active[t.id], key=lambda s: s[1] - s[0])
            b = min(b, a + int(20 * fps))
            jobs.append(render.FollowCrop(out / "people" / f"p{t.id:03d}.mp4", t, a, b, size))
        for person, t in zip(state["people"], tracklets):
            person["preview"] = f"people/p{t.id:03d}.mp4" if t.id in previewed else None
        if opts.overview:
            # whoever the whole-body model looked at is drawn from it: the clips' people, and the replayed stretch
            careful = {tid: [track] for tid, track in poses.items()}
            for i, track in hand_tracks_of.items():
                careful.setdefault(cands[i].track, []).append(track)
            jobs.append(render.Overview(out / "overview.mp4", first, last, size, fps, raw["step"], tracklets, active,
                                        [(cands[i].start, cands[i].end, cands[i].crop) for i in picked], poses=careful))
        if kit:
            jobs += kit.cut_jobs(tracklets, active, [
                {"k": k, "track": c.track, "start": c.start, "end": c.end, "crop": c.crop, "hands": c.hands,
                 "score": c.score, "clip": state["candidates"][k]["clip"]} for k, c in enumerate(cands)], poses)
        with run.lock:
            st = run._stage("upscale")
            st["stats"] = {"hand_model": hand_model, "looked_at": len(hands_of), "unlocated": numbers["unlocated"],
                           "replaced": len(unlocated), "look_seconds": round(time.time() - looked_at, 1)}
            if hand_model:
                st["more"] = (f"before cutting, {hand_model} located the hands in each moment"
                              + (f"; {numbers['unlocated']} with no locatable hand were left out" if numbers["unlocated"] else ""))
        refresh()
        render.run(src, jobs, heads, progress=lambda frac: stage_progress(LOOK_SHARE + CUT_SHARE * frac))
        if kit:
            kit.after_cut()
        with run.lock:
            if opts.overview:
                state["files"]["overview"] = "overview.mp4"
            for cid, _c, _rank in plan:
                run.clip(cid)["files"].update(raw=f"clips/{cid}/raw.mp4", context=f"clips/{cid}/context.mp4")
                run.clip(cid)["status"] = "cut"
            for person in state["people"]:
                person["ready"] = True
        refresh()
        run.say(f"cut: {len(plan)} clips" + (f" (best of {len(cands)})" if len(plan) < len(cands) else ""))
        if gt_thread.is_alive():
            gt_thread.join()
        gt = gt_box.get("gt")

        def evaluate_now() -> None:
            found = measure()
            if gt is not None:
                with run.lock:
                    state["eval"] = evaluate.evaluate(gt, **found)
            refresh()

        evaluate_now()

        # 5-7. upscale on the GPU, label and multiply in worker threads as clips become ready
        clip_tracks = {state["candidates"][i]["clip"]: track for i, track in hand_tracks_of.items()
                       if state["candidates"][i].get("clip")}
        _refine(run, opts, plan, fps, refresh, clip_tracks)
        evaluate_now()

        with run.lock:
            for c in state["clips"]:
                if c["status"] == "accepted":
                    c["tiers"] = tiers(c)
            catalog = build_catalog(run)
            media.dump(catalog, out / "catalog.json")
            state["files"]["catalog"] = "catalog.json"
            state["status"] = "done"
            state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        refresh()
        run.say(f"done in {time.time() - run.t0:.0f}s: {numbers.get('accepted', 0)} clip(s) accepted, "
                f"{numbers.get('variants_ok', 0)} variant(s); Claude ${state['costs']['anthropic_usd']:.3f} "
                f"({CLAUDE_MODEL}), fal ${state['costs']['fal_usd']:.2f} -> {out}")
    except BaseException as e:
        with run.lock:
            state["status"] = "error"
            state["error"] = f"{type(e).__name__}: {e}"
            for s in state["stages"]:
                if s["status"] == "running":
                    s["status"] = "error"
        run.say(f"FAILED: {type(e).__name__}: {e}")
        raise
    finally:
        if kit:
            kit.quiet()
    return out


def _refine(run: Run, opts: Options, plan: list, fps: float, refresh, clip_tracks: dict | None = None) -> None:
    """`clip_tracks` are the hand tracks of the clips ({clip id: hands.Track}), where the hands were located:
    the variant check measures against them."""
    state, numbers, out = run.state, run.state["_numbers"], run.dir
    clip_tracks = clip_tracks or {}
    if not plan:
        run.end("upscale", "nothing to refine")
        for k in ("label", "multiply"):
            run._stage(k)["status"] = "skipped"
        return
    from .upscale import Upscaler, compare

    numbers.update(upscaled=0, labeled=0, accepted=0, rejected=0)
    fal_lock, fal_reserved = threading.Lock(), [0.0]
    gpu_lock = threading.Lock()
    multiplied = [0]
    planned = [0]         # planner calls so far; at most four per variant wanted, whoever asks
    label_pool = cf.ThreadPoolExecutor(max_workers=max(1, opts.label_workers), thread_name_prefix="label")
    mult_pool = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="multiply")
    futures: list[cf.Future] = []

    def consider(cid: str) -> bool:
        """Is this clip fit for an object swap? If so, queue it. The cheap checks first, then the planner
        (one small Claude call, cached with the clip), and only then fal. Returns whether it was queued."""
        from . import multiply as mult

        clip = run.clip(cid)
        with run.lock:
            if multiplied[0] >= opts.multiply or clip.get("multiply"):
                return False
            ok, why = swap_candidate(clip, opts.multiply_min_quality)
            if not ok:
                clip["multiply_note"] = why
                return False
            if planned[0] >= 4 * opts.multiply:     # the planner has been asked often enough for this run
                return False
            planned[0] += 1
            multiplied[0] += 1                      # a place is held while the planner looks
            clip["multiply"] = "planning"
        run.begin("multiply")
        card = json.loads((out / clip["files"]["card"]).read_text())
        secs = min(opts.multiply_seconds, clip["seconds"])
        start = best_window(card, clip["seconds"], secs)
        pkey = signature(state["source"]["file"], clip["frames"], clip["crop"], opts.upscaler, opts.max_height, start,
                         secs, card.get("main_object"), card.get("caption"), mult.SWAP_MODEL or CLAUDE_MODEL,
                         mult.SWAP_VERSION, opts.variants)
        pcache = CACHE / "swaps" / f"{pkey}.json"
        try:
            if opts.cache and pcache.exists():
                swap = mult.SwapPlan(**json.loads(pcache.read_text())["plan"])
            else:
                hands_json = json.loads((out / clip["files"]["hands"]).read_text())
                swap, info = mult.plan_swap(out / clip["files"]["upscaled"], window_card(card, start, secs), start=start,
                                            seconds=secs, n=max(2, opts.variants), hands=hands_json,
                                            purpose=f"swap:{state['name']}:{cid}")
                media.dump({"plan": swap.model_dump(), "info": info}, pcache)
            ok, why = mult.fit(swap, card.get("main_object"))
        except Exception as e:      # a planner that cannot be asked is a no: nothing is spent on a guess
            ok, why, swap = False, f"the swap could not be planned ({type(e).__name__}: {e})", None
        names = mult.good_replacements(swap, card.get("main_object"))[:opts.variants] if ok else []
        with run.lock:
            if not names:
                multiplied[0] -= 1
                clip["multiply"], clip["multiply_note"] = "skipped", why
            else:
                clip["multiply"] = "queued"
                clip["swap"] = {"from": card["main_object"], "to": names, "seen": swap.object_seen, "why": swap.why}
        refresh()
        if not names:
            run.say(f"multiply {cid}: not swapped: {why}")
            return False
        futures.append(mult_pool.submit(guard, do_multiply, cid, names, start, secs))
        return True

    def do_multiply(cid: str, names: list[str], start: float, secs: float) -> None:
        from . import multiply as mult

        clip = run.clip(cid)
        card = json.loads((out / clip["files"]["card"]).read_text())
        est = mult.estimate(secs, opts.multiply_model, 1)
        local = out / "clips" / cid / "variants"
        # Variants are kept in a shared cache keyed by what the clip is, so a re-run of the same
        # footage reuses them instead of paying fal again. The run folder gets its own copy.
        vkey = signature(state["source"]["file"], clip["frames"], clip["crop"], opts.upscaler, opts.max_height,
                         opts.multiply_model, start, secs)
        work = CACHE / "variants" / vkey if opts.cache else local
        edits, reserved = [], 0.0
        for new in names:
            e = mult.presets((card["main_object"], new))["swap"]
            e.name = "swap-" + re.sub(r"[^a-z0-9]+", "-", new.lower()).strip("-")
            if not (work / f"{e.name}-{opts.multiply_model}.mp4").exists():
                with fal_lock:  # reserve budget before spending, so parallel jobs cannot overshoot the cap
                    if ledger.spent("fal") + fal_reserved[0] + est > FAL_BUDGET_USD:
                        run.say(f"multiply {cid}: fal budget cap ${FAL_BUDGET_USD:.2f} reached, skipping '{new}'")
                        continue
                    fal_reserved[0] += est
                    reserved += est
            edits.append(e)
        if not edits:
            run.set_clip(cid, multiply="stopped")
            return
        run.set_clip(cid, multiply="running")
        run.say(f"multiply {cid}: {card['main_object']} -> {', '.join(e.swap[1] for e in edits)}")
        track = clip_tracks.get(cid)

        def hand_paths(video: Path) -> dict:
            """Where the hands are in a video of this window, against the clip's hand track."""
            with gpu_lock:      # one model call at a time: two variants are checked in parallel threads
                return hands.same_path(video, track, tuple(clip["crop"]), clip["frames"][0], start, secs)

        try:
            summary = mult.multiply(out / clip["files"]["upscaled"], window_card(card, start, secs), edits,
                                    opts.multiply_model, start=start, seconds=secs, crop=None, out_root=local,
                                    verify_variants=True, dry_run=False, force=False, out_dir=work,
                                    hand_paths=hand_paths if track is not None and not setting("LOW_MEMORY", "") else None)
        except SystemExit as e:  # multiply's own budget stop
            run.say(f"multiply {cid}: {e}")
            run.set_clip(cid, multiply="stopped")
            return
        finally:
            with fal_lock:
                fal_reserved[0] -= reserved
        if work != local:
            shutil.copytree(work, local, dirs_exist_ok=True)
        vdir = local
        with run.lock:
            for rec in summary["variants"]:
                clip["variants"].append({
                    "id": f"{cid}-{rec['variant']}", "edit": rec["labels"].get("scene", {}).get("object_swap"),
                    "prompt": rec["prompt"], "model": rec["model"], "passed": bool(rec.get("passed")),
                    "verify": rec.get("verify"), "verify_model": rec.get("verify_model"),
                    "hand_paths": rec.get("hand_paths"),
                    "cost_usd": rec.get("cost_usd"), "labels": rec["labels"],
                    "start_s": start, "seconds": secs,
                    "files": {"video": rel(run, vdir / rec["file"]), "original": rel(run, vdir / "original.mp4"),
                              "card": rel(run, vdir / (Path(rec["file"]).stem + ".json"))}})
                numbers["variants_made"] = numbers.get("variants_made", 0) + 1
                numbers["variants_ok"] = numbers.get("variants_ok", 0) + int(bool(rec.get("passed")))
            clip["multiply"] = "done"
        refresh()

    def do_label(cid: str, cand: interactions.Candidate) -> None:
        run.begin("label")
        run.set_clip(cid, status="labeling")
        clip = run.clip(cid)
        cdir = out / "clips" / cid
        hands_json = json.loads((cdir / "hands.json").read_text())
        # a label written from frames that carried the L / R marks is not the label of the same clip without them
        key = signature(state["source"]["file"], clip["frames"], clip["crop"], opts.upscaler, CLAUDE_MODEL,
                        opts.label_effort, LABEL_VERSION, labeler.marked(hands_json))
        cached = CACHE / "labels" / f"{key}.json"
        try:
            if opts.cache and cached.exists():
                data = json.loads(cached.read_text())
                lab, info = labeler.ClipLabel(**data["label"]), {**data["info"], "usd": 0.0, "cached": True}
            else:
                hint = (None if not cand.hands else "both of the person's hands" if len(cand.hands) > 1
                        else f"the person's {cand.hands[0]} hand")
                lab, info = labeler.label_clip(cdir / "upscaled.mp4", cdir / "context.mp4", hint=hint,
                                               effort=opts.label_effort, purpose=f"label:{state['name']}:{cid}",
                                               hands=hands_json)
                media.dump({"label": lab.model_dump(), "info": info}, cached)
        except Exception as e:  # one failed call must not sink the run
            run.say(f"label {cid} failed: {type(e).__name__}: {e}")
            run.set_clip(cid, status="error", reject_reason=f"labeling failed: {e}")
            return
        ok, why = labeler.accepted(lab, opts.min_quality)
        src = state["source"]
        card = {
            "id": f"{state['name']}-{cid}", **lab.model_dump(),
            "source": f"{src.get('dataset', 'video')} {src['file']} frames {clip['frames'][0]}-{clip['frames'][1]} "
                      f"crop {','.join(str(v) for v in clip['crop'])}",
            "accepted": ok, "reject_reason": None if ok else why,
            "label_info": {"model": info["model"], "usd": info["usd"], "frames_seen": info["frames"],
                           "cached": bool(info.get("cached"))},
            "lineage": {
                "source": {k: src[k] for k in ("file", "dataset", "camera", "recorded", "license", "attribution")
                           if k in src},
                "source_fps": fps, "source_size": src["size"], "frames": clip["frames"],
                "seconds": [round(clip["frames"][0] / fps, 2), round(clip["frames"][1] / fps, 2)],
                "crop_xywh": clip["crop"], "context_crop_xywh": clip["context_crop"], "person_track": clip["track"],
                "busy_hands": clip["hands"], "hand_tracks": clip.get("hand_tracking"), "faces_blurred": True,
                "stages": {"pose_model": POSE_WEIGHTS, "hand_model": (clip.get("hand_tracking") or {}).get("model"),
                           "upscaler": state["engine"]["upscaler"],
                           "upscaled_from": clip.get("native_size"), "labeler": info["model"]},
            },
        }
        media.dump(card, cdir / "card.json")
        with run.lock:
            clip["files"]["card"] = f"clips/{cid}/card.json"
            clip["label"] = lab.model_dump()
            clip["label_usd"] = info["usd"]
            clip["status"] = "accepted" if ok else "rejected"
            clip["reject_reason"] = None if ok else why
            numbers["labeled"] += 1
            numbers["accepted" if ok else "rejected"] += 1
            # a clip that is plainly a good candidate goes to the planner right away, while the others upscale
            want = (ok and multiplied[0] < opts.multiply
                    and swap_score(clip["label"], clip["seconds"]) >= opts.multiply_min_quality + 2.5)
        refresh()
        run.say(f"label {cid}: {'accepted' if ok else 'rejected (' + why + ')'} q{lab.quality} "
                f"${info['usd']:.3f} - {lab.caption}")
        if want:
            consider(cid)

    def guard(fn, *args) -> None:
        try:
            fn(*args)
        except Exception as e:
            run.say(f"{fn.__name__} failed: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")

    up = Upscaler(opts.upscaler, max_height=opts.max_height)
    order = sorted(plan, key=lambda p: p[2])  # best first, so good clips show up early
    for n, (cid, cand, _rank) in enumerate(order):
        cdir = out / "clips" / cid
        run.set_clip(cid, status="upscaling", progress=0.0)

        def upscaled_so_far(frac: float, cid: str = cid, n: int = n) -> None:
            with run.lock:
                run.clip(cid)["progress"] = round(float(frac), 2)
                run._stage("upscale")["progress"] = round(DONE_BEFORE + (1 - DONE_BEFORE) * (n + frac) / len(order), 3)
            run.save(throttle=0.5)

        info = up.video(cdir / "raw.mp4", cdir / "upscaled.mp4", progress=upscaled_so_far)
        compare(cdir / "raw.mp4", cdir / "upscaled.mp4", cdir / "compare.mp4")
        media.poster(cdir / "upscaled.mp4", cdir / "poster.jpg")
        with run.lock:
            clip = run.clip(cid)
            clip["files"].update(upscaled=f"clips/{cid}/upscaled.mp4", compare=f"clips/{cid}/compare.mp4",
                                 poster=f"clips/{cid}/poster.jpg")
            clip.update(status="upscaled", native_size=info["in_size"], size=info["out_size"],
                        upscale_s=info["seconds"], progress=1.0)
            numbers["upscaled"] += 1
            run._stage("upscale")["progress"] = round(DONE_BEFORE + (1 - DONE_BEFORE) * (n + 1) / len(order), 3)
        refresh()
        if opts.label:
            futures.append(label_pool.submit(guard, do_label, cid, cand))
    run.end("upscale", f"{numbers['upscaled']} clips cut and upscaled 4×")
    run.say(f"upscale: {numbers['upscaled']} clips done")

    label_pool.shutdown(wait=True)
    if opts.label:
        run.end("label", f"{numbers['accepted']} accepted, {numbers['rejected']} rejected")
    else:
        run._stage("label")["status"] = "skipped"
    # Places left: the remaining accepted clips, best candidate first. Every one still has to pass the cheap
    # checks and the planner; a clip that is not fit for a swap is passed over, never swapped anyway.
    if opts.multiply and multiplied[0] < opts.multiply:
        rest = [c for c in state["clips"] if c["status"] == "accepted" and not c.get("multiply")]
        for c in sorted(rest, key=lambda c: -swap_score(c["label"], c["seconds"])):
            if multiplied[0] >= opts.multiply or planned[0] >= 4 * opts.multiply:      # consider() keeps the bound
                break
            consider(c["id"])
    mult_pool.shutdown(wait=True)
    if opts.multiply:
        made = numbers.get("variants_made", 0)
        run.begin("multiply")
        run.end("multiply", f"{numbers.get('variants_ok', 0)} of {made} variants passed the check" if made
                else "no clip was fit for an object swap")
    else:
        run._stage("multiply")["status"] = "skipped"
    for f in futures:
        f.result()
    refresh()


def build_catalog(run: Run) -> list[dict]:
    """Everything this run produced that can be sold: accepted originals and their verified variants."""
    state, items = run.state, []
    src = state["source"]
    for c in state["clips"]:
        if c["status"] != "accepted":
            continue
        lab = c["label"]
        base = {"run": state["name"], "seconds": c["seconds"], "fps": src["fps"], "scene": lab["scene"],
                "labeled_by": state["engine"]["labeler"], "attribution": src.get("attribution"),
                "license": src.get("license")}
        items.append({
            **base, "id": f"{state['name']}-{c['id']}", "kind": "original", "caption": lab["caption"],
            "objects": lab["objects"], "main_object": lab["main_object"], "labels": lab["labels"],
            "quality": lab["quality"], "tiers": c.get("tiers", ["bulk"]), "resolution": c.get("size"),
            "native_resolution": c.get("native_size"),
            "files": {"video": c["files"]["upscaled"], "original": c["files"]["raw"], "context": c["files"]["context"],
                      "compare": c["files"]["compare"], "hands": c["files"]["hands"], "card": c["files"]["card"],
                      **({"poster": c["files"]["poster"]} if c["files"].get("poster") else {})},
            "lineage": {"source": src["file"], "frames": c["frames"], "crop_xywh": c["crop"], "person_track": c["track"]},
        })
        for v in c.get("variants", []):
            if not v.get("passed"):
                continue
            vl = v["labels"]
            items.append({
                **base, "id": f"{state['name']}-{v['id']}", "kind": "variant", "parent": f"{state['name']}-{c['id']}",
                "seconds": v["seconds"], "caption": vl.get("caption", lab["caption"]),
                "objects": vl.get("objects", []), "main_object": vl.get("main_object"),
                "labels": vl.get("labels", []), "scene": vl.get("scene", lab["scene"]), "edit": v["edit"],
                "verify": v["verify"], "tiers": ["studio"], "files": v["files"],
                "lineage": {"source": src["file"], "frames": c["frames"], "crop_xywh": c["crop"],
                            "person_track": c["track"], "generated_by": v["model"], "prompt": v["prompt"]},
            })
    return items


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the video")
    ap.add_argument("--seconds", type=float, help="how much to process (default: to the end)")
    ap.add_argument("--max-clips", type=int, default=12, help="refine at most this many hand clips (0 = all)")
    ap.add_argument("--name", help="run folder name under data/runs (default: derived from the video)")
    ap.add_argument("--out", type=Path, help="write the run into exactly this folder (used by the dashboard)")
    ap.add_argument("--upscaler", choices=["general", "x4plus"], default="general")
    ap.add_argument("--min-quality", type=int, default=4)
    ap.add_argument("--effort", default="medium", help="Claude effort for labeling: low, medium, high")
    ap.add_argument("--no-label", action="store_true", help="stop after upscaling (no API calls)")
    ap.add_argument("--no-screen", action="store_true", help="skip the quick screen and pick clips by motion score")
    ap.add_argument("--fast", action="store_true", help="cap upscaled clips at 720p (about twice as fast)")
    ap.add_argument("--multiply", type=int, default=0, help="make variants for this many accepted clips (uses fal credits)")
    ap.add_argument("--variants", type=int, default=1, help="object swaps per multiplied clip")
    ap.add_argument("--multiply-model", choices=sorted(FAL_MODELS), default="gemini")
    ap.add_argument("--no-cache", action="store_true", help="redo tracking and labeling even if cached")
    ap.add_argument("--live", action="store_true",
                    help="also write previews of what each stage is looking at into live/ (for the website's job page)")
    a = ap.parse_args(argv)
    if not a.video.exists():
        raise SystemExit(f"no such video: {a.video}")
    limit_threads()
    opts = Options(start=a.start, seconds=a.seconds, max_clips=a.max_clips, upscaler=a.upscaler,
                   min_quality=a.min_quality, label=not a.no_label, screen=not a.no_screen,
                   max_height=720 if a.fast else 1080, label_effort=a.effort, multiply=a.multiply,
                   variants=a.variants, multiply_model=a.multiply_model, cache=not a.no_cache, live=a.live)
    run_pipeline(a.video, opts, out=a.out, name=a.name)


if __name__ == "__main__":
    main()
