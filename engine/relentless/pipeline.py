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
which is what the live dashboard reads.

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

from . import activity, evaluate, interactions, ledger, media, meva, motion, people, render, screen
from . import label as labeler
from .config import CLAUDE_MODEL, DATA, FAL_BUDGET_USD, FAL_MODELS, POSE_WEIGHTS, RUNS, limit_threads

CACHE = DATA / "cache"
STAGES = [("motion", "Motion filter"), ("people", "Person tracking"), ("activity", "Activity filter"),
          ("zoom", "Hand zoom"), ("screen", "Quick screen"), ("upscale", "Cut + upscale"), ("label", "Label"),
          ("multiply", "Multiply")]
LABEL_VERSION = "v1"  # bump when the labeling prompt or schema changes, so cached labels are not reused


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
    multiply_min_quality: int = 6
    previews: int = 24            # person preview videos to write (most active people first)
    overview: bool = True
    cache: bool = True


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
    """Per-frame hand positions for one clip, in coordinates normalised to the crop (0-1), from the pose keypoints."""
    x, y, w, h = c.crop
    frames = np.arange(c.start, c.end)
    side = 0.5 * float(np.median(sig.scale[t.at(c.start):t.at(c.end - 1) + 1]))  # a hand is about half a torso long
    out = {"fps": t.fps, "frames": int(len(frames)), "source": "estimated from body-pose keypoints",
           "box_size": [round(side / w, 4), round(side / h, 4)]}
    for s, name in enumerate(("left", "right")):
        hx = np.interp(frames, t.frames, sig.hand[:, s, 0])
        hy = np.interp(frames, t.frames, sig.hand[:, s, 1])
        seen = np.interp(frames, t.frames, sig.seen[:, s].astype(float)) > 0.5
        out[name] = {"x": [round(float(v), 4) for v in (hx - x) / w], "y": [round(float(v), 4) for v in (hy - y) / h],
                     "seen": [bool(v) for v in seen]}
    return out


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


VAGUE = ("item", "object", "thing", "something", "unclear", "unknown")


def swap_score(label: dict, seconds: float) -> float:
    """How good a clip is for an object swap: a clearly named object that is handled for much of the clip."""
    main = (label.get("main_object") or "").lower()
    if not main or not label.get("swap_candidates"):
        return -1.0
    specific = not any(w in main.split() for w in VAGUE)
    handled = sum(l["end"] - l["start"] for l in label["labels"] if l.get("object") == label["main_object"])
    return label["quality"] + (2.0 if specific else 0.0) + min(handled / max(seconds, 1e-6), 1.0)


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
        "engine": {"pose_model": POSE_WEIGHTS, "upscaler": f"Real-ESRGAN {opts.upscaler} x4 + temporal blend",
                   "labeler": CLAUDE_MODEL, "multiply_model": FAL_MODELS[opts.multiply_model]["endpoint"]},
        "stages": [{"key": k, "title": t, "status": "pending", "progress": 0.0} for k, t in STAGES],
        "funnel": [], "motion": None, "people": [], "candidates": [], "clips": [], "eval": None, "log": [],
        "files": {},
        "_numbers": {"raw_s": round((last - first) / fps, 1)},
    }
    run = Run(out, state)
    numbers = state["_numbers"]

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
            m = motion.scan(src, first, last, progress=run.progress("motion"))
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
                raw = people.track(src, m["spans"], progress=run.progress("people"))
                people.save(cached, raw)
                note = f"{sum(b - a for a, b in m['spans']) / fps / max(raw['seconds'], 1e-6):.1f}x real time"
            tracklets, dropped = people.build_tracklets(raw)
            heads = people.head_circles(raw)
            numbers["tracklets"] = len(tracklets)
            numbers["person_s"] = round(sum(t.seconds for t in tracklets), 1)
            st["detail"] = f"{len(tracklets)} people-tracks, {numbers['person_s']:.0f} s of person-time"
            st["more"] = (f"{raw['samples']} frames through the pose model ({note}); dropped "
                          f"{dropped['short']} appearances as too brief and {dropped['small']} as too small")
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
                skey = signature(file_sig(src), [(cands[i].start, cands[i].end, cands[i].crop) for i in sorted(stills)],
                                 screen.SCREEN_MODEL or CLAUDE_MODEL, screen.VERSION)
                scache = CACHE / "screen" / f"{skey}.json"
                if opts.cache and scache.exists():
                    saved = json.loads(scache.read_text())
                    rated = {int(k): v for k, v in saved["rated"].items()}
                    cost = {**saved["cost"], "usd": 0.0, "cached": True}
                else:
                    rated, cost = screen.screen(stills, purpose=f"screen:{out.name}", progress=run.progress("screen"))
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
            run.say(f"screen: {st['detail']} ({st['more']})")
        else:
            run._stage("screen")["status"] = "skipped"
        chosen = interactions.select(cands, limit, rating)
        rank = {i: n for n, i in enumerate(chosen)}                     # 0 = best
        picked = sorted(chosen, key=lambda i: (cands[i].start, cands[i].track))  # clip ids follow time order
        numbers["refined"] = len(picked)
        refresh()

        # 6a. cut: one pass over the source writes every chosen clip, its context, the previews and the overview
        run.begin("upscale")
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
            media.dump(hand_tracks(t, sigs[t.id], c), cdir / "hands.json")
            plan.append((cid, c, rank[i]))
            state["candidates"][i]["clip"] = cid
            state["clips"].append({
                "id": cid, "track": c.track, "start_s": round((c.start - first) / fps, 2),
                "seconds": round((c.end - c.start) / fps, 2), "frames": [c.start, c.end], "crop": list(c.crop),
                "context_crop": list(ctx), "hands": c.hands, "score": c.score,
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
        for t in tracklets:
            act_s = sum(b - a for a, b in active[t.id]) / fps
            state["people"].append({
                "id": t.id, "start_s": round((int(t.frames[0]) - first) / fps, 2), "seconds": round(t.seconds, 2),
                "height_px": round(t.height), "active_s": round(act_s, 1),
                "active": [[round((a - first) / fps, 2), round((b - first) / fps, 2)] for a, b in active[t.id]],
                "candidates": sum(1 for c in cands if c.track == t.id),
                "preview": f"people/p{t.id:03d}.mp4" if t.id in previewed else None})
        if opts.overview:
            jobs.append(render.Overview(out / "overview.mp4", first, last, size, fps, raw["step"], tracklets, active,
                                        [(cands[i].start, cands[i].end, cands[i].crop) for i in picked]))
        refresh()
        render.run(src, jobs, heads, progress=run.progress("upscale"))
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
        _refine(run, opts, plan, fps, refresh)
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
    return out


def _refine(run: Run, opts: Options, plan: list, fps: float, refresh) -> None:
    state, numbers, out = run.state, run.state["_numbers"], run.dir
    if not plan:
        run.end("upscale", "nothing to refine")
        for k in ("label", "multiply"):
            run._stage(k)["status"] = "skipped"
        return
    from .upscale import Upscaler, compare

    numbers.update(upscaled=0, labeled=0, accepted=0, rejected=0)
    fal_lock, fal_reserved = threading.Lock(), [0.0]
    multiplied = [0]
    label_pool = cf.ThreadPoolExecutor(max_workers=max(1, opts.label_workers), thread_name_prefix="label")
    mult_pool = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="multiply")
    futures: list[cf.Future] = []

    def do_multiply(cid: str) -> None:
        from . import multiply as mult

        clip = run.clip(cid)
        card = json.loads((out / clip["files"]["card"]).read_text())
        secs = min(opts.multiply_seconds, clip["seconds"])
        start = best_window(card, clip["seconds"], secs)
        est = mult.estimate(secs, opts.multiply_model, 1)
        local = out / "clips" / cid / "variants"
        # Variants are kept in a shared cache keyed by what the clip is, so a re-run of the same
        # footage reuses them instead of paying fal again. The run folder gets its own copy.
        vkey = signature(state["source"]["file"], clip["frames"], clip["crop"], opts.upscaler, opts.max_height,
                         opts.multiply_model, start, secs)
        work = CACHE / "variants" / vkey if opts.cache else local
        edits, reserved = [], 0.0
        for new in card.get("swap_candidates", [])[:opts.variants]:
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
            return
        run.begin("multiply")
        run.set_clip(cid, multiply="running")
        run.say(f"multiply {cid}: {card['main_object']} -> {', '.join(e.swap[1] for e in edits)}")
        try:
            summary = mult.multiply(out / clip["files"]["upscaled"], window_card(card, start, secs), edits,
                                    opts.multiply_model, start=start, seconds=secs, crop=None, out_root=local,
                                    verify_variants=True, dry_run=False, force=False, out_dir=work)
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
        key = signature(state["source"]["file"], clip["frames"], clip["crop"], opts.upscaler, CLAUDE_MODEL,
                        opts.label_effort, LABEL_VERSION)
        cached = CACHE / "labels" / f"{key}.json"
        try:
            if opts.cache and cached.exists():
                data = json.loads(cached.read_text())
                lab, info = labeler.ClipLabel(**data["label"]), {**data["info"], "usd": 0.0, "cached": True}
            else:
                hint = "the person's " + (" and ".join(cand.hands)) + (" hands" if len(cand.hands) > 1 else " hand")
                lab, info = labeler.label_clip(cdir / "upscaled.mp4", cdir / "context.mp4", hint=hint,
                                               effort=opts.label_effort, purpose=f"label:{state['name']}:{cid}")
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
                "busy_hands": clip["hands"], "faces_blurred": True,
                "stages": {"pose_model": POSE_WEIGHTS, "upscaler": state["engine"]["upscaler"],
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
            want = (ok and multiplied[0] < opts.multiply
                    and swap_score(clip["label"], clip["seconds"]) >= opts.multiply_min_quality + 2.5)
            if want:
                multiplied[0] += 1
                clip["multiply"] = "queued"
        refresh()
        run.say(f"label {cid}: {'accepted' if ok else 'rejected (' + why + ')'} q{lab.quality} "
                f"${info['usd']:.3f} - {lab.caption}")
        if want:
            futures.append(mult_pool.submit(guard, do_multiply, cid))

    def guard(fn, *args) -> None:
        try:
            fn(*args)
        except Exception as e:
            run.say(f"{fn.__name__} failed: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")

    up = Upscaler(opts.upscaler, max_height=opts.max_height)
    order = sorted(plan, key=lambda p: p[2])  # best first, so good clips show up early
    for n, (cid, cand, _rank) in enumerate(order):
        cdir = out / "clips" / cid
        run.set_clip(cid, status="upscaling")
        info = up.video(cdir / "raw.mp4", cdir / "upscaled.mp4")
        compare(cdir / "raw.mp4", cdir / "upscaled.mp4", cdir / "compare.mp4")
        media.poster(cdir / "upscaled.mp4", cdir / "poster.jpg")
        with run.lock:
            clip = run.clip(cid)
            clip["files"].update(upscaled=f"clips/{cid}/upscaled.mp4", compare=f"clips/{cid}/compare.mp4",
                                 poster=f"clips/{cid}/poster.jpg")
            clip.update(status="upscaled", native_size=info["in_size"], size=info["out_size"],
                        upscale_s=info["seconds"])
            numbers["upscaled"] += 1
            run._stage("upscale")["progress"] = round((n + 1) / len(order), 3)
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
    # Not enough clips cleared the quality bar for multiplying: take the best accepted ones that are left.
    if multiplied[0] < opts.multiply:
        rest = [c for c in state["clips"] if c["status"] == "accepted" and not c.get("multiply")
                and swap_score(c["label"], c["seconds"]) >= 0]
        for c in sorted(rest, key=lambda c: -swap_score(c["label"], c["seconds"]))[:opts.multiply - multiplied[0]]:
            multiplied[0] += 1
            c["multiply"] = "queued"
            futures.append(mult_pool.submit(guard, do_multiply, c["id"]))
    mult_pool.shutdown(wait=True)
    if run._stage("multiply")["status"] == "running":
        run.end("multiply", f"{numbers.get('variants_ok', 0)} of {numbers.get('variants_made', 0)} variants passed the check")
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
                "seconds": v["seconds"], "caption": lab["caption"], "objects": vl.get("objects", []),
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
    a = ap.parse_args(argv)
    if not a.video.exists():
        raise SystemExit(f"no such video: {a.video}")
    limit_threads()
    opts = Options(start=a.start, seconds=a.seconds, max_clips=a.max_clips, upscaler=a.upscaler,
                   min_quality=a.min_quality, label=not a.no_label, screen=not a.no_screen,
                   max_height=720 if a.fast else 1080, label_effort=a.effort, multiply=a.multiply,
                   variants=a.variants, multiply_model=a.multiply_model, cache=not a.no_cache)
    run_pipeline(a.video, opts, out=a.out, name=a.name)


if __name__ == "__main__":
    main()
