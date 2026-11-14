"""Measure a run: how much footage each stage keeps, and how it does against MEVA's human-made labels.

`footage` gives the share of the raw footage each stage still holds. It is counted in
pixel-seconds, the one unit that stays meaningful from a full frame down to a hand crop.

`evaluate` takes each ground-truth grasp event (pick up / put down / hand over) inside the
processed window and checks how far through the funnel it survived.
"""

from __future__ import annotations

import numpy as np

from . import meva
from .interactions import Candidate
from .people import Tracklet

# Ground-truth activity -> the labeler verbs that count as agreeing with it.
VERB_MATCH = {
    "person_picks_up_object": {"pick_up", "reach", "receive", "search_bag"},
    "person_puts_down_object": {"put_down", "hand_over"},
    "person_transfers_object": {"hand_over", "receive", "pay", "pick_up", "put_down"},
}


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _match(gt: meva.GroundTruth, act: meva.Activity, tracklets: list[Tracklet]) -> Tracklet | None:
    """The tracklet that follows the same person as the ground-truth event (best box overlap at its midpoint)."""
    mid = (act.start + act.end) // 2
    best, hit = 0.2, None
    for p in act.people:
        box = gt.box(p, mid)
        if box is None:
            continue
        for t in tracklets:
            if t.frames[0] <= mid <= t.frames[-1] + t.step:
                v = _iou(box, t.box[t.at(mid)])
                if v > best:
                    best, hit = v, t
    return hit


def footage(*, fps: float, first: int, last: int, frame_size: tuple[int, int],
            motion_spans: list[tuple[int, int]], tracklets: list[Tracklet],
            active: dict[int, list[tuple[int, int]]], candidates: list[Candidate], clips: list[dict]) -> dict:
    """Share of the raw window (in pixel-seconds) still in play after each stage."""
    raw_px_s = frame_size[0] * frame_size[1] * (last - first) / fps

    def box_px_s(t: Tracklet, spans: list[tuple[int, int]] | None) -> float:
        area = (t.box[:, 2] - t.box[:, 0]) * (t.box[:, 3] - t.box[:, 1])
        if spans is None:
            return float(area.sum()) * t.step / fps
        keep = np.zeros(len(t.frames), bool)
        for a, b in spans:
            keep |= (t.frames >= a) & (t.frames < b)
        return float(area[keep].sum()) * t.step / fps

    def crops(items) -> float:
        return sum(c["crop"][2] * c["crop"][3] * (c["end"] - c["start"]) / fps for c in items) / raw_px_s

    return {
        "motion": sum(b - a for a, b in motion_spans) / max(1, last - first),
        "people": sum(box_px_s(t, None) for t in tracklets) / raw_px_s,
        "activity": sum(box_px_s(t, active.get(t.id, [])) for t in tracklets) / raw_px_s,
        "zoom": sum(c.crop[2] * c.crop[3] * (c.end - c.start) / fps for c in candidates) / raw_px_s,
        "refined": crops(clips),
        "accepted": crops([c for c in clips if c["status"] == "accepted"]),
    }


def evaluate(gt: meva.GroundTruth, *, fps: float, first: int, last: int, kept: dict,
             motion_spans: list[tuple[int, int]], tracklets: list[Tracklet],
             active: dict[int, list[tuple[int, int]]], candidates: list[Candidate],
             clips: list[dict]) -> dict:
    """`clips` are the refined clips: dicts with track, start, end (source frames), status and label (or None).
    `kept` is the result of `footage` for the same run."""
    events = [a for a in gt.activities if a.name in meva.GRASP and first <= (a.start + a.end) // 2 < last]
    found = dict.fromkeys(kept, 0)
    agree = labeled_events = 0
    per_event = []
    for act in events:
        mid = (act.start + act.end) // 2
        row = {"activity": act.name.replace("person_", ""), "start_s": round((act.start - first) / fps, 2),
               "end_s": round((act.end - first) / fps, 2), "reached": "raw"}
        if any(a <= mid < b for a, b in motion_spans):
            found["motion"] += 1
            row["reached"] = "motion"
            t = _match(gt, act, tracklets)
            if t is not None:
                found["people"] += 1
                row["reached"], row["track"] = "people", t.id
                if any(a <= mid < b for a, b in active.get(t.id, [])):
                    found["activity"] += 1
                    row["reached"] = "activity"
                    if any(c.track == t.id and c.start <= mid < c.end for c in candidates):
                        found["zoom"] += 1
                        row["reached"] = "zoom"
                        mine = [c for c in clips if c["track"] == t.id and c["start"] <= mid < c["end"]]
                        if mine:
                            found["refined"] += 1
                            row["reached"] = "refined"
                            ok = [c for c in mine if c["status"] == "accepted"]
                            if ok:
                                found["accepted"] += 1
                                row["reached"], row["clip"] = "accepted", ok[0]["id"]
                                labeled_events += 1
                                want = VERB_MATCH.get(act.name, set())
                                t0, t1 = (act.start - ok[0]["start"]) / fps, (act.end - ok[0]["start"]) / fps
                                verbs = {l["verb"] for l in (ok[0].get("label") or {}).get("labels", [])
                                         if l["end"] >= t0 - 1.0 and l["start"] <= t1 + 1.0}
                                row["verbs"] = sorted(verbs)
                                row["agrees"] = bool(verbs & want)
                                agree += row["agrees"]
        per_event.append(row)
    n = len(events)
    order = ["motion", "people", "activity", "zoom", "refined", "accepted"]
    return {
        "events": n,
        "stages": [{"stage": k, "found": found[k], "recall": (found[k] / n) if n else None,
                    "footage_kept": kept[k]} for k in order],
        "labeled_events": labeled_events,
        "label_agreement": (agree / labeled_events) if labeled_events else None,
        "per_event": per_event,
        "note": "Events are MEVA pick-up, put-down and hand-over annotations whose midpoint lies in the processed "
                "window. Footage kept is in pixel-seconds relative to the raw window. Hand-zoom crops are 16:9 and "
                "leave room for the object, so they cover more pixels than the person boxes of the row above.",
    }
