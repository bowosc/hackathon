"""Screen: a quick look at every hand-zoom candidate before any expensive work is spent on it.

Motion alone cannot tell handling an object from scratching one's nose. So each candidate is
reduced to four small frames, six candidates are tiled into one contact sheet, and Claude rates
every row 0-3 in a single cheap call (a fraction of a cent per candidate). Only well-rated
candidates go on to be cut, upscaled and fully labeled.
"""

from __future__ import annotations

import concurrent.futures as cf

import cv2
import numpy as np
from pydantic import BaseModel, Field

from . import claude
from .config import setting

VERSION = "v1"      # bump when the prompt or sheet layout changes, so cached ratings are not reused
ROWS_PER_SHEET = 6
TILE = (320, 180)
SCREEN_MODEL = setting("SCREEN_MODEL", "") or None   # default: the same model as the labeler

PROMPT = (
    "Each numbered row is one short clip from a ceiling security camera, shown as {n} frames from left to "
    "right in time order. Every clip was cut around a person's moving hand. Faces are blurred on purpose.\n\n"
    "For every row, rate how useful it is as footage of a hand working on a physical object:\n"
    "- 3: a hand clearly picks up, puts down, opens, passes or otherwise manipulates an object, and both the "
    "hand and the object are easy to see\n"
    "- 2: a hand clearly holds or uses an object (phone, cup, bag, money, food), or manipulates one that is "
    "small or partly hidden\n"
    "- 1: an object may be involved but you cannot really tell\n"
    "- 0: no object in hand (gesturing, walking, arms resting, touching face or clothes), or the hands are "
    "not visible\n\n"
    "Return exactly one entry per row number. `what` is a few words on what the hand does."
)


class Row(BaseModel):
    row: int = Field(description="The row number printed at the left of the row.")
    handles_object: bool
    rating: int = Field(description="0-3 as defined.")
    what: str


class Sheet(BaseModel):
    rows: list[Row]


def sheet_image(rows: list[tuple[int, list[np.ndarray]]]) -> bytes:
    """Tile rows of frames into one JPEG, each row numbered in a margin at the left."""
    margin = 70
    lines = []
    for number, frames in rows:
        strip = np.hstack([cv2.resize(f, TILE, interpolation=cv2.INTER_AREA) for f in frames])
        label = np.full((TILE[1], margin, 3), 255, np.uint8)
        text = str(number)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(label, text, ((margin - tw) // 2, (TILE[1] + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3,
                    cv2.LINE_AA)
        line = np.hstack([label, strip])
        lines.append(cv2.copyMakeBorder(line, 0, 4, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    return cv2.imencode(".jpg", np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def screen(stills: dict[int, list[np.ndarray]], purpose: str = "screen", workers: int = 4, progress=None,
           on_rated=None) -> tuple[dict[int, dict], dict]:
    """Rate candidates by their stills. Returns ({candidate index: {rating, what}}, cost info).
    `on_rated`, if given, gets each sheet's ratings as soon as that sheet comes back.

    A candidate missing from the result (failed call, row skipped by the model) is simply unrated;
    the caller falls back to the motion score for it.
    """
    keys = sorted(stills)
    groups = [keys[i:i + ROWS_PER_SHEET] for i in range(0, len(keys), ROWS_PER_SHEET)]
    out: dict[int, dict] = {}
    cost = {"usd": 0.0, "calls": 0, "failed": 0}

    def one(group: list[int]):
        n = len(stills[group[0]])
        rows = [(i + 1, stills[k]) for i, k in enumerate(group)]
        content = [claude.jpeg_block(sheet_image(rows)), {"type": "text", "text": PROMPT.format(n=n)}]
        verdict, info = claude.ask_full(content, Sheet, purpose=purpose, effort="low", model=SCREEN_MODEL)
        return {group[r.row - 1]: {"rating": int(min(max(r.rating, 0), 3)) if r.handles_object or r.rating <= 1 else 1,
                                   "what": r.what}
                for r in verdict.rows if 1 <= r.row <= len(group)}, info

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, g) for g in groups]
        for done, f in enumerate(cf.as_completed(futures)):
            try:
                rated, info = f.result()
                out.update(rated)
                if on_rated:
                    on_rated(rated)
                cost["usd"] += info["usd"]
                cost["calls"] += 1
            except Exception as e:  # leave these candidates unrated
                cost["failed"] += 1
                print(f"screening call failed: {type(e).__name__}: {e}", flush=True)
            if progress:
                progress((done + 1) / len(groups))
    cost["usd"] = round(cost["usd"], 4)
    return out, cost
