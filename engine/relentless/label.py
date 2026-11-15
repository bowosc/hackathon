"""Label: Claude watches each upscaled hand clip and writes its label card.

It sees the clip as time-stamped frames plus one wider context frame, and returns what the
hands do (action segments with start/end times, which hand, which object), what is in view,
and how usable the clip is. A clip with no real hand-object contact, or one too unclear to
trust, is marked rejected: the labeler is the last filter of the funnel.

Which hand is the left one is not something to guess from a blurry crop: the engine has located
both hands (hands.py), so the frames the labeler sees carry a thin box marked L or R around each
located hand. The marks are only on the labeler's copy of the frames, never in the clip.

Usage:
  uv run python -m relentless.label upscaled.mp4 [--context context.mp4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from . import claude, media

Verb = Literal["reach", "pick_up", "put_down", "hold", "carry", "hand_over", "receive", "open", "close",
               "unwrap", "pour", "stir", "press", "tap_screen", "type", "write", "eat", "drink", "push",
               "pull", "wipe", "search_bag", "pay", "inspect", "adjust", "other"]

SYSTEM = (
    "You label short surveillance-camera clips for a catalog of training data for robot manipulation models. "
    "Each clip is a zoomed, upscaled crop around one person's hands, given to you as still frames in time "
    "order with their timestamps. Faces are blurred on purpose.\n\n"
    "Buyers train on these labels, so accuracy matters more than coverage: describe only what the frames "
    "show. If you cannot tell what an object is, say what it looks like (\"small white packet\") instead "
    "of naming a product. If a hand is hidden or it is unclear whether it touches anything, say so and "
    "lower the quality score instead of filling the gap with a plausible story. Timing matters: give each "
    "action the start and end time you can actually support from the frames."
)

INSTRUCTIONS = (
    "Label this clip.\n\n"
    "- `interaction` is true only if a hand visibly touches, holds or moves a physical object (including "
    "phones, bags, cups, money, doors, screens). Gesturing, resting, crossing arms or touching one's own "
    "face or clothes is not an interaction.\n"
    "- `labels`: one entry per distinct action, in time order, covering the clip without large overlaps. "
    "`hand` is the acting person's own left or right hand; use \"both\" for two-handed actions and "
    "\"unknown\" if you cannot tell. Times are seconds from the start of the clip.\n"
    "- Use one consistent name for each object everywhere (in `objects`, `main_object` and every label), "
    "for example always \"paper cup\", never \"paper cup\" in one place and \"coffee\" in another.\n"
    "- `quality` rates how well the clip shows the hand-object interaction, 0 to 10: 0-3 hands or object "
    "mostly hidden or too blurry to follow; 4-6 the action is recognisable but details are soft or partly "
    "occluded; 7-10 hands, object and contact are clearly visible throughout.\n"
    "- `swap_candidates`: two or three objects that could stand in for `main_object` in a later video edit. "
    "Each must be the same kind of thing (a drink container for a drink container, a snack for a snack, a "
    "paper item for a paper item), about the same size, held in the same grip, and something a person in "
    "this place would plausibly be handling in this way. Leave it empty if there is no main object, if you "
    "cannot tell what the object is, if the hand does not hold it (it only touches, presses or reaches for "
    "it), or if it is not a loose item: something fixed in place or part of a building, vehicle, machine or "
    "piece of furniture (a rail, pole, handle, door, counter, screen), or something the person wears.\n"
    "- `issues`: one short sentence on the main limitation of this clip, or \"none\"."
)


class Action(BaseModel):
    verb: Verb = Field(description="Normalised action class.")
    action: str = Field(description="Short phrase in present tense, e.g. 'picks up bag of chips from basket'.")
    object: str | None = Field(description="The object acted on, using the same name as in `objects`; null if none.")
    hand: Literal["left", "right", "both", "unknown"]
    start: float = Field(description="Seconds from clip start.")
    end: float = Field(description="Seconds from clip start.")


class Scene(BaseModel):
    setting: str = Field(description="Where this is, e.g. 'cafe counter', 'hallway snack table'.")
    lighting: str = Field(description="E.g. 'bright daylight from windows', 'even indoor fluorescent'.")
    viewpoint: str = Field(description="Camera angle relative to the person, e.g. 'high angle from the front-left'.")


class ClipLabel(BaseModel):
    hands_visible: bool = Field(description="At least one hand is visible for most of the clip.")
    interaction: bool = Field(description="A hand visibly touches, holds or moves a physical object.")
    caption: str = Field(description="One sentence describing what happens in the clip.")
    objects: list[str] = Field(description="Objects that are handled or directly relevant, most important first.")
    main_object: str | None = Field(description="The object most involved in the interaction, or null.")
    labels: list[Action]
    scene: Scene
    quality: int = Field(description="0-10 as defined in the instructions.")
    swap_candidates: list[str]
    issues: str


MARK = {"left": ("L", (240, 150, 60)), "right": ("R", (40, 110, 240))}     # letter and colour (BGR): blue, orange


def marked(hands: dict | None) -> bool:
    """Are these hand tracks good enough to draw on the labeler's frames? Only the located ones are: a
    track guessed from the wrist would mark the wrong place with the same confidence."""
    return bool(hands) and all(isinstance((hands.get(side) or {}).get("w"), list) for side in MARK) \
        and any(any(hands[side]["seen"]) for side in MARK)


def mark_hands(frame: np.ndarray, hands: dict, index: int) -> int:
    """Draw a thin box and a letter around each hand located in clip frame `index` (in place).
    Returns how many hands were marked."""
    h, w = frame.shape[:2]
    n = 0
    for side, (letter, colour) in MARK.items():
        track = hands[side]
        if index >= len(track["x"]) or not track["seen"][index]:
            continue
        cx, cy, bw, bh = track["x"][index] * w, track["y"][index] * h, track["w"][index] * w, track["h"][index] * h
        if not (0 <= cx <= w and 0 <= cy <= h):       # the hand is outside this clip's frame: nothing to mark
            continue
        # a little air around the hand, so the line never runs over the fingers or what they hold
        x1, y1, x2, y2 = int(cx - 0.65 * bw), int(cy - 0.65 * bh), int(cx + 0.65 * bw), int(cy + 0.65 * bh)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = y1 - 6 if y1 - th - 10 > 0 else y2 + th + 8          # above the box, or under it at the top edge
        tx = min(max(x1, 2), w - tw - 6)
        cv2.rectangle(frame, (tx - 3, ty - th - 4), (tx + tw + 3, ty + 4), colour, -1)
        cv2.putText(frame, letter, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        n += 1
    return n


def sample_frames(clip: Path, width: int = 960, per_second: float = 2.5, lo: int = 6, hi: int = 12,
                  hands: dict | None = None) -> tuple[list[tuple[float, bytes]], float]:
    """Evenly spaced JPEG frames with their timestamps, plus the clip duration. With `hands` (a clip's
    hands.json from the whole-body model), each frame carries the L / R marks of the hands located in it."""
    out = []
    mark = marked(hands)
    with media.Reader(clip) as rd:
        duration = rd.count / rd.fps
        n = int(min(hi, max(lo, round(duration * per_second))))
        size = (width, media.even(width * rd.height / rd.width))
        for i in range(n):
            f = min(rd.count - 1, int(round((i + 0.5) * rd.count / n)))
            frame = rd.read(f)
            if frame is None:
                break
            if frame.shape[1] != width:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if frame.shape[1] > width else cv2.INTER_CUBIC)
            if mark:
                mark_hands(frame, hands, min(hands["frames"] - 1, int(round(f / rd.fps * hands["fps"]))))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            out.append((f / rd.fps, buf.tobytes()))
    return out, duration


def middle_frame(clip: Path, width: int = 640) -> bytes | None:
    with media.Reader(clip) as rd:
        frame = rd.read(rd.count // 2)
    if frame is None:
        return None
    frame = cv2.resize(frame, (width, media.even(width * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


MARKS = (
    "\n\nThe thin boxes marked L and R were drawn on these frames by hand tracking: they show where the acting "
    "person's own left hand (L, blue) and right hand (R, orange) are. They are annotations, not part of the "
    "scene: never mention them. Use them for `hand` in every label. A hand without a box in a frame could not "
    "be located there (hidden or out of view). Hands of other people carry no box."
)


def label_clip(clip: Path, context: Path | None = None, *, hint: str | None = None, effort: str = "medium",
               purpose: str = "label", hands: dict | None = None) -> tuple[ClipLabel, dict]:
    """Label one upscaled clip. Returns (label, usage info with the cost in USD).

    `hands` is the clip's hands.json. If its hands were located by the whole-body model, the frames the
    labeler sees are marked with them, and `hint` names the hand that moves most."""
    frames, duration = sample_frames(clip, hands=hands)
    content: list[dict] = []
    ctx = middle_frame(context) if context and context.exists() else None
    if ctx:
        content += [{"type": "text", "text": "CONTEXT: a wider view of the same person at the middle of the clip. "
                                             "The orange box marks the area the clip zooms into."},
                    claude.jpeg_block(ctx)]
    content.append({"type": "text", "text": f"CLIP: {len(frames)} frames in time order, duration {duration:.1f} s."})
    for t, data in frames:
        content += [{"type": "text", "text": f"t = {t:.2f} s"}, claude.jpeg_block(data)]
    if marked(hands):
        tail = INSTRUCTIONS + MARKS + (f" Hand tracking found the most motion in {hint}."
                                       + ("" if hint.startswith("both") else " The other hand may still be the one "
                                                                             "that matters.") if hint else "")
    else:
        tail = INSTRUCTIONS + (f"\n\nMotion tracking flagged this hand as the busy one: {hint}. It can be wrong "
                               f"about left and right: say \"unknown\" unless the frames show it." if hint else "")
    content.append({"type": "text", "text": tail})
    label, info = claude.ask_full(content, ClipLabel, purpose=purpose, effort=effort, system=SYSTEM)
    # keep times inside the clip and in order
    for a in label.labels:
        a.start = round(min(max(a.start, 0.0), duration), 2)
        a.end = round(min(max(a.end, a.start), duration), 2)
    label.quality = int(min(max(label.quality, 0), 10))
    info["frames"] = len(frames)
    info["duration"] = round(duration, 2)
    info["hands_marked"] = marked(hands)
    return label, info


def accepted(label: ClipLabel, min_quality: int = 4) -> tuple[bool, str]:
    """Does this clip go in the catalog? Returns (yes/no, reason)."""
    if not label.hands_visible:
        return False, "hands not visible"
    if not label.interaction:
        return False, "no hand-object interaction"
    if not label.labels:
        return False, "no action could be labeled"
    if label.quality < min_quality:
        return False, f"quality {label.quality} below {min_quality}"
    return True, "ok"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--context", type=Path)
    ap.add_argument("--hands", type=Path, help="the clip's hands.json: located hands are marked on the labeler's frames")
    ap.add_argument("--effort", default="medium")
    a = ap.parse_args(argv)
    label, info = label_clip(a.clip, a.context, effort=a.effort,
                             hands=json.loads(a.hands.read_text()) if a.hands else None)
    ok, why = accepted(label)
    print(json.dumps(label.model_dump(), indent=2, ensure_ascii=False))
    print(f"\n{'ACCEPTED' if ok else 'REJECTED: ' + why}  |  ${info['usd']:.3f}  "
          f"({info['input_tokens']} in / {info['output_tokens']} out tokens, {info['frames']} frames)")


if __name__ == "__main__":
    main()
