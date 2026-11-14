"""Label: Claude watches each upscaled hand clip and writes its label card.

It sees the clip as time-stamped frames plus one wider context frame, and returns what the
hands do (action segments with start/end times, which hand, which object), what is in view,
and how usable the clip is. A clip with no real hand-object contact, or one too unclear to
trust, is marked rejected: the labeler is the last filter of the funnel.

Usage:
  uv run python -m relentless.label upscaled.mp4 [--context context.mp4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import cv2
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
    "- `swap_candidates`: two or three everyday objects of similar size and shape that could replace "
    "`main_object` in this scene with the same grasp staying plausible (for a later video edit). Leave "
    "empty if there is no main object.\n"
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


def sample_frames(clip: Path, width: int = 960, per_second: float = 2.5, lo: int = 6, hi: int = 12
                  ) -> tuple[list[tuple[float, bytes]], float]:
    """Evenly spaced JPEG frames with their timestamps, plus the clip duration."""
    out = []
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


def label_clip(clip: Path, context: Path | None = None, *, hint: str | None = None, effort: str = "medium",
               purpose: str = "label") -> tuple[ClipLabel, dict]:
    """Label one upscaled clip. Returns (label, usage info with the cost in USD)."""
    frames, duration = sample_frames(clip)
    content: list[dict] = []
    ctx = middle_frame(context) if context and context.exists() else None
    if ctx:
        content += [{"type": "text", "text": "CONTEXT: a wider view of the same person at the middle of the clip. "
                                             "The orange box marks the area the clip zooms into."},
                    claude.jpeg_block(ctx)]
    content.append({"type": "text", "text": f"CLIP: {len(frames)} frames in time order, duration {duration:.1f} s."})
    for t, data in frames:
        content += [{"type": "text", "text": f"t = {t:.2f} s"}, claude.jpeg_block(data)]
    tail = INSTRUCTIONS + (f"\n\nMotion tracking flagged this hand as the busy one: {hint}." if hint else "")
    content.append({"type": "text", "text": tail})
    label, info = claude.ask_full(content, ClipLabel, purpose=purpose, effort=effort, system=SYSTEM)
    # keep times inside the clip and in order
    for a in label.labels:
        a.start = round(min(max(a.start, 0.0), duration), 2)
        a.end = round(min(max(a.end, a.start), duration), 2)
    label.quality = int(min(max(label.quality, 0), 10))
    info["frames"] = len(frames)
    info["duration"] = round(duration, 2)
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
    ap.add_argument("--effort", default="medium")
    a = ap.parse_args(argv)
    label, info = label_clip(a.clip, a.context, effort=a.effort)
    ok, why = accepted(label)
    print(json.dumps(label.model_dump(), indent=2, ensure_ascii=False))
    print(f"\n{'ACCEPTED' if ok else 'REJECTED: ' + why}  |  ${info['usd']:.3f}  "
          f"({info['input_tokens']} in / {info['output_tokens']} out tokens, {info['frames']} frames)")


if __name__ == "__main__":
    main()
