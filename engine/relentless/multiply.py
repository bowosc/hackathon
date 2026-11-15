"""Multiply: turn one labeled clip into N labeled variants.

  original clip + labels
      -> a plan: is this clip fit for an object swap at all, and what would be a sensible replacement
      -> fal.ai video edit (one call per variant; motion, timing and framing kept)
      -> labels carried over from the original + the attributes the edit changed
      -> Claude compares variant frames to the original frames; failures are discarded

A swap only makes sense for a loose item a hand actually holds, that can be told apart in the frames, and
only by something of the same kind that belongs in the scene. A refuse collector's grab handle turned into
a bag of crisps is a paid-for clip nobody can use; `plan_swap` is there so that edit is never asked for.

Every paid call is checked against a budget first and written to data/spend-ledger.jsonl.

Usage:
  uv run python -m relentless.multiply CLIP --card card.json --variants evening,store
  uv run python -m relentless.multiply CLIP --variants swap --swap "chip bag=apple" --dry-run
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import re

import cv2
import fal_client
from pydantic import BaseModel, Field

from . import claude, ledger, media
from .config import CLAUDE_MODEL, FAL_BUDGET_USD, FAL_MODELS, MULTIPLY_OUT, secret, setting

KEEP = ("Keep the camera angle, framing, timing, the people's bodies, clothing and hands, the background, every "
        "other object and every movement exactly the same. Faces are blurred on purpose: keep every blurred face "
        "blurred.")
SWAP_VERSION = "v1"     # bump when the planning prompt or its rules change, so cached plans are not reused
SWAP_MODEL = setting("SWAP_MODEL", "") or None      # default: the same model as the labeler


@dataclass
class Edit:
    name: str
    prompt: str
    attributes: dict = field(default_factory=dict)  # labels the edit itself guarantees
    swap: tuple[str, str] | None = None             # (old object, new object)


def presets(swap: tuple[str, str] | None) -> dict[str, Edit]:
    p = {
        "evening": Edit("evening", f"Change the lighting to evening: dim, warm indoor lights and dark windows. {KEEP}",
                        {"lighting": "evening, warm artificial"}),
        "store": Edit("store", f"Turn the background into a small convenience store with shelves of packaged snacks "
                               f"and drinks. {KEEP}", {"setting": "convenience store"}),
        "clothing": Edit("clothing", f"Change the person's clothes to a red hoodie. {KEEP}",
                         {"clothing": "red hoodie"}),
        "fluorescent": Edit("fluorescent", f"Change the lighting to cold, bright fluorescent office light. {KEEP}",
                            {"lighting": "cold fluorescent"}),
    }
    if swap:
        old, new = swap
        p["swap"] = Edit("swap", f"Replace only the {old} in the person's hand with {with_article(new)} of about the "
                                 f"same size, held in the same grip. Do not change the hand itself: its position, "
                                 f"fingers, skin, glove and sleeve stay as they are. {KEEP}",
                         {"object_swap": f"{old} -> {new}"}, swap)
    return p


def with_article(name: str) -> str:
    """ "apple" -> "an apple", "paper cup" -> "a paper cup"; a name that brings its own article keeps it."""
    first = name.split()[0].lower() if name.split() else ""
    if first in ("a", "an", "the", "some"):
        return name
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name


# ---------- the plan ----------

class Replacement(BaseModel):
    name: str = Field(description="The replacement object as a short plain noun phrase, e.g. 'ceramic mug'.")
    same_kind: bool = Field(description="It is the same kind of thing as the object, used for the same purpose.")
    fits_scene: int = Field(description="0-3: how natural it is to see a person handle this here, in this way. "
                                        "3 = entirely ordinary, 0 = out of place.")
    same_grip: int = Field(description="0-3: how well the hand pose in the frames would hold it "
                                       "(size, shape, weight). 3 = the same grip works unchanged.")


class SwapPlan(BaseModel):
    object_seen: str = Field(description="What the tracked hand is handling, in a few plain words, from the "
                                         "frames alone. 'cannot tell' if you cannot make it out.")
    is_main_object: bool = Field(description="That is the object the label card names as its main object.")
    clearly_visible: bool = Field(description="You can tell what kind of thing the object is (a cup, a snack bag, "
                                              "a bottle) and see its outline against the hand and the background in "
                                              "most frames. False if it is so small, blurred or hidden that an editor "
                                              "could not tell where it starts and ends.")
    loose_item: bool = Field(description="A separate item a person lifts and carries in one hand. Not something "
                                         "fixed in place or part of a building, vehicle, machine or piece of "
                                         "furniture (rail, pole, handle, door, counter, shelf, screen, keypad), "
                                         "not something the person wears (glove, sleeve, bag strap), not a body part.")
    in_hand: bool = Field(description="The hand holds, lifts, carries or sets the object down in these frames. "
                                      "False if it only touches, presses, pushes, leans on or reaches for it.")
    swappable: bool = Field(description="All four hold, and a replacement of the same kind exists that would look "
                                        "ordinary in this scene.")
    why: str = Field(description="One short sentence: why the clip is fit for a swap, or the main reason it is not.")
    replacements: list[Replacement] = Field(description="Up to the number asked for, best first. Empty if the clip "
                                                        "is not fit.")


PLAN = (
    "These frames are one short clip from a security camera, in time order, zoomed on one person's hands. Faces "
    "are blurred on purpose.{marks}\n\n"
    "The clip is a candidate for an object swap: a video edit that replaces the one object a hand is handling with "
    "another object and leaves everything else as it is. The result is sold as training data for robot "
    "manipulation, so the swap has to make sense to anyone who watches it. A swap is paid for, and a wrong one "
    "produces nonsense (a refuse collector's grab handle turned into a bag of crisps, a card terminal turned into "
    "a sandwich). When in doubt, the clip is not fit.\n\n"
    "The clip's label card says:\n{card}\n\n"
    "Look at the frames and answer about the object the hand is handling. Judge from what you see: the label card "
    "can be wrong, and saying so is the most useful answer you can give.\n"
    "- `replacements`: at most {n}, best first, and only if the clip is fit. Each must be the same kind of thing as "
    "the object (a drink container for a drink container, a snack for a snack, a paper item for a paper item, a "
    "hand tool for a hand tool), about the same size and weight, held in the same grip, and something this "
    "person would plausibly be handling in this place in this way. Name plain everyday objects, not brands."
)
HAND_MARKS = (" The thin boxes marked L and R are annotations from hand tracking: the acting person's own left and "
              "right hand. They are not part of the scene.")


def window_frames(clip: Path, start: float, seconds: float, n: int = 6, width: int = 960,
                  hands: dict | None = None) -> list[bytes]:
    """`n` evenly spaced JPEG frames of [start, start + seconds) of a clip, the way the labeler sees frames."""
    from . import label as labeler      # here: the labeler does not need this module

    out = []
    mark = labeler.marked(hands)
    with media.Reader(clip) as rd:
        first = int(round(start * rd.fps))
        count = max(1, min(rd.count - first, int(round(seconds * rd.fps))))
        size = (width, media.even(width * rd.height / rd.width))
        for i in range(n):
            f = min(rd.count - 1, first + int((i + 0.5) * count / n))
            frame = rd.read(f)
            if frame is None:
                break
            if frame.shape[1] != width:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if frame.shape[1] > width else cv2.INTER_CUBIC)
            if mark:
                labeler.mark_hands(frame, hands, min(hands["frames"] - 1, int(round(f / rd.fps * hands["fps"]))))
            out.append(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())
    return out


def plan_swap(clip: Path, card: dict, *, start: float, seconds: float, n: int = 2, hands: dict | None = None,
              purpose: str = "swap") -> tuple[SwapPlan, dict]:
    """Ask whether [start, start + seconds) of a clip is fit for an object swap, and with what.
    `card` is the clip's label card restricted to that window. Returns (plan, usage info with the cost)."""
    from . import label as labeler

    frames = window_frames(clip, start, seconds, hands=hands)
    said = {"caption": card.get("caption"), "main_object": card.get("main_object"), "objects": card.get("objects"),
            "actions": [{k: l.get(k) for k in ("verb", "action", "object", "hand")} for l in card.get("labels", [])],
            "quality": card.get("quality"), "issues": card.get("issues")}
    content: list[dict] = [{"type": "text", "text": f"CLIP: {len(frames)} frames in time order."}]
    content += [claude.jpeg_block(data) for data in frames]
    content.append({"type": "text", "text": PLAN.format(marks=HAND_MARKS if labeler.marked(hands) else "",
                                                         card=json.dumps(said, ensure_ascii=False), n=max(1, n))})
    plan, info = claude.ask_full(content, SwapPlan, purpose=purpose, effort="medium", model=SWAP_MODEL)
    for r in plan.replacements:
        r.fits_scene, r.same_grip = int(min(max(r.fits_scene, 0), 3)), int(min(max(r.same_grip, 0), 3))
    return plan, info


def fit(plan: SwapPlan, main_object: str | None = None) -> tuple[bool, str]:
    """Is the plan a go? Returns (yes/no, the reason it is not). The rules are checked here one by one, not
    taken from the model's own summary: a plan that says "swappable" about a fixed handle is still a no."""
    seen = (plan.object_seen or "the object").strip()
    if "cannot tell" in seen.lower() or not seen:
        return False, "the object in the hand cannot be made out"
    if not plan.loose_item:
        return False, f"{seen}: not a loose item a hand carries"
    if not plan.in_hand:
        return False, f"{seen}: the hand does not hold it"
    if not plan.clearly_visible:
        return False, f"{seen}: too small, blurred or hidden to replace cleanly"
    if not plan.is_main_object:
        return False, f"the hand handles {seen}, not the labeled {main_object or 'main object'}"
    if not plan.swappable:
        return False, plan.why or "not fit for a swap"
    if not good_replacements(plan, main_object):
        return False, "no replacement of the same kind fits the scene"
    return True, ""


def good_replacements(plan: SwapPlan, main_object: str | None = None) -> list[str]:
    """The replacements worth paying for, best first: same kind, at home in the scene, same grip."""
    old = (main_object or "").strip().lower()
    out = []
    for r in sorted(plan.replacements, key=lambda r: -(r.fits_scene + r.same_grip)):
        name = re.sub(r"^(a|an|the|some)\s+", "", r.name.strip(), flags=re.I)
        if name and name.lower() != old and r.same_kind and r.fits_scene >= 2 and r.same_grip >= 2 and name not in out:
            out.append(name)
    return out


# ---------- labels ----------

COLOUR = ("(?:red|maroon|orange|yellow|green|blue|navy|purple|pink|brown|beige|tan|cream|white|black|grey|gray|"
          "silver|gold|clear|transparent|dark|light|pale|bright)")
# one colour or several ("red and blue", "pink-and-white", "dark red") in front of an object's name
COLOURS = rf"(?:\b{COLOUR}(?:(?:\s+and\s+|-and-|[-\s]+){COLOUR})*[-\s]+)"


def carry_labels(card: dict, edit: Edit) -> dict:
    """Variant labels = original labels with the edited attributes applied.

    For a swap the old object's name becomes the new one in the caption, the object list and each label's
    action and object, by whole words: "cup" is not found in "cupboard", and "snack bags" on the table are
    not the "snack bag" in the hand (the edit leaves them alone). The article in front is redone for the new
    word ("a paper cup" -> "an apple"), a colour in front goes with the old object ("a red and blue snack
    packet" -> "a glass"), and a capital at the start of a sentence is kept."""
    out = copy.deepcopy(card)
    if edit.swap and edit.swap[0].strip():
        old, new = edit.swap[0].strip(), edit.swap[1].strip()
        word = re.compile(rf"(?:\b(an?)\s+)?({COLOURS})?(?<![\w-]){re.escape(old)}(?![\w-])", re.I)

        def put(m: re.Match) -> str:
            text = new
            if m.group(1):
                text = ("an " if new[:1].lower() in "aeiou" else "a ") + new
            return text[:1].upper() + text[1:] if m.group(0)[:1].isupper() else text

        def sub(v):
            if isinstance(v, str):
                return word.sub(put, v)
            if isinstance(v, list):
                return [sub(x) for x in v]
            return v

        for key in ("objects", "main_object", "caption"):
            if key in out:
                out[key] = sub(out[key])
        # in a label only what names the object: its verb ("drink") and hand are not the object's name
        out["labels"] = [{**l, **{k: sub(l[k]) for k in ("action", "object") if k in l}} if isinstance(l, dict) else l
                         for l in out.get("labels", [])]
        out.pop("swap_candidates", None)      # they were candidates for the original object
    out.setdefault("scene", {}).update(edit.attributes)
    return out


class Verdict(BaseModel):
    edit_applied: bool = Field(description="The requested change is clearly visible in the variant.")
    motion_preserved: bool = Field(description="Same people doing the same movements at the same moments.")
    hands_intact: bool = Field(description="Hands look anatomically plausible and still hold/touch the same "
                                           "(or the swapped) object; no melting, extra fingers or vanished objects.")
    faces_blurred: bool = Field(description="Every face that is blurred in the original is still blurred in the "
                                            "variant. True if no face is in view.")
    only_edit_changed: bool = Field(description="Nothing but what the instruction asked for differs: the people's "
                                                "clothing, gloves and skin, the other objects and the background "
                                                "are the same as in the original.")
    plausible: bool = Field(description="The variant looks like something that could have been filmed: the new "
                                        "object is at a believable size, belongs in this scene, and the hand holds "
                                        "it the way a person would.")
    actions_seen: list[str] = Field(description="Short action labels you can see in the variant, e.g. 'opens bag'.")
    score: int = Field(description="0-10 overall usefulness of the variant as robot-training data.")
    notes: str = Field(description="One sentence on the main problem, or 'ok'.")


def verify(original: Path, variant: Path, edit: Edit, card: dict, work: Path) -> tuple[Verdict, str]:
    """Compare the variant with its original. Returns the verdict and the model that gave it."""
    a = media.frames(original, work / "orig", n=5)
    b = media.frames(variant, work / edit.name, n=5)
    content: list[dict] = [{"type": "text", "text": "ORIGINAL clip, 5 frames in time order:"}]
    content += [claude.image_block(p) for p in a]
    content += [{"type": "text", "text": "VARIANT clip, the same 5 moments:"}]
    content += [claude.image_block(p) for p in b]
    content.append({"type": "text", "text": (
        f"The variant was made by a video editing model with this instruction:\n\"{edit.prompt}\"\n\n"
        f"Original labels: {json.dumps(card.get('labels', []))}\n\n"
        "Decide whether the variant is usable training data: the edit must be applied while the people, their "
        "hand movements and the object interactions stay the same as in the original. Compare the two rows "
        "frame by frame. Edit models like to redraw more than they were asked to: a gloved hand that comes back "
        "bare, a sleeve or a vest that changes, an object on the table that disappears, a hand that now holds "
        "the object differently. Any of those makes the variant unusable, however good it looks on its own. So "
        "does a result nobody would film: an object far too large or small for the hand, or one that has no "
        "business in this scene. Faces were blurred for privacy before the edit; a variant that shows a sharp "
        "face where the original has a blur is not usable.")})
    verdict, info = claude.ask_full(content, Verdict, purpose=f"verify:{edit.name}", effort="medium")
    return verdict, info["model"]


def passed(v: Verdict, hand_paths: dict | None = None) -> bool:
    """Does the variant go in the dataset? The checker's answers, and where there is one, the measurement."""
    return bool(v.edit_applied and v.motion_preserved and v.hands_intact and v.faces_blurred
                and v.only_edit_changed and v.plausible and v.score >= 6 and hands_kept(hand_paths))


HANDS_LOST = 0.25       # a variant may lose this much of the "hand in the same place" share of its own original


def hands_kept(hand_paths: dict | None) -> bool:
    """`hand_paths` is {"original": hands.same_path(...), "variant": hands.same_path(...)}: where the hands
    are in the stretch that was sent to the edit model, and in what came back. True if every hand that can be
    followed in the original is still where it was in the variant; also true where nothing could be measured."""
    if not hand_paths or not hand_paths.get("original") or not hand_paths.get("variant"):
        return True
    for side in ("left", "right"):
        before, after = hand_paths["original"].get(side), hand_paths["variant"].get(side)
        if not before or before["same"] < 0.5:      # the measurement does not work on this hand in this crop
            continue
        if not after or after["same"] < before["same"] - HANDS_LOST:
            return False
    return True


# ---------- fal ----------

def estimate(seconds: float, model: str, n: int) -> float:
    return seconds * FAL_MODELS[model]["usd_per_s"] * n


def run_edit(src: Path, edit: Edit, model: str, dst: Path) -> float:
    cfg = FAL_MODELS[model]
    secs = media.probe(src)["seconds"]
    cost = estimate(secs, model, 1)
    already = ledger.spent("fal")
    if already + cost > FAL_BUDGET_USD:
        raise SystemExit(f"fal budget stop: spent ${already:.2f}, this call ~${cost:.2f}, cap ${FAL_BUDGET_USD:.2f} "
                         f"(raise FAL_BUDGET_USD in .env to continue)")
    client = fal_client.SyncClient(key=secret("FAL_KEY"))
    url = client.upload_file(src)
    args = {"video_url": url, "prompt": edit.prompt}
    if model == "gemini":
        args["resolution"] = "720p"
    print(f"  -> {cfg['endpoint']} ({secs:.1f}s, ~${cost:.2f}) ...", flush=True)
    result = client.subscribe(cfg["endpoint"], arguments=args, with_logs=False)
    video_url = result["video"]["url"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(video_url) as r, dst.open("wb") as f:
        shutil.copyfileobj(r, f)
    billed = estimate(media.probe(dst)["seconds"], model, 1)
    ledger.record("fal", billed, endpoint=cfg["endpoint"], variant=edit.name, src=src.name, out=str(dst))
    return billed


# ---------- driver ----------

def widen_to_16x9(crop: tuple[int, int, int, int], src_w: int, src_h: int) -> tuple[int, int, int, int]:
    """Grow the crop box (keeping its center) to 16:9, clamped to the frame; even sizes for H.264."""
    x, y, w, h = crop
    cx, cy = x + w / 2, y + h / 2
    if w / h < 16 / 9:
        w = h * 16 / 9
    else:
        h = w * 9 / 16
    if w > src_w:
        w, h = src_w, src_w * 9 / 16
    if h > src_h:
        h, w = src_h, src_h * 16 / 9
    w, h = int(w) // 2 * 2, int(h) // 2 * 2
    x = int(min(max(cx - w / 2, 0), src_w - w))
    y = int(min(max(cy - h / 2, 0), src_h - h))
    return x, y, w, h


def multiply(clip: Path, card: dict, edits: list[Edit], model: str, *, start: float, seconds: float,
             crop: tuple[int, int, int, int] | None, out_root: Path, verify_variants: bool,
             dry_run: bool, force: bool, out_dir: Path | None = None, hand_paths=None) -> dict:
    """`out_dir` fixes the output folder (default: a folder under `out_root` named after the clip and window).
    Variants already in that folder are reused, including their check, unless `force` is set.
    `hand_paths`, if given, measures where the hands are in a video of this window (see hands.same_path):
    it is run on the original and on every variant, and a variant whose hands moved does not pass."""
    info = media.probe(clip)
    seconds = min(seconds, info["seconds"] - start, FAL_MODELS[model]["max_s"])
    if crop:
        # The edit models return 16:9; widen the crop so they don't zoom in and cut people off.
        crop = widen_to_16x9(crop, info["width"], info["height"])
    est = estimate(seconds, model, len(edits))
    print(f"{clip.name}: {len(edits)} variant(s) x {seconds:.1f}s on {model} ~ ${est:.2f} "
          f"(fal spent so far ${ledger.spent('fal'):.2f} / cap ${FAL_BUDGET_USD:.2f})")
    if dry_run:
        return {"estimate_usd": est}

    tag = hashlib.sha1(f"{clip.resolve()}|{start}|{seconds}|{crop}".encode()).hexdigest()[:8]
    out = out_dir or out_root / f"{clip.stem}-{tag}"
    original = out / "original.mp4"
    if force or not original.exists():
        media.cut(clip, original, start=start, seconds=seconds, crop=crop, max_height=720)
    media.dump(card, out / "original.json")

    results = []
    paths_before = None
    if hand_paths and verify_variants:
        try:
            paths_before = hand_paths(original)
        except Exception as e:      # a measurement that cannot be taken is not a reason to stop: the checker still looks
            print(f"  hand paths could not be measured: {type(e).__name__}: {e}")
            hand_paths = None
    with tempfile.TemporaryDirectory() as tmp:
        for edit in edits:
            dst = out / f"{edit.name}-{model}.mp4"
            meta = out / f"{edit.name}-{model}.json"
            print(f"[{edit.name}]")
            cost = 0.0
            if dst.exists() and not force:
                print("  cached, skipping generation")
                if meta.exists():
                    old = json.loads(meta.read_text())
                    # Records written before `verify_model` existed were all checked by Opus 5.5.
                    same_checker = (old.get("verify_model", "claude-opus-5-5") == CLAUDE_MODEL
                                    and old.get("check_version") == SWAP_VERSION)
                    if not verify_variants or ("verify" in old and same_checker):
                        # the video and its check are reused; the labels are made again from the card
                        results.append({**old, "labels": carry_labels(card, edit), "cost_usd": 0.0, "cached": True})
                        continue
                    # Checked by a different model than the one configured now: keep the video, check it again.
                    edit = dataclasses.replace(edit, prompt=old.get("prompt", edit.prompt))
            else:
                cost = run_edit(original, edit, model, dst)
            labels = carry_labels(card, edit)
            rec = {"variant": edit.name, "model": FAL_MODELS[model]["endpoint"], "prompt": edit.prompt,
                   "file": dst.name, "cost_usd": round(cost, 4), "labels": labels}
            if verify_variants:
                v, rec["verify_model"] = verify(original, dst, edit, card, Path(tmp))
                rec["verify"], rec["check_version"] = v.model_dump(), SWAP_VERSION
                if hand_paths and paths_before:
                    try:
                        rec["hand_paths"] = {"original": paths_before, "variant": hand_paths(dst)}
                    except Exception as e:
                        print(f"  hand paths could not be measured: {type(e).__name__}: {e}")
                rec["passed"] = passed(v, rec.get("hand_paths"))
                moved = "" if hands_kept(rec.get("hand_paths")) else " - the hands are not where they were"
                print(f"  verify: {'PASS' if rec['passed'] else 'FAIL'} score={v.score} - {v.notes}{moved}")
            media.dump(rec, meta)
            results.append(rec)

    keep = [out / r["file"] for r in results if r.get("passed", True)]
    if keep:
        media.side_by_side([original, *keep], out / f"grid-{model}.mp4")
    summary = {"original": str(original), "variants": results,
               "fal_spent_total": round(ledger.spent("fal"), 3),
               "anthropic_spent_total": round(ledger.spent("anthropic"), 3)}
    media.dump(summary, out / f"summary-{model}.json")
    print(f"done -> {out}  (fal total ${summary['fal_spent_total']:.2f}, "
          f"claude total ${summary['anthropic_spent_total']:.3f})")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--card", type=Path, help="JSON with 'labels' (and optional 'objects') for the clip")
    ap.add_argument("--variants", default="evening,store", help="comma list: evening,store,clothing,fluorescent,swap")
    ap.add_argument("--swap", help='object swap, e.g. "chip bag=apple"')
    ap.add_argument("--model", choices=sorted(FAL_MODELS), default="gemini")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=4.0, help="clip length sent to fal (cost scales with it)")
    ap.add_argument("--crop", help="x,y,w,h in source pixels")
    ap.add_argument("--out", type=Path, default=MULTIPLY_OUT)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the cost estimate and exit")
    ap.add_argument("--force", action="store_true", help="regenerate even if a cached variant exists")
    a = ap.parse_args(argv)

    swap = tuple(s.strip() for s in a.swap.split("=", 1)) if a.swap else None
    table = presets(swap)
    names = [n.strip() for n in a.variants.split(",") if n.strip()]
    unknown = [n for n in names if n not in table]
    if unknown:
        sys.exit(f"unknown variant(s) {unknown}; available: {sorted(table)} (swap needs --swap)")
    card = json.loads(a.card.read_text()) if a.card else {"labels": []}
    crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    multiply(a.clip, card, [table[n] for n in names], a.model, start=a.start, seconds=a.seconds, crop=crop,
             out_root=a.out, verify_variants=not a.no_verify, dry_run=a.dry_run, force=a.force)


if __name__ == "__main__":
    main()
