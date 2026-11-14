"""Multiply: turn one labeled clip into N labeled variants.

  original clip + labels
      -> fal.ai video edit (one call per variant; motion, timing and framing kept)
      -> labels carried over from the original + the attributes the edit changed
      -> Claude compares variant frames to the original frames; failures are discarded

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

import fal_client
from pydantic import BaseModel, Field

from . import claude, ledger, media
from .config import CLAUDE_MODEL, FAL_BUDGET_USD, FAL_MODELS, MULTIPLY_OUT, secret

KEEP = ("Keep the camera angle, framing, timing, the people's bodies and hands, and every movement "
        "exactly the same. Faces are blurred on purpose: keep every blurred face blurred.")


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
        article = "" if new.split()[0].lower() in ("a", "an", "the") else ("an " if new[:1].lower() in "aeiou" else "a ")
        p["swap"] = Edit("swap", f"Replace the {old} the person is handling with {article}{new}. {KEEP}",
                         {"object_swap": f"{old} -> {new}"}, swap)
    return p


# ---------- labels ----------

def carry_labels(card: dict, edit: Edit) -> dict:
    """Variant labels = original labels with the edited attributes applied."""
    out = copy.deepcopy(card)
    if edit.swap:
        old, new = edit.swap

        def sub(v):
            if isinstance(v, str):
                return v.replace(old, new)
            if isinstance(v, list):
                return [sub(x) for x in v]
            if isinstance(v, dict):
                return {k: sub(x) for k, x in v.items()}
            return v

        for key in ("labels", "objects", "main_object", "caption"):
            if key in out:
                out[key] = sub(out[key])
    out.setdefault("scene", {}).update(edit.attributes)
    return out


class Verdict(BaseModel):
    edit_applied: bool = Field(description="The requested change is clearly visible in the variant.")
    motion_preserved: bool = Field(description="Same people doing the same movements at the same moments.")
    hands_intact: bool = Field(description="Hands look anatomically plausible and still hold/touch the same "
                                           "(or the swapped) object; no melting, extra fingers or vanished objects.")
    faces_blurred: bool = Field(description="Every face that is blurred in the original is still blurred in the "
                                            "variant. True if no face is in view.")
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
        "hand movements and the object interactions stay the same as in the original. Faces were blurred for "
        "privacy before the edit; a variant that shows a sharp face where the original has a blur is not usable.")})
    verdict, info = claude.ask_full(content, Verdict, purpose=f"verify:{edit.name}", effort="medium")
    return verdict, info["model"]


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
             dry_run: bool, force: bool, out_dir: Path | None = None) -> dict:
    """`out_dir` fixes the output folder (default: a folder under `out_root` named after the clip and window).
    Variants already in that folder are reused, including their check, unless `force` is set."""
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
                    same_checker = old.get("verify_model", "claude-opus-5-5") == CLAUDE_MODEL
                    if not verify_variants or ("verify" in old and same_checker):
                        results.append({**old, "cost_usd": 0.0, "cached": True})  # reuse the earlier check too
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
                rec["verify"] = v.model_dump()
                rec["passed"] = (v.edit_applied and v.motion_preserved and v.hands_intact and v.faces_blurred
                                 and v.score >= 6)
                print(f"  verify: {'PASS' if rec['passed'] else 'FAIL'} score={v.score} - {v.notes}")
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
