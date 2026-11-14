"""The footage library: clips the engine produced, in the shape the website shows.

Clips come from two places and are merged into one catalog:

  data/library/           the seed library that ships with the site (catalog.json + small web copies)
  <store>/runs/<run>/     runs the engine finished on this machine, for example from an admin upload

<store> is PEELSET_DATA, or data/ in the project when that is not set. Each engine item is one
accepted hand clip with its label card; verified object-swap variants are attached to their source clip.
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "library"
STORE = Path(os.environ.get("PEELSET_DATA") or ROOT / "data")
RUNS = STORE / "runs"
MEDIA_URL = "/api/media"
MEDIA_TYPES = {".mp4", ".webm", ".jpg", ".jpeg", ".png", ".json"}

# Engine verb -> the action family shown in filters and on cards. Families are listed most telling first.
FAMILIES = {
    "Pick & place": ["pick_up", "put_down"],
    "Handover": ["hand_over", "receive", "pay"],
    "Open & close": ["open", "close", "unwrap", "search_bag"],
    "Tool & device use": ["pour", "stir", "press", "tap_screen", "type", "write", "wipe"],
    "Push & pull": ["push", "pull"],
    "Eating & drinking": ["eat", "drink"],
    "Inspecting & adjusting": ["inspect", "adjust"],
    "Holding & carrying": ["hold", "carry"],
    "Reaching": ["reach"],
    "Other handling": ["other"],
}
FAMILY_OF = {verb: family for family, verbs in FAMILIES.items() for verb in verbs}
VERB_ORDER = [verb for verbs in FAMILIES.values() for verb in verbs]
VERB_TITLES = {
    "pick_up": "Picking up", "put_down": "Putting down", "hand_over": "Handing over", "receive": "Receiving",
    "pay": "Paying with", "open": "Opening", "close": "Closing", "unwrap": "Unwrapping", "search_bag": "Searching",
    "pour": "Pouring", "stir": "Stirring", "press": "Pressing", "tap_screen": "Tapping", "type": "Typing on",
    "write": "Writing on", "wipe": "Wiping", "push": "Pushing", "pull": "Pulling", "eat": "Eating",
    "drink": "Drinking from", "inspect": "Inspecting", "adjust": "Adjusting", "hold": "Holding",
    "carry": "Carrying", "reach": "Reaching for", "other": "Handling",
}
ANNOTATIONS = ["Action segments", "Hand tracks", "Object labels", "Scene descriptions"]

# The labeler describes the place in free text; filters need a short list. First match wins.
ENVIRONMENTS = [
    ("Grocery store", ("grocery", "supermarket", "aisle", "produce")),
    ("Gas station", ("gas station", "fuel", "forecourt", "pump")),
    ("Convenience store", ("convenience", "corner shop", "checkout", "retail")),
    ("Kitchen", ("kitchen",)),
    ("Warehouse", ("warehouse", "stock room", "loading")),
    ("Hallway", ("hallway", "corridor", "locker")),
    ("Café counter", ("counter",)),
    ("Café seating", ("cafe", "café", "coffee", "lounge", "lobby", "seating", "table", "stool", "waiting")),
    ("Office", ("office", "desk")),
]
NO_ARTICLE = ("a ", "an ", "the ", "some ", "two ", "three ", "rows ", "bags ", "pieces ", "bottled ", "paper money")


def environment(setting):
    text = setting.lower()
    return next((name for name, words in ENVIRONMENTS if any(word in text for word in words)), "Indoor space")


def lighting(text):
    text = text.lower()
    if any(word in text for word in ("dark", "night", "low light")) or ("dim" in text and "slightly dim" not in text):
        return "Low light"
    if any(word in text for word in ("daylight", "sun", "window", "glass door", "outdoor", "skylight")):
        return "Daylight"
    return "Indoor"


def resolution(size):
    """'1080p' for anything within a few rows of it: clips keep the crop's own even-numbered height."""
    height = size[1] if size else 0
    for label, low in (("4K", 2000), ("1080p", 1000), ("720p", 630), ("480p", 440), ("360p", 330)):
        if height >= low:
            return label
    return f"{height}p" if height else "unknown"


def with_article(name):
    if name.startswith(NO_ARTICLE) or (name.endswith("s") and not name.endswith("ss")):
        return name
    return f"{'an' if name[0] in 'aeiou' else 'a'} {name}"


def title(item):
    """'Picking up a paper cup': the most telling action on the clip's main object."""
    labels = item.get("labels") or []
    main = item.get("main_object")
    if not labels:
        return f"Handling {with_article(main)}" if main else "Hand-object interaction"
    best = min(labels, key=lambda label: (label["object"] != main, VERB_ORDER.index(label["verb"])
                                          if label["verb"] in VERB_ORDER else len(VERB_ORDER)))
    return f"{VERB_TITLES.get(best['verb'], 'Handling')} {with_article(best['object'])}"


def camera(source):
    match = re.search(r"\.(G\d{3})\.", source)   # MEVA file names carry the camera id
    return match.group(1) if match else re.sub(r"[^A-Za-z0-9]+", "-", Path(source).stem)[:12].upper()


def families(labels):
    found = {FAMILY_OF.get(label["verb"], "Other handling") for label in labels}
    return [family for family in FAMILIES if family in found]


def lineage_key(item):
    lineage = item.get("lineage", {})
    return (lineage.get("source"), tuple(lineage.get("frames", [])), tuple(lineage.get("crop_xywh", [])))


def sources():
    """Every catalog file with the URL its files are served under, oldest first."""
    found = []
    seed = SEED / "catalog.json"
    if seed.exists():
        found.append((0.0, seed, f"{MEDIA_URL}/seed", None))
    if RUNS.is_dir():
        for path in RUNS.glob("*/catalog.json"):
            found.append((path.stat().st_mtime, path, f"{MEDIA_URL}/runs/{path.parent.name}", path.parent.name))
    return sorted(found, key=lambda source: source[0])


def arrived(path):
    """When a run's catalog landed on this machine, in UTC, to the second. Runs can come from workers in other
    time zones, so this clock, not theirs, decides what is newest. Sorts after the seed library's plain dates."""
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def load_items():
    """Engine items from the seed and from finished runs. The same footage refined twice counts once (newest wins).
    Also returns the files those items name: the only files the site hands out."""
    originals, variants, files = {}, {}, set()
    for _, path, url, run in sources():
        try:
            items = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        added = arrived(path) if run else None
        for item in items if isinstance(items, list) else []:
            try:
                item = {**item, "url": url, "origin": "processed here" if run else "seed library",
                        "added_at": added or item.get("added_at") or "2026-11-14"}
                key = lineage_key(item)
                files.update(("runs", f"{run}/{rel}") if run else ("seed", rel) for rel in item.get("files", {}).values())
            except (TypeError, AttributeError):
                continue   # not an item: a damaged catalog must not take the others down with it
            if item.get("kind") == "variant":
                variants[(key, item.get("edit"))] = item
            else:
                originals[key] = item
    return originals, variants, files


def link(item, kind):
    rel = item.get("files", {}).get(kind)
    return f"{item['url']}/{rel}" if rel else None


def record(item, variants):
    """One engine item as a website clip record. The keys the explorer needs come first."""
    labels, scene = item.get("labels") or [], item.get("scene") or {}
    size, native = item.get("resolution"), item.get("native_resolution")
    video = link(item, "video")
    versions = [{"key": "upscaled", "title": "Upscaled 4×", "url": video,
                 "note": f"{size[0]}×{size[1]}" if size else ""}]
    if link(item, "original"):
        versions.append({"key": "source", "title": "Original crop", "url": link(item, "original"),
                         "note": f"{native[0]}×{native[1]}" if native else ""})
    if link(item, "context"):
        versions.append({"key": "context", "title": "Full scene", "url": link(item, "context"),
                         "note": "where the clip was cut from"})
    for variant in variants:
        check = variant.get("verify") or {}
        versions.append({"key": "augmented", "title": f"Object swap: {variant.get('edit', '').replace(' -> ', ' → ')}",
                         "url": link(variant, "video"), "note": f"checked {check.get('score', '?')}/10",
                         "id": variant["id"], "segments": variant.get("labels") or [],
                         "objects": variant.get("objects") or [], "check": check.get("notes", "")})
    lineage = item.get("lineage") or {}
    best = resolution(size)
    return {
        "id": item["id"], "title": title(item), "description": item.get("caption", ""),
        "environment": environment(scene.get("setting", "")), "actions": families(labels),
        "annotations": [name for name in ANNOTATIONS if name != "Hand tracks" or link(item, "hands")],
        "objects": item.get("objects") or [], "duration_seconds": item.get("seconds", 0), "fps": round(item.get("fps", 30)),
        "source_resolution": f"{native[0]}×{native[1]}" if native else best, "resolution": best,
        "resolutions": [best], "lighting": lighting(scene.get("lighting", "")),
        "variants": ["source", "upscaled"] + (["augmented"] if variants else []),
        "added_at": item["added_at"], "camera": camera(lineage.get("source", "")),
        "image": link(item, "poster"), "image_alt": f"Frame from the clip: {item.get('caption', '')}"[:240],
        "image_position": "50% 50%", "video_url": video, "preview_url": video,
        # beyond the explorer's own fields: what the engine knows about the clip
        "versions": versions, "segments": labels, "main_object": item.get("main_object"), "scene": scene,
        "quality": item.get("quality"), "tiers": item.get("tiers") or [], "labeled_by": item.get("labeled_by"),
        "hands_url": link(item, "hands"), "card_url": link(item, "card"), "origin": item["origin"],
        "lineage": {"source": lineage.get("source"), "frames": lineage.get("frames"),
                    "crop_xywh": lineage.get("crop_xywh"), "person_track": lineage.get("person_track"),
                    "run": item.get("run")},
        "license": item.get("license"), "attribution": item.get("attribution"),
    }


_cache = {"stamp": None, "clips": [], "files": set()}


def clips():
    """The catalog. Rebuilt only when a catalog file appears, changes or goes away."""
    stamp = tuple((str(path), path.stat().st_mtime_ns) for _, path, _, _ in sources())
    if stamp != _cache["stamp"]:
        originals, variants, files = load_items()
        by_parent = {}
        for (key, _), variant in variants.items():
            by_parent.setdefault(key, []).append(variant)
        built = []
        for key, item in originals.items():
            try:
                built.append(record(item, by_parent.get(key, [])))
            except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
                # One clip with a field missing (say, from a worker's damaged upload) is left out; the rest stay up.
                print(f"library: left out {item.get('id', '?')}: {type(error).__name__}: {error}", flush=True)
        built.sort(key=lambda clip: (clip["added_at"], clip["quality"] or 0, clip["id"]), reverse=True)
        _cache.update(stamp=stamp, clips=built, files=files)
    return _cache["clips"]


def version():
    """Changes whenever the catalog does: a key for anything computed from it."""
    clips()
    return _cache["stamp"]


def facets():
    """The filter groups of the explorer, built from what the catalog actually holds."""
    groups = [("environment", "Environment", "environment"), ("action", "Action", "actions"),
              ("annotation", "Annotations", "annotations"), ("resolution", "Resolution", "resolutions"),
              ("lighting", "Lighting", "lighting")]
    out = []
    for key, label, field in groups:
        counts = {}
        for clip in clips():
            for value in clip[field] if isinstance(clip[field], list) else [clip[field]]:
                counts[value] = counts.get(value, 0) + 1
        order = list(FAMILIES) if key == "action" else sorted(counts, key=lambda value: (-counts[value], value))
        out.append({"key": key, "label": label,
                    "options": [{"value": value, "count": counts[value]} for value in order if value in counts]})
    return out


def media_file(place, rel):
    """The file behind /api/media/<place>/<rel>, or None.

    Only files that a catalog item names are handed out. A run folder also holds the run's state, its log,
    rejected clips and previews of people; none of that is public."""
    root = {"seed": SEED, "runs": RUNS}.get(place)
    clips()   # brings the list of named files up to date
    if root is None or (place, rel) not in _cache["files"] or Path(rel).suffix.lower() not in MEDIA_TYPES:
        return None
    path = (root / rel).resolve()
    return path if path.is_relative_to(root.resolve()) and path.is_file() else None
