"""Shared catalog loading and filtering for individual clips and bulk datasets."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = json.loads((ROOT / "config/settings.json").read_text())


def catalog():
    return json.loads((ROOT / SETTINGS["catalog_file"]).read_text())


def search_words(value):
    aliases = {"picking": "pick", "placing": "place", "restocking": "restock", "objects": "object"}
    return [aliases.get(word, word) for word in re.findall(r"\w+", value.lower())
            if word not in {"a", "an", "the", "of", "up", "and"}]


def filter_clips(clips, q="", filters=None, variant=None, sort="relevance"):
    words = search_words(q)
    matches = []
    for clip in clips:
        searchable = " ".join(search_words(json.dumps(clip)))
        if not all(word in searchable for word in words):
            continue
        if variant and variant not in clip["variants"]:
            continue
        if any(values and not set(values).intersection(
            clip[key] if isinstance(clip[key], list) else [clip[key]])
            for key, values in (filters or {}).items()):
            continue
        matches.append(clip)
    if sort == "newest":
        matches.sort(key=lambda clip: clip["added_at"], reverse=True)
    elif sort == "duration":
        matches.sort(key=lambda clip: clip["duration_seconds"], reverse=True)
    elif words:
        matches.sort(key=lambda clip: sum(word in " ".join(search_words(clip["title"]))
                                         for word in words), reverse=True)
    return matches
