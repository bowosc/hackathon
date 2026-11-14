"""Shared catalog loading and filtering for individual clips and bulk datasets."""
import json
import os
import re
from pathlib import Path

from . import library

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = json.loads((ROOT / "config/settings.json").read_text())
# Words a request is full of that say nothing about which clip is wanted.
STOP_WORDS = set("""a an the of up and or to in on at by for from with into over as is are be it its this that these those
    i we me my our you your need want find show get give some any all every each more most
    someone somebody person people man woman customer their his her them they
    video videos clip clips footage data dataset datasets training train robot robots model models""".split())


def secret(name):
    """A key from the project's .env, then from the environment. The file wins, so a variable some other tool
    left in the shell cannot stand in for the project's own key."""
    try:
        for line in (ROOT / ".env").read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip().strip("'\"")
    except OSError:
        pass
    return os.environ.get(name) or None


def catalog():
    """Every clip in the footage library: the seed library plus what the engine processed here."""
    return library.clips()


def stem(word):
    """'picking', 'picks' and 'picked' all become 'pick', so a request and a label meet in the middle."""
    for ending in ("ing", "ed", "es", "s", "e"):
        if word.endswith(ending) and len(word) - len(ending) >= 3 and not (ending == "s" and word.endswith("ss")):
            word = word[:-len(ending)]
            break
    if len(word) > 3 and word[-1] == word[-2]:
        word = word[:-1]   # putting -> putt -> put, and pull -> pul so that it still meets pulling
    return word


def search_words(value):
    return [stem(word) for word in re.findall(r"\w+", value.lower()) if word not in STOP_WORDS]


def clip_text(clip):
    """What a search runs over: the words a person would use for the clip, not its file names."""
    scene = clip.get("scene") or {}
    parts = [clip["id"], clip["title"], clip["description"], clip["environment"], clip["lighting"], clip["camera"],
             *clip["actions"], *clip["objects"], *scene.values()]
    for segment in clip.get("segments", []):
        parts += [segment["verb"].replace("_", " "), segment["action"], segment["object"]]
    for version in clip.get("versions", []):
        parts += [version["title"], *version.get("objects", [])]
    return " ".join(str(part) for part in parts)


def filter_clips(clips, q="", filters=None, variant=None, sort="relevance"):
    """Filter groups combine with AND, options inside a group with OR. Text matches on word stems:
    clips holding every word come first; if none does, clips holding at least half of them are kept."""
    words = list(dict.fromkeys(search_words(q)))
    matches = []
    for clip in clips:
        if variant and variant not in clip["variants"]:
            continue
        if any(values and not set(values).intersection(
            clip[key] if isinstance(clip[key], list) else [clip[key]])
            for key, values in (filters or {}).items()):
            continue
        found = set(search_words(clip_text(clip)))
        matches.append((clip, sum(word in found for word in words)))
    if words:
        needed = len(words) if any(hits == len(words) for _, hits in matches) else max(1, (len(words) + 1) // 2)
        matches = [(clip, hits) for clip, hits in matches if hits >= needed]
    if sort == "newest":
        matches.sort(key=lambda match: match[0]["added_at"], reverse=True)
    elif sort == "duration":
        matches.sort(key=lambda match: match[0]["duration_seconds"], reverse=True)
    elif words:
        matches.sort(key=lambda match: (match[1], sum(word in search_words(match[0]["title"]) for word in words)),
                     reverse=True)
    return [clip for clip, _ in matches]
