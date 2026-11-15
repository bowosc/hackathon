"""Reading JSON that this site did not write itself.

A run's state, the index of its previews and its catalog can come from a worker: another machine that
holds the admin token. They are read, never trusted. A file is parsed only if it is small and plainly
shaped, and whatever comes out can always be sent on as JSON:

  - no NaN or infinity (a reply holding one cannot be encoded: the route would answer 500)
  - not nested deeper than anything the engine writes
  - not made of more lists and objects than a real run has (each costs far more memory than its text)
"""
import json
import math

MAX_PARTS = 150_000      # lists and objects in one file; an hour of footage makes about a third of this
MAX_DEPTH = 24           # the engine's deepest is 7


class Unreadable(ValueError):
    """A file that is not read: too large, too deep, or not JSON."""


def finite(text):
    value = float(text)
    return value if math.isfinite(value) else None


def load(text):
    """Parse JSON from outside. NaN, infinity and numbers too large for a float come out as None."""
    if text.count("{") + text.count("[") > MAX_PARTS:
        raise Unreadable("made of too many parts")
    try:
        value = json.loads(text, parse_constant=lambda name: None, parse_float=finite)
    except RecursionError as error:
        raise Unreadable("nested too deeply") from error
    todo = [(value, 1)]
    while todo:
        item, depth = todo.pop()
        if depth > MAX_DEPTH:
            raise Unreadable("nested too deeply")
        children = item.values() if isinstance(item, dict) else item if isinstance(item, list) else ()
        todo.extend((child, depth + 1) for child in children if isinstance(child, (dict, list)))
    return value


def read(path, limit):
    """The content of a JSON file, or None when it is missing, larger than `limit` bytes, or unreadable."""
    try:
        if path.stat().st_size > limit:
            return None
        return load(path.read_text())
    except (OSError, ValueError):      # Unreadable, bad JSON and bad UTF-8 are all ValueErrors
        return None


def rows(value):
    """The objects in a list (anything else in it, or anything that is not a list, is left out)."""
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def record(value):
    return value if isinstance(value, dict) else {}


def number(value):
    """A real, finite number, or None."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def plain_path(rel, part, depth=6):
    """Whether `rel` is a path of plain names (each matching the pattern `part`) and so stays where it is put:
    no '..', no leading dot, no backslash, nothing absolute."""
    parts = rel.split("/") if isinstance(rel, str) else []
    return 1 <= len(parts) <= depth and all(part.fullmatch(name) for name in parts)
