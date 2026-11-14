"""Read MEVA ground-truth annotations (Kitware KPF yaml) for the clips in data/raw/meva.

Each source clip has three files in data/raw/meva/annotations:
  *.activities.yml  what happened, when, and which actors took part
  *.geom.yml        a box per actor per frame
  *.types.yml       whether an actor is a person or an object
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# Ground-truth activities where a hand works on an object: what the engine is supposed to find.
HAND_OBJECT = {
    "person_picks_up_object", "person_puts_down_object", "person_transfers_object", "person_purchases",
    "person_opens_trunk", "person_closes_trunk", "person_opens_vehicle_door", "person_closes_vehicle_door",
    "person_opens_facility_door", "person_closes_facility_door", "person_loads_vehicle",
    "person_unloads_vehicle", "person_interacts_with_laptop", "person_reads_document",
    "person_texts_on_phone", "person_talks_on_phone", "hand_interacts_with_person",
}
# The short, unambiguous grasp events used for the headline recall number.
GRASP = {"person_picks_up_object", "person_puts_down_object", "person_transfers_object"}


@dataclass
class Activity:
    name: str
    start: int            # source frame, inclusive
    end: int              # source frame, inclusive
    people: list[int] = field(default_factory=list)   # actor ids typed as person
    objects: list[int] = field(default_factory=list)  # other actor ids


@dataclass
class GroundTruth:
    activities: list[Activity]
    geom: dict[int, tuple[np.ndarray, np.ndarray]]  # actor id -> (frames sorted, boxes x1 y1 x2 y2)

    def box(self, actor: int, frame: int) -> np.ndarray | None:
        """The actor's box at the annotated frame nearest to `frame` (within 15 frames)."""
        if actor not in self.geom:
            return None
        frames, boxes = self.geom[actor]
        i = int(np.clip(np.searchsorted(frames, frame), 0, len(frames) - 1))
        if i > 0 and abs(frames[i - 1] - frame) < abs(frames[i] - frame):
            i -= 1
        return boxes[i] if abs(int(frames[i]) - frame) <= 15 else None


def _entries(path: Path):
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        try:
            yield ast.literal_eval(body)  # most files are written as Python literals (fast path)
        except (ValueError, SyntaxError):
            yield yaml.load(body, Loader=loader)


def annotation_files(video: Path) -> dict[str, Path] | None:
    """The three annotation files for a raw clip, or None if it has none."""
    base = video.parent / "annotations" / video.stem
    files = {k: base.with_name(f"{video.stem}.{k}.yml") for k in ("activities", "geom", "types")}
    return files if all(p.exists() for p in files.values()) else None


def load(video: Path) -> GroundTruth | None:
    files = annotation_files(video)
    if not files:
        return None
    kinds: dict[int, str] = {}
    for e in _entries(files["types"]):
        t = e.get("types") if isinstance(e, dict) else None
        if t:
            kinds[int(t["id1"])] = next(iter(t["cset3"]))
    acts = []
    for e in _entries(files["activities"]):
        a = e.get("act") if isinstance(e, dict) else None
        if not a:
            continue
        start, end = a["timespan"][0]["tsr0"]
        ids = [int(x["id1"]) for x in a.get("actors", [])]
        acts.append(Activity(
            name=next(iter(a["act2"])), start=int(start), end=int(end),
            people=[i for i in ids if kinds.get(i, "person").lower() == "person"],
            objects=[i for i in ids if kinds.get(i, "person").lower() != "person"]))
    rows: dict[int, list] = {}
    for e in _entries(files["geom"]):
        g = e.get("geom") if isinstance(e, dict) else None
        if g:
            rows.setdefault(int(g["id1"]), []).append((int(g["ts0"]), *[float(v) for v in str(g["g0"]).split()]))
    geom = {}
    for actor, r in rows.items():
        arr = np.array(sorted(r), dtype=np.float32)
        geom[actor] = (arr[:, 0].astype(np.int64), arr[:, 1:5])
    return GroundTruth(sorted(acts, key=lambda a: a.start), geom)


if __name__ == "__main__":
    import collections
    import sys

    for arg in sys.argv[1:]:
        gt = load(Path(arg))
        if gt is None:
            print(f"{arg}: no annotations")
            continue
        names = collections.Counter(a.name for a in gt.activities)
        print(f"{Path(arg).name}: {len(gt.activities)} activities, {len(gt.geom)} actors with boxes")
        for n, c in names.most_common():
            print(f"  {c:4d}  {n}{'  *' if n in GRASP else ''}")
