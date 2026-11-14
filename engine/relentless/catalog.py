"""Collect what every run produced into one catalog, for the website.

  uv run python -m relentless.catalog                  # writes data/catalog.json (paths relative to data/)
  uv run python -m relentless.catalog --export DIR     # also copies the files: DIR is self-contained
  uv run python -m relentless.catalog --export DIR --web   # the same, with small streaming copies of the videos

Each item is one sellable clip: an accepted original (upscaled clip, raw crop, context, hand
tracks, label card) or a verified variant of one, with its labels, tier and lineage.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from . import media
from .config import DATA, RUNS


def collect() -> list[dict]:
    """Every run's catalog items, oldest run first."""
    items = []
    for path in sorted(RUNS.glob("*/catalog.json"), key=lambda p: p.stat().st_mtime):
        run = path.parent.name
        state = json.loads(path.with_name("state.json").read_text()) if path.with_name("state.json").exists() else {}
        labeler = state.get("engine", {}).get("labeler")
        for item in json.loads(path.read_text()):
            item["files"] = {k: f"runs/{run}/{v}" for k, v in item.get("files", {}).items()}
            item.setdefault("labeled_by", labeler)   # runs from before the field existed: take it from the run
            item.setdefault("added_at", state.get("started", "")[:10] or None)   # the day the run was made
            items.append(item)
    return items


def dedupe(items: list[dict]) -> list[dict]:
    """The same footage refined by several runs counts once: keep the newest run's version."""
    latest: dict[tuple, dict] = {}
    for item in items:  # oldest run first, so a newer run's version replaces an older one
        lin = item.get("lineage", {})
        key = (lin.get("source"), tuple(lin.get("frames", [])), tuple(lin.get("crop_xywh", [])), item["kind"],
               item.get("edit"))
        latest[key] = item
    return list(latest.values())


# What a website needs of each clip, and how hard to compress it (x264 CRF). The side-by-side
# comparison is left out: a page can switch between the raw crop and the upscaled clip itself.
WEB_VIDEOS = {"video": 28, "original": 24, "context": 32}
WEB_COPIES = {"hands", "card", "poster"}


def export(items: list[dict], dst: Path, web: bool = False) -> list[dict]:
    """Copy every item's files into `dst`. With `web`, videos are re-encoded small and each item gets a poster."""
    out = []
    for item in items:
        files, folder = {}, dst / "clips" / item["id"]
        for kind, rel in item["files"].items():
            src = DATA / rel
            if not src.exists() or (web and kind not in WEB_COPIES and kind not in WEB_VIDEOS):
                continue
            if web and item["kind"] == "variant" and kind == "original":
                continue   # the stretch that was sent to the edit model: the parent clip already covers it
            target = folder / src.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if web and kind in WEB_VIDEOS:
                if not target.exists() or target.stat().st_mtime < src.stat().st_mtime:
                    media.web_copy(src, target, crf=WEB_VIDEOS[kind])
            else:
                shutil.copy2(src, target)
            files[kind] = str(target.relative_to(dst))
        if web and "poster" not in files and "video" in files:
            media.poster(DATA / item["files"]["video"], folder / "poster.jpg")
            files["poster"] = str((folder / "poster.jpg").relative_to(dst))
        out.append({**item, "files": files})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, help="copy every file into this folder and write its catalog.json there")
    ap.add_argument("--web", action="store_true", help="with --export: small streaming copies and posters only")
    ap.add_argument("--keep-duplicates", action="store_true", help="keep the same clip from several runs")
    a = ap.parse_args(argv)
    items = collect()
    if not a.keep_duplicates:
        items = dedupe(items)
    if a.export:
        items = export(items, a.export, web=a.web)
        target = a.export / "catalog.json"
    else:
        target = DATA / "catalog.json"
    media.dump(items, target)
    originals = sum(1 for i in items if i["kind"] == "original")
    seconds = sum(i["seconds"] for i in items)
    tiers = {t: sum(1 for i in items if t in i.get("tiers", [])) for t in ("bulk", "labeled+", "studio")}
    print(f"{len(items)} catalog items ({originals} originals, {len(items) - originals} variants, {seconds:.0f} s) "
          f"-> {target}")
    print("  tiers: " + ", ".join(f"{k} {v}" for k, v in tiers.items()))


if __name__ == "__main__":
    main()
