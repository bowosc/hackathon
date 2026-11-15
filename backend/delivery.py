"""Package existing library assets, labels and provenance; never invent generated outputs."""
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_STORED, ZipFile

from fastapi import HTTPException

from . import library


def asset(url):
    parsed = urlsplit(url)
    parts = unquote(parsed.path).removeprefix("/api/media/").split("/", 1)
    file = library.media_file(*parts) if not parsed.netloc and parsed.path.startswith("/api/media/") and len(parts) == 2 else None
    if file is None:
        raise HTTPException(409, "A dataset file is unavailable. Please refresh the selection before purchasing.")
    return file


def archive_plan(clips, max_bytes):
    files, records = [], []
    for index, clip in enumerate(clips):
        folder, urls = f"clips/{index + 1:06d}", {}

        def include(url, name):
            if not url:
                return None
            if url not in urls:
                file = asset(url)
                member = f"{folder}/{name}{file.suffix}"
                files.append((file, member))
                urls[url] = member
            return urls[url]

        versions = [{**version, "url": include(version["url"], f"{version['key']}-{number}")}
                    for number, version in enumerate(clip.get("versions", []))
                    if version["key"] in {"source", "upscaled", "augmented"} and version.get("url")]
        if not versions:
            raise HTTPException(409, "A selected clip has no downloadable video.")
        records.append({**clip, "versions": versions,
                        "video_url": urls.get(clip.get("video_url"), versions[0]["url"]),
                        "preview_url": urls.get(clip.get("preview_url"), versions[0]["url"]),
                        "image": include(clip.get("image"), "poster"),
                        "hands_url": include(clip.get("hands_url"), "hands"),
                        "card_url": include(clip.get("card_url"), "labels")})
    size = sum(file.stat().st_size for file, _ in files)
    if size > max_bytes:
        raise HTTPException(422, "This dataset is too large for an instant ZIP. Select a smaller bundle or use request mode for bulk delivery.")
    return {"files": files, "catalog": records, "bytes": size,
            "video_count": sum(len(clip["versions"]) for clip in records)}


def write_archive(path: Path, manifest, package):
    partial = path.with_suffix(".partial")
    try:
        with ZipFile(partial, "x", compression=ZIP_STORED, allowZip64=True) as archive:
            archive.writestr("dataset.json", json.dumps({**manifest, "clips": package["catalog"]}, indent=2))
            archive.writestr("README.txt", "Peelset dataset\n\n"
                "dataset.json maps source clip IDs to the video, poster and annotation files in clips/.\n"
                "Each bundle retains its original instructions, selected source IDs and transformation plan.\n"
                "This archive contains existing library files. Requested new transformations have not been generated.\n"
                "License and attribution information is preserved in each clip's metadata.\n")
            for file, member in package["files"]:
                archive.write(file, member)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
