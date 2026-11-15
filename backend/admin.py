"""Admin API: upload raw footage and run the engine on it. Every route needs the admin token.

Footage is uploaded in pieces to <store>/uploads/. A job runs the engine (engine/) on one upload and
ends with a run folder under <store>/runs/. Its accepted clips are that job's dataset: they join the
public catalog when the job is published from its page (see library.py). A job runs in one of two places:

  on this machine    as a separate process, one job at a time
  on a worker        another machine with the engine (and usually a GPU) that asks this site for work:
                     `python -m relentless.worker <site>` fetches the footage, processes it there and
                     sends the finished clips back

The job page follows a run stage by stage. What it shows comes from the run folder: the engine's
state.json and the previews it writes with --live (engine/relentless/live.py). A worker keeps a copy
of those up to date in <store>/live/<job>/ while its run goes on.
"""
import hashlib
import hmac
import importlib.util
import json
import os
import platform
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import documents, library
from .catalog import secret
from .documents import number, record, rows
from .library import ROOT, RUNS, STORE

UPLOADS = STORE / "uploads"
PARTIAL = UPLOADS / ".partial"
JOBS_FILE = STORE / "jobs.json"
INCOMING = STORE / "incoming"     # results a worker is still sending
LIVE = STORE / "live"             # a worker's run as it stands (state, previews), until its result arrives
FILM = UPLOADS / ".film"          # one strip of small frames per upload
TRASH = STORE / "trash"           # folders on their way out: moved here in one step, then removed
RESULT_TYPES = {".mp4", ".jpg", ".json", ".log"}
MAX_PREVIEW = 8 * 1024 * 1024         # one file of a worker's run in progress
MAX_PREVIEWS = 400 * 1024 * 1024      # all of them, per job
MAX_PREVIEW_FILES = 400               # per job; a run of 24 clips with variants has about 150
MAX_ALL_PREVIEWS = 1536 * 1024 * 1024     # of all jobs together
MAX_RESULT_FILES = 2000               # files in a worker's finished run
NAME_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}")
# Files of a run that this site reads itself, and the most it reads of each. They can come from a worker, and a
# parsed file costs many times its size in memory: a larger one is left unread (and refused when it is offered).
READ_LIMITS = {"state.json": 3 * 1024 * 1024, "live/now.json": 512 * 1024, "live/kit.json": 64 * 1024,
               "catalog.json": library.MAX_CATALOG}
FILM_MEMORY = 400 * 1024 * 1024       # free memory the strip of small frames needs before it is made
ENGINE_LABELER = "claude-haiku-5-5"   # the engine's default model (engine/relentless/config.py)
MAX_RESULT = 1024 * 1024 * 1024
WORKER_SILENCE = 180              # seconds without a word before a worker's job counts as lost
ENGINE = ROOT / "engine"
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
MB = 1024 * 1024
MAX_UPLOAD = int(os.environ.get("PEELSET_MAX_UPLOAD_MB", "2048")) * MB
CHUNK = 8 * MB            # the admin page sends a file in pieces of this size
MAX_CHUNK = 32 * MB
FREE_MARGIN = 500 * MB    # never fill the disk to the last byte: other things live on it
# The engine shares its machine. A job starts only if this much memory is free, and is stopped if the
# machine as a whole gets down to the second number, so it is the job that gives way, not a neighbour.
JOB_MEMORY = int(os.environ.get("PEELSET_JOB_MEMORY_MB", "1200")) * MB
SCARCE_MEMORY = int(os.environ.get("PEELSET_SCARCE_MEMORY_MB", "300")) * MB


MEDIA_COOKIE = "peelset_media"


def admin_token():
    token = secret("ADMIN_TOKEN") or ""
    if not token:
        raise HTTPException(503, "The admin panel is off. Set ADMIN_TOKEN on the server to turn it on.")
    return token


def bearer(authorization):
    return authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""


def admin(authorization: str = Header("")):
    if not hmac.compare_digest(bearer(authorization).encode(), admin_token().encode()):
        raise HTTPException(401, "That admin token is not right.")


_session = {"key": secrets.token_hex(16)}      # changes when this process starts and when anyone signs out


def media_pass(token):
    """What the admin page's browser holds, as a cookie, to load pictures and videos: an <img> or a <video>
    cannot send the token. It is made from the token and a key this process keeps to itself. It opens the two
    read-only routes below and nothing else, and stops working when the token changes, when anyone signs out
    and when the site restarts (the page then asks for a new one)."""
    return hmac.new(token.encode(), f"peelset-media:{_session['key']}".encode(), "sha256").hexdigest()


def session_name():
    """Which cookie is good right now, as a name that gives nothing away. The page asks again when it changes."""
    return hashlib.sha256(f"session:{_session['key']}".encode()).hexdigest()[:12]


def viewer(request: Request, authorization: str = Header("")):
    """The admin token, or the cookie the admin page was given for it."""
    token = admin_token()
    cookie = request.cookies.get(MEDIA_COOKIE, "")
    if not (hmac.compare_digest(bearer(authorization).encode(), token.encode())
            or hmac.compare_digest(cookie.encode(), media_pass(token).encode())):
        raise HTTPException(401, "That admin token is not right.")


router = APIRouter(prefix="/api/admin", dependencies=[Depends(admin)])
media_router = APIRouter(prefix="/api/admin", dependencies=[Depends(viewer)])   # files only, read only


@router.post("/session")
def open_session(request: Request, response: Response):
    """Give this browser the cookie that lets the page show a job's pictures and videos."""
    https = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip() == "https"
    response.set_cookie(MEDIA_COOKIE, media_pass(admin_token()), max_age=7 * 24 * 3600, path="/api/admin",
                        httponly=True, samesite="strict", secure=https)
    return {"ok": True}


@router.delete("/session")
def close_session(response: Response):
    """Sign out: the cookie is taken back from this browser, and every copy of it stops working. The browser is
    also told to drop what it has cached of this site, so pictures of a job it has shown cannot be opened from
    its cache afterwards."""
    _session["key"] = secrets.token_hex(16)
    response.delete_cookie(MEDIA_COOKIE, path="/api/admin")
    response.headers["Clear-Site-Data"] = '"cache"'
    return {"ok": True}


# ---------- the machine ----------

def engine_python():
    """The Python that can run the engine: PEELSET_ENGINE_PYTHON, the engine's own environment, or this one."""
    explicit = os.environ.get("PEELSET_ENGINE_PYTHON")
    if explicit:
        return explicit if Path(explicit).exists() else None
    for candidate in (ENGINE / ".venv/bin/python", ENGINE / ".venv/Scripts/python.exe"):
        if candidate.exists():
            return str(candidate)
    needed = ("torch", "ultralytics", "spandrel", "cv2", "imageio_ffmpeg", "anthropic", "lap")
    return sys.executable if all(importlib.util.find_spec(name) for name in needed) else None


def memory():
    """Total and available memory in bytes (available is only known on Linux)."""
    try:
        fields = dict(line.split(":") for line in Path("/proc/meminfo").read_text().splitlines())
        return int(fields["MemTotal"].split()[0]) * 1024, int(fields["MemAvailable"].split()[0]) * 1024
    except (OSError, KeyError, ValueError):
        try:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"), None
        except (ValueError, OSError):
            return None, None


def storage_kind():
    if os.path.ismount(STORE):
        return "volume"
    return "container" if Path("/.dockerenv").exists() else "disk"


def ledger_usd():
    """What the engine's Claude calls have cost on this machine so far."""
    try:
        rows = [json.loads(line) for line in (STORE / "spend-ledger.jsonl").read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return 0.0
    return round(sum(row.get("usd", 0) for row in rows if row.get("service") == "anthropic"), 4)


@router.get("/status")
def status():
    STORE.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(STORE)
    total, available = memory()
    python = engine_python()
    clips = library.clips()
    with _lock:
        jobs = [dict(job) for job in _jobs]
        at_work = {job.get("worker") for job in jobs if job["status"] in ("running", "stopping")}
        workers = [{"name": name, "device": seen["device"], "labeler": seen.get("labeler") or "",
                    "fal": seen.get("fal") or {}, "busy": name in at_work}
                   for name, seen in sorted(_workers.items()) if time.time() - seen["at"] < 30]
    done = [picture(job) for job in jobs if job["status"] == "done"]
    return {
        "session": session_name(),
        "engine": {"ready": bool(python), "device": os.environ.get("RELENTLESS_DEVICE") or "auto",
                   "claude_key": bool(secret("ANTHROPIC_API_KEY")), "claude_usd": ledger_usd(),
                   "labeler": os.environ.get("RELENTLESS_CLAUDE_MODEL") or ENGINE_LABELER,
                   "busy": any(job["status"] in ("running", "stopping") and not job.get("worker") for job in jobs)},
        "storage": {"path": str(STORE), "kind": storage_kind(), "free_bytes": disk.free, "total_bytes": disk.total},
        "machine": {"cpus": os.cpu_count(), "arch": platform.machine(), "system": platform.system(),
                    "memory_bytes": total, "memory_available_bytes": available},
        "library": {"clips": len(clips), "processed_here": sum(clip["origin"] == "processed here" for clip in clips)},
        "limits": {"upload_bytes": MAX_UPLOAD, "chunk_bytes": CHUNK, "job_memory_bytes": JOB_MEMORY},
        "workers": workers,
        # what the jobs on this site have made so far
        "totals": {"jobs": len(done), "footage_seconds": round(sum(job["source"].get("seconds") or 0 for job in done), 1),
                   "clips": sum(job["clips"]["accepted"] for job in done), "variants": sum(job["variants"] for job in done),
                   "usd": round(sum((job["claude_usd"] or 0) + (job["fal_usd"] or 0) for job in done), 4)},
    }


# ---------- footage ----------

def safe_name(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).name).strip("-.")
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    if not stem or suffix not in VIDEO_TYPES:
        raise HTTPException(422, f"Upload a video file: {', '.join(sorted(VIDEO_TYPES))}.")
    return stem[:100] + Path(name).suffix   # a long name is cut in the middle, never at its extension


def ffmpeg():
    """An ffmpeg to read video details with: this Python's, the engine environment's, or the system's."""
    if importlib.util.find_spec("imageio_ffmpeg"):
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    python = engine_python()
    if python:
        bundled = sorted(Path(python).parent.parent.glob("lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*"))
        if bundled:
            return str(bundled[0])
    return shutil.which("ffmpeg")


_probed = {}


def probe(path):
    """Length, size and frame rate of a video, read from ffmpeg's banner. Empty when ffmpeg is not around."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _probed:
        info, tool = {}, ffmpeg()
        if tool:
            try:
                text = subprocess.run([tool, "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                                      timeout=20).stderr
                length = re.search(r"Duration: (\d+):(\d+):([\d.]+)", text)
                size = re.search(r"Video: .*?, (\d{2,5})x(\d{2,5})", text)
                rate = re.search(r"([\d.]+) fps", text)
                codec = re.search(r"Video: (\w+)", text)
                if length and size:
                    hours, minutes, seconds = length.groups()
                    info = {"seconds": round(int(hours) * 3600 + int(minutes) * 60 + float(seconds), 1),
                            "width": int(size.group(1)), "height": int(size.group(2)),
                            "fps": round(float(rate.group(1)), 2) if rate else None,
                            "codec": codec.group(1) if codec else None}
                else:
                    info = {"unreadable": True}
            except (OSError, subprocess.SubprocessError):
                pass
        _probed[key] = info
    return _probed[key]


def described(name):
    """What a file name says about where footage comes from. MEVA names carry the day, the time and the camera."""
    match = re.match(r"(\d{4}-\d\d-\d\d)\.(\d\d)-(\d\d)-\d\d\..*\.(G\d+)$", Path(name).stem)
    if not match:
        return None
    return {"dataset": "MEVA", "camera": match.group(4), "recorded": f"{match.group(1)} {match.group(2)}:{match.group(3)}",
            "license": "CC BY 4.0"}


_film = threading.Lock()
_film_failed = {}      # (file, size, changed) -> when a strip could not be made of it


def make_strip(path, strip):
    """Write the strip of `path`. One frame at a time: ten decoders of 1080p at once take a gigabyte."""
    tool, seconds = ffmpeg(), probe(path).get("seconds")
    available = memory()[1]
    if not tool or not seconds or (available is not None and available < FILM_MEMORY):
        return False
    quiet = ["nice", "-n", "10"] if shutil.which("nice") else []
    quiet += [tool, "-hide_banner", "-loglevel", "error", "-y"]
    FILM.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=FILM, prefix=".making-") as scratch:
            frames = [Path(scratch) / f"{index}.jpg" for index in range(10)]
            for index, frame in enumerate(frames):
                subprocess.run([*quiet, "-threads", "1", "-ss", f"{seconds * (index + 0.5) / 10:.2f}", "-i", str(path),
                                "-frames:v", "1", "-vf", "scale=128:72,setsar=1", "-q:v", "5", str(frame)],
                               capture_output=True, timeout=30)
                if not frame.is_file():
                    return False
            inputs = [part for frame in frames for part in ("-i", str(frame))]
            whole = Path(scratch) / "strip.jpg"
            subprocess.run([*quiet, *inputs, "-filter_complex", "hstack=inputs=10", "-frames:v", "1", "-q:v", "5", str(whole)],
                           capture_output=True, timeout=30)
            if not whole.is_file():
                return False
            os.replace(whole, strip)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def film_strip(path):
    """Ten frames of a video side by side in one small picture, made once per upload. Returns its path, or None.

    Each frame is 128 pixels wide: enough to tell one piece of footage from another, and far too little to
    tell who is in it. (No face has been blurred at this point: the engine does that when it runs.)
    Only one strip is made at a time, and a file no strip could be made of is not tried again for ten minutes:
    a page asking for the picture must not be able to keep ffmpeg busy."""
    strip = FILM / f"{path.name}.jpg"
    try:
        stat = path.stat()
        if strip.is_file() and strip.stat().st_mtime >= stat.st_mtime:
            return strip
    except OSError:
        return None
    key = (path.name, stat.st_size, stat.st_mtime_ns)
    if time.time() - _film_failed.get(key, 0) < 600 or not _film.acquire(timeout=15):
        return None
    try:
        if strip.is_file() and strip.stat().st_mtime >= stat.st_mtime:      # made while this call waited
            return strip
        if make_strip(path, strip):
            return strip
        if len(_film_failed) > 500:
            _film_failed.clear()
        _film_failed[key] = time.time()
        return None
    finally:
        _film.release()


def footage_entry(path):
    stat = path.stat()
    return {"name": path.name, "bytes": stat.st_size, "source": described(path.name),
            "uploaded_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(), **probe(path)}


@router.get("/footage")
def list_footage():
    files = [path for path in UPLOADS.glob("*") if path.is_file() and path.suffix.lower() in VIDEO_TYPES]
    return {"footage": [footage_entry(path) for path in sorted(files, key=lambda path: -path.stat().st_mtime)]}


async def take_piece(request, part, total, offset):
    """Append one piece of an upload to `part`. Returns how many bytes have arrived."""
    part.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    if offset == 0:
        if shutil.disk_usage(STORE).free < total + FREE_MARGIN:
            raise HTTPException(507, "Not enough free disk space for this file.")
    elif offset != have:
        raise HTTPException(409, f"Expected the piece starting at byte {have}.")
    piece = await request.body()
    if not piece or len(piece) > MAX_CHUNK or offset + len(piece) > total:
        raise HTTPException(413 if piece else 422, "That piece is empty or too large.")
    with open(part, "r+b" if offset else "wb") as file:
        file.seek(offset)
        file.write(piece)
    return offset + len(piece)


@router.put("/footage")
async def upload(request: Request, name: str = Query(..., max_length=300), total: int = Query(..., ge=1),
                 offset: int = Query(0, ge=0)):
    """One piece of a file. `offset` is where the piece starts; the reply says how much has arrived."""
    name = safe_name(name)
    if total > MAX_UPLOAD:
        raise HTTPException(413, f"Files can be up to {MAX_UPLOAD // MB} MB.")
    part = PARTIAL / f"{name}.part"
    arrived = await take_piece(request, part, total, offset)
    if arrived < total:
        return {"done": False, "offset": arrived}
    final = UPLOADS / name
    os.replace(part, final)
    entry = await run_in_threadpool(footage_entry, final)   # reads the video with ffmpeg: not on the event loop
    if entry.get("unreadable"):
        final.unlink()
        raise HTTPException(422, "That file does not look like a video ffmpeg can read.")
    await run_in_threadpool(film_strip, final)
    return {"done": True, "footage": entry}


@media_router.get("/footage/{name}/film")
def footage_film(name: str):
    """Ten small frames of the footage, side by side."""
    path = UPLOADS / safe_name(name)
    strip = film_strip(path) if path.is_file() else None
    if not strip:
        raise HTTPException(404, "No frames for this footage.")
    return FileResponse(strip, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/footage/{name}/file")
def footage_file(name: str):
    """The raw file, for a worker that will process it."""
    path = UPLOADS / safe_name(name)
    if not path.is_file():
        raise HTTPException(404, "No such footage.")
    return FileResponse(path)


@router.delete("/footage/{name}")
def delete_footage(name: str):
    path = UPLOADS / safe_name(name)
    with _lock:
        if any(job["footage"] == path.name and job["status"] in ("queued", "running", "stopping") for job in _jobs):
            raise HTTPException(409, "This footage is being processed. Stop the job first.")
    if not path.is_file():
        raise HTTPException(404, "No such footage.")
    path.unlink()
    (FILM / f"{path.name}.jpg").unlink(missing_ok=True)
    return {"deleted": path.name}


# ---------- jobs ----------

class JobRequest(BaseModel):
    footage: str = Field(max_length=300)
    start: float = Field(0, ge=0, le=86400)
    seconds: Optional[float] = Field(None, gt=0, le=3600)       # None: to the end of the video
    max_clips: int = Field(4, ge=1, le=24)
    quality: Literal["fast", "full"] = "fast"                   # fast: 720p clips; full: 1080p, about twice as slow
    where: Literal["server", "worker"] = "server"               # this machine, or a worker that asks for the job
    multiply: int = Field(0, ge=0, le=3)                        # object-swap variants for this many clips (a worker's fal key pays)


_lock = threading.Lock()
_wake = threading.Event()
_jobs = []        # oldest first; kept in JOBS_FILE
_running = {}     # job id -> its process
_workers = {}     # worker name -> when it last asked for work, and what it runs on


def now():
    return datetime.now(timezone.utc).isoformat()


def save_jobs():
    STORE.mkdir(parents=True, exist_ok=True)
    temporary = JOBS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(_jobs[-200:], indent=2))
    os.replace(temporary, JOBS_FILE)


def load_jobs():
    try:
        _jobs.extend(json.loads(JOBS_FILE.read_text()))
    except (OSError, ValueError):
        return
    for job in _jobs:
        if job["status"] in ("running", "stopping") and not job.get("worker"):   # this process is new: what ran here is gone
            job.update(status="failed" if job["status"] == "running" else "stopped", finished_at=now(),
                       error="The server restarted while this was running.")
        elif job.get("worker"):
            job["heard"] = time.time()   # a worker's job goes on; give it a fresh grace period
    save_jobs()
    known = {job["id"] for job in _jobs}
    for folder in LIVE.glob("*") if LIVE.is_dir() else []:
        if folder.name not in known:       # previews of jobs that are gone
            shutil.rmtree(folder, ignore_errors=True)
        else:                              # what the others hold counts against the room for previews
            _previews[folder.name] = measure(folder)
    shutil.rmtree(TRASH, ignore_errors=True)   # whatever a restart interrupted on its way out


def run_folder(job):
    """Where a job's run can be read: its run folder, or the copy a worker keeps here while its run goes on."""
    folder = RUNS / job["run"]
    return folder if (folder / "state.json").is_file() or not (LIVE / job["id"]).is_dir() else LIVE / job["id"]


_documents = {}                 # file -> (what it was when read, its content): the files of the jobs being looked at
_reading = threading.Lock()     # one file is parsed at a time: a parsed file is many times its size in memory


def read_json(path, limit, keep=True):
    """A JSON file of a run as a dict ({} when it is missing, larger than `limit`, or not something the engine
    writes: see documents.py). With `keep`, a file is parsed once per version of it however often it is asked
    for; what comes back is shared, so it must not be changed."""
    try:
        stat = path.stat()
    except OSError:
        return {}
    stamp = (stat.st_mtime_ns, stat.st_size)
    with _reading:
        if _documents.get(path, (None,))[0] == stamp:
            return _documents[path][1]
        content = record(documents.read(path, limit))
        if keep:
            _documents.pop(path, None)
            while len(_documents) >= 9:      # the three files of the last few jobs anyone opened
                _documents.pop(next(iter(_documents)))
            _documents[path] = (stamp, content)
    return content


def run_state(job, keep=True):
    return read_json(run_folder(job) / "state.json", READ_LIMITS["state.json"], keep) if job.get("run") else {}


def log_tail(job, lines=12):
    try:
        text = (RUNS / job["run"] / "run.log").read_text(errors="replace")
    except (OSError, KeyError, TypeError):
        return ""
    keep = [line for line in text.splitlines() if line.strip() and "Warning" not in line]
    return "\n".join(keep[-lines:])


def text(value, most=300):
    return value[:most] if isinstance(value, str) else None


def stage_row(stage):
    """One stage of a run as the jobs list shows it. A state can come from a worker: it is read, never trusted."""
    return {"key": text(stage.get("key"), 40), "title": text(stage.get("title"), 80), "status": text(stage.get("status"), 20),
            "progress": number(stage.get("progress")), "detail": text(stage.get("detail")), "seconds": number(stage.get("seconds"))}


def summary(state):
    """The short picture of a run: stages, clip counts, what it cost. A worker reports the first of these itself."""
    clips, costs, source = rows(state.get("clips")), record(state.get("costs")), record(state.get("source"))
    size = source.get("size")
    labeler = record(state.get("engine")).get("labeler")
    return {"stages": [stage_row(stage) for stage in rows(state.get("stages"))[:20] if isinstance(stage.get("key"), str)],
            "clips": {"planned": len(clips), "accepted": sum(clip.get("status") == "accepted" for clip in clips),
                      "rejected": sum(clip.get("status") == "rejected" for clip in clips)},
            "claude_usd": number(costs.get("anthropic_usd")), "fal_usd": number(costs.get("fal_usd")),
            "variants": sum(bool(variant.get("passed")) for clip in clips for variant in rows(clip.get("variants"))),
            "labeler": text(labeler, 60), "elapsed": number(state.get("elapsed")),
            "source": {"start_s": number(source.get("start_s")), "seconds": number(source.get("seconds")),
                       "fps": number(source.get("fps")),
                       "size": size if isinstance(size, list) and len(size) == 2 and all(number(v) for v in size) else None}}


_pictures = {}    # state file -> (what it was when read, its summary): the jobs list is asked for every two seconds


def picture(job):
    """The summary of a job's run, read again only when its state file has changed."""
    path = run_folder(job) / "state.json"
    try:
        stat = path.stat()
    except OSError:
        return summary({})
    stamp = (stat.st_mtime_ns, stat.st_size)
    if _pictures.get(path, (None,))[0] != stamp:
        if len(_pictures) > 500:
            _pictures.clear()
        _pictures[path] = (stamp, summary(run_state(job, keep=False)))      # only the summary is kept: it is small
    return _pictures[path][1]


def published(job):
    """Whether the job's clips are in the public catalog."""
    folder = RUNS / job["run"]
    return (folder / "catalog.json").is_file() and not (folder / library.HOLD).exists()


def job_view(job):
    """A job with the short picture of its run: stages, clip counts and what it has cost.

    The picture comes from the run's state when there is one, else from what a worker last reported."""
    remote = job.get("remote") or {}
    view = {key: value for key, value in job.items() if key not in ("remote", "heard")}
    seen = picture(job)
    view.update(seen if seen["stages"] else {key: remote.get(key, seen[key]) if remote.get(key) is not None else seen[key]
                                              for key in seen})
    view["published"] = job["status"] == "done" and published(job)
    if job["status"] in ("running", "stopping", "failed", "stopped"):
        view["log"] = log_tail(job) or remote.get("log", "")
    return view


SHOWN = ("name", "status", "started", "finished", "elapsed", "error", "source", "options", "engine", "stages",
         "funnel", "motion", "people", "candidates", "clips", "costs", "log")


def shown_state(state):
    """The run's state as the job page needs it: no paths of the machine it ran on, no more log than fits."""
    shown = {key: state.get(key) for key in SHOWN if key in state}
    if isinstance(shown.get("source"), dict):
        shown["source"] = {key: value for key, value in shown["source"].items() if key != "path"}
    if isinstance(shown.get("motion"), dict):
        shown["motion"] = {key: value for key, value in shown["motion"].items() if key != "frame_spans"}
    if isinstance(shown.get("log"), list):
        shown["log"] = [{**line, "msg": re.sub(r" -> /\S*$", "", str(line.get("msg", "")))}
                        for line in shown["log"][-150:] if isinstance(line, dict)]
    return shown


_listed = {}     # folder -> (when, its files): a job page asks every second, and so may several pages at once


def versions(folder):
    """Every file of a run the job page may ask for, with a number that changes when the file does."""
    at, found = _listed.get(folder, (0.0, None))
    if found is not None and time.time() - at < 0.8:
        return found
    found = {}
    if folder.is_dir():
        for path in folder.rglob("*"):
            rel = path.relative_to(folder).as_posix()
            if plain(rel) and path.is_file():
                found[rel] = path.stat().st_mtime_ns // 1_000_000
            if len(found) >= MAX_RESULT_FILES:
                break
    if len(_listed) > 50:
        _listed.clear()
    _listed[folder] = (time.time(), found)
    return found


def plain(rel):
    """Whether `rel` names a file the way a run folder does: a few plain names, a known kind of file, and not
    the marker that holds a run back from the catalog (nothing a worker sends may stand in its place)."""
    return (documents.plain_path(rel, NAME_PART, depth=5) and Path(rel).suffix.lower() in RESULT_TYPES
            and rel.split("/")[0] != library.HOLD)


def shown_path(rel):
    """A path inside a run folder, as the job page or a worker names it."""
    if not plain(rel):
        raise HTTPException(404, "No such file.")
    return Path(*rel.split("/"))


def discard(folder, root):
    """Remove a folder under `root` in one step, as far as anyone reading can tell: it is moved aside first.

    A run folder taken apart file by file could lose the marker that holds it back before it loses its
    catalog, and would be public for that moment."""
    folder, root = folder.resolve(), root.resolve()
    if not folder.is_relative_to(root) or folder == root or not folder.is_dir():
        return
    aside = TRASH / f"{folder.name}-{uuid4().hex[:8]}"
    try:
        TRASH.mkdir(parents=True, exist_ok=True)
        os.replace(folder, aside)
    except OSError:
        aside = folder      # it could not be moved: remove it where it is
    shutil.rmtree(aside, ignore_errors=True)


def engine_command(job, python):
    options = job["options"]
    command = [python, "-m", "relentless.pipeline", str(UPLOADS / job["footage"]), "--out", str(RUNS / job["run"]),
               "--start", str(options["start"]), "--max-clips", str(options["max_clips"])]
    if options.get("seconds"):
        command += ["--seconds", str(options["seconds"])]
    if options["quality"] == "fast":
        command.append("--fast")
    if options.get("multiply"):
        command += ["--multiply", str(options["multiply"])]
    return command + ["--live"]   # the previews the job page shows


class JobError(Exception):
    """A job that cannot run, with a message meant for the admin page."""


def stop_process(process, sig=signal.SIGTERM):
    """Signal the engine and everything it started (its tracking child, ffmpeg)."""
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def run_job(job):
    python = engine_python()
    available = memory()[1]
    if available is not None and available < JOB_MEMORY:
        raise JobError(f"Only {available // MB} MB of memory is free on this machine right now, and a job needs about "
                       f"{JOB_MEMORY // MB} MB. Nothing was started; try again later.")
    run_dir = RUNS / job["run"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / library.HOLD).touch()   # its clips stay out of the catalog until the job is published
    env = {**os.environ, "RELENTLESS_DATA": str(STORE), "PYTHONUNBUFFERED": "1"}
    env.pop("ADMIN_TOKEN", None)   # the engine has no use for it
    if secret("ANTHROPIC_API_KEY"):   # the engine reads the project's .env itself; this covers a server's environment
        env["RELENTLESS_ANTHROPIC_API_KEY"] = secret("ANTHROPIC_API_KEY")
    command = engine_command(job, python)
    if shutil.which("nice"):   # leave the processor to whatever else the machine serves
        command = ["nice", "-n", "10", *command]
    starved, asked = False, None      # asked: when the engine was first told to stop
    with open(run_dir / "run.log", "w") as log:
        # Its own process group, so that stopping it also stops what it started.
        process = subprocess.Popen(command, cwd=ENGINE, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)
        with _lock:
            _running[job["id"]] = process
        while process.poll() is None:
            time.sleep(2)
            available = memory()[1]
            if available is not None and available < SCARCE_MEMORY and not starved:
                starved = True
                stop_process(process)
            if starved or job["status"] == "stopping":
                if asked is None:
                    asked = time.time()
                    stop_process(process)   # again: Stop may have been pressed before the process was known
                elif time.time() - asked > 10:
                    stop_process(process, signal.SIGKILL)   # it did not go when asked
        code = process.returncode
    with _lock:
        _running.pop(job["id"], None)
        if job["status"] == "stopping":
            job.update(status="stopped", error=None)
        elif starved:
            job.update(status="failed", error="Stopped: this machine was running out of memory, and the job gives way "
                                              "to whatever else runs here. Try a shorter stretch of footage.")
        elif code == 0:
            job.update(status="done", error=None)
        elif code in (-9, 137):
            job.update(status="failed", error="The engine was stopped by the system, most likely because it needed "
                                              "more memory than this machine allows. Try a shorter stretch of footage.")
        else:
            job.update(status="failed", error=log_tail(job, 4) or f"The engine exited with code {code}.")
        job["finished_at"] = now()
        save_jobs()


def here(job):
    return job["options"].get("where", "server") == "server"


def sweep():
    """A worker that went quiet has lost its job. Call with the lock held."""
    for job in _jobs:
        if job.get("worker") and job["status"] in ("running", "stopping") and time.time() - job.get("heard", 0) > WORKER_SILENCE:
            job.update(status="failed" if job["status"] == "running" else "stopped", finished_at=now(),
                       error=f"The worker {job['worker']} stopped reporting.")
            save_jobs()


def worker():
    """Runs this machine's queued jobs one after another: the engine wants the whole machine."""
    while True:
        job = None
        try:
            with _lock:
                sweep()
                job = next((job for job in _jobs if job["status"] == "queued" and here(job)), None)
                if job:
                    job.update(status="running", started_at=now(), device=os.environ.get("RELENTLESS_DEVICE") or "auto")
                    save_jobs()
            if not job:
                _wake.wait(5)
                _wake.clear()
                continue
            run_job(job)
        except Exception as error:   # no job, and no full disk, may end this thread: the queue would stop for good
            print(f"engine-jobs: {type(error).__name__}: {error}", flush=True)
            with _lock:
                if job and job["status"] in ("running", "stopping"):
                    job.update(status="failed", finished_at=now(),
                               error=str(error) if isinstance(error, JobError) else f"{type(error).__name__}: {error}")
                    try:
                        save_jobs()
                    except OSError:
                        pass
            time.sleep(1)


def safe_view(job):
    """A job as the list shows it. If its run cannot be read at all, it is still listed, so it can be deleted."""
    try:
        return job_view(job)
    except Exception as error:      # nothing about one job may take the list down
        print(f"jobs: {job.get('id')} could not be read: {type(error).__name__}: {error}", flush=True)
        blank = summary({})
        return {**{key: value for key, value in job.items() if key not in ("remote", "heard")}, **blank, "published": False,
                "error": job.get("error") or "What this job's run reported cannot be read."}


@router.get("/jobs")
def list_jobs():
    with _lock:
        sweep()
        jobs = [dict(job) for job in _jobs]
    return {"jobs": [safe_view(job) for job in reversed(jobs)]}


def find(job_id):
    """The job with this id. Call with the lock held."""
    job = next((job for job in _jobs if job["id"] == job_id), None)
    if not job:
        raise HTTPException(404, "No such job.")
    return job


@router.get("/jobs/{job_id}")
def job_detail(job_id: str, have: str = Query("", max_length=60)):
    """Everything the job page shows: the run's state, which previews exist, and the frame a stage is on now.

    `have` is the frame the page already holds (stage:number); the same frame is not sent twice."""
    with _lock:
        sweep()
        job = dict(find(job_id))
    folder, state = run_folder(job), run_state(job)
    # the frame a stage is on, while that stage is the one running (a worker's last frame outlives its stage here)
    at_work = {stage.get("key") for stage in rows(state.get("stages"))
               if isinstance(stage.get("key"), str) and stage.get("status") == "running"}
    now = read_json(folder / "live" / "now.json", READ_LIMITS["live/now.json"]) if job["status"] in ("running", "stopping") else {}
    if not isinstance(now.get("stage"), str) or now["stage"] not in at_work or number(now.get("seq")) is None:
        now = {}
    elif f"{now['stage']}:{now['seq']}" == have:
        now = {"stage": now["stage"], "seq": now["seq"], "same": True}
    return {"job": job_view(job), "state": shown_state(state),
            "kit": read_json(folder / "live" / "kit.json", READ_LIMITS["live/kit.json"]),
            "now": now or None, "files": versions(folder)}


@media_router.get("/jobs/{job_id}/file/{path:path}")
def job_file(job_id: str, path: str):
    """One file of the job's run: a preview, a clip, its labels, the engine's log."""
    with _lock:
        job = dict(find(job_id))
    rel = shown_path(path)
    if rel.as_posix() in ("state.json", "live/now.json", "live/kit.json"):
        raise HTTPException(404, "No such file.")      # the job page gets these read and trimmed, with the job itself
    for root in (RUNS / job["run"], LIVE / job["id"]):
        file = (root / rel).resolve()
        if file.is_relative_to(root.resolve()) and file.is_file():
            return FileResponse(file, headers={"Cache-Control": "private, max-age=3600"})
    raise HTTPException(404, "No such file.")


def set_published(job_id, on):
    with _lock:
        job = dict(find(job_id))
    folder = RUNS / job["run"]
    if job["status"] != "done" or not (folder / "catalog.json").is_file():
        raise HTTPException(409, "Only a finished job can be published.")
    # What goes public is what the catalog would make of the run's own list of clips, read the way the catalog
    # reads it: a clip whose record does not hold up is left out, and with none left there is nothing to publish.
    if on and not library.publishable(job["run"]):
        raise HTTPException(409, "This job has no accepted clips to publish.")
    hold = folder / library.HOLD
    if on:
        os.utime(folder / "catalog.json")   # its clips are the newest in the catalog from now
        if hold.is_dir():
            shutil.rmtree(hold)
        else:
            hold.unlink(missing_ok=True)
    elif not hold.exists():
        hold.touch()
    return job_view(job)


@router.post("/jobs/{job_id}/publish")
def publish_job(job_id: str):
    """Put the job's accepted clips, and their verified variants, into the public catalog."""
    return set_published(job_id, True)


@router.post("/jobs/{job_id}/unpublish")
def unpublish_job(job_id: str):
    """Take the job's clips out of the public catalog again. The job keeps them."""
    return set_published(job_id, False)


@router.post("/jobs", status_code=201)
def start_job(request: JobRequest):
    name = safe_name(request.footage)
    if not (UPLOADS / name).is_file():
        raise HTTPException(404, "No such footage. Upload it first.")
    if request.where == "server":   # a worker brings its own engine and key
        if not engine_python():
            raise HTTPException(503, "The engine is not installed on this machine. Run the job on a worker instead.")
        if not secret("ANTHROPIC_API_KEY"):
            raise HTTPException(503, "The engine needs ANTHROPIC_API_KEY to label clips. Set it on the server.")
        if request.multiply:
            raise HTTPException(422, "Object-swap variants are made by a worker with a fal key. Run this job on a worker, "
                                     "or leave variants off.")
    job_id = uuid4().hex[:8]
    job = {"id": job_id, "footage": name, "options": request.model_dump(exclude={"footage"}), "status": "queued",
           "run": f"{re.sub(r'[^A-Za-z0-9]+', '-', Path(name).stem)[:40].strip('-')}-{job_id}",
           "created_at": now(), "started_at": None, "finished_at": None, "error": None}
    with _lock:
        if sum(other["status"] in ("queued", "running", "stopping") for other in _jobs) >= 10:
            raise HTTPException(429, "Ten jobs are already waiting. Let some finish first.")
        # Held back from the first moment: wherever the engine writes this run (here, or a worker that shares
        # this machine's data folder), its clips stay out of the catalog until the job is published.
        (RUNS / job["run"]).mkdir(parents=True, exist_ok=True)
        (RUNS / job["run"] / library.HOLD).touch()
        _jobs.append(job)
        save_jobs()
    _wake.set()
    return job_view(job)


@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    with _lock:
        job = next((job for job in _jobs if job["id"] == job_id), None)
        if not job:
            raise HTTPException(404, "No such job.")
        if job["status"] == "queued":
            job.update(status="stopped", finished_at=now())
        elif job["status"] == "running":
            job["status"] = "stopping"   # a worker hears of it with its next report; the loop in run_job sees it too
            if job_id in _running:
                stop_process(_running[job_id])
        save_jobs()
        seen = dict(job)
    return job_view(seen)      # reads the run's state: not while holding the lock every other route waits on


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Forget a finished job and remove what it produced, including its clips in the catalog."""
    with _lock:
        job = next((job for job in _jobs if job["id"] == job_id), None)
        if not job:
            raise HTTPException(404, "No such job.")
        if job["status"] in ("queued", "running", "stopping"):
            raise HTTPException(409, "Stop the job first.")
        _jobs.remove(job)
        save_jobs()
        _previews.pop(job["id"], None)
    discard(RUNS / job["run"], RUNS)
    discard(LIVE / job["id"], LIVE)
    return {"deleted": job_id}


# ---------- workers ----------

class Fal(BaseModel):
    """What a worker can spend on object-swap variants: its own key, and the cap its engine keeps to."""
    key: bool = False
    spent_usd: float = Field(0, ge=0, le=1e6, allow_inf_nan=False)
    cap_usd: float = Field(0, ge=0, le=1e6, allow_inf_nan=False)


class Hello(BaseModel):
    worker: str = Field(min_length=1, max_length=80)
    device: str = Field("", max_length=20)
    labeler: str = Field("", max_length=60)
    fal: Optional[Fal] = None


class Progress(BaseModel):
    """A worker's short report on its run. It is kept with the job and sent on to the page, so every part of it
    has a kind: a number that is not one (NaN) or a list where a name belongs would break the jobs list later."""
    stages: list[dict] = Field(default_factory=list, max_length=20)
    clips: dict = Field(default_factory=dict)
    claude_usd: Optional[float] = Field(None, allow_inf_nan=False)
    source: dict = Field(default_factory=dict)
    log: str = Field("", max_length=6000)

    def kept(self):
        counts, window = record(self.clips), record(self.source)
        count = lambda value: int(value) if number(value) is not None and 0 <= value < 1e9 else 0      # noqa: E731
        return {"stages": [stage_row(stage) for stage in self.stages if isinstance(stage.get("key"), str)],
                "clips": {key: count(counts.get(key)) for key in ("planned", "accepted", "rejected")},
                "claude_usd": self.claude_usd, "log": self.log,
                "source": {key: number(window.get(key)) for key in ("start_s", "seconds")}}


class Failure(BaseModel):
    error: str = Field("", max_length=3000)


def worker_job(job_id):
    """The job with this id, if a worker has it. Call with the lock held."""
    job = next((job for job in _jobs if job["id"] == job_id), None)
    if not job or not job.get("worker"):
        raise HTTPException(404, "No such worker job.")
    return job


def heard(job):
    """A worker said something about its job: the job is alive, and so is the worker. Call with the lock held."""
    job["heard"] = time.time()
    # It asks for work only between jobs; while it has one, this keeps it on the list of workers (also after
    # this site restarted in the middle of its job, when the list starts out empty).
    seen = _workers.setdefault(job["worker"], {"device": job.get("device") or "", "labeler": "", "fal": None})
    seen["at"] = time.time()


@router.post("/worker/next")
def worker_next(hello: Hello):
    """A worker asks for work: the oldest job that is waiting for one, or nothing."""
    with _lock:
        _workers[hello.worker] = {"at": time.time(), "device": hello.device, "labeler": hello.labeler,
                                  "fal": hello.fal.model_dump() if hello.fal else None}
        sweep()
        job = next((job for job in _jobs if job["status"] == "queued" and not here(job)), None)
        if job:
            job.update(status="running", started_at=now(), worker=hello.worker, device=hello.device, heard=time.time())
            _previews.pop(job["id"], None)
            save_jobs()
            discard(LIVE / job["id"], LIVE)      # nothing of an earlier attempt is shown as this one's
        return {"job": {key: job[key] for key in ("id", "footage", "run", "options")} if job else None}


@router.post("/jobs/{job_id}/progress")
def worker_progress(job_id: str, progress: Progress):
    """How the worker's run is going. The reply tells it whether the job was stopped from the admin page."""
    with _lock:
        job = worker_job(job_id)
        job["remote"] = progress.kept()
        heard(job)
        return {"stop": job["status"] != "running"}


@router.post("/jobs/{job_id}/failed")
def worker_failed(job_id: str, failure: Failure):
    with _lock:
        job = worker_job(job_id)
        if job["status"] in ("running", "stopping"):
            job.update(status="stopped" if job["status"] == "stopping" else "failed", finished_at=now(),
                       error=None if job["status"] == "stopping" else failure.error or "The worker could not finish this job.")
            save_jobs()
        seen = dict(job)
    return job_view(seen)


_previews = {}                   # job -> [bytes, files] of its previews here
_previewing = threading.Lock()


def measure(folder):
    files = [path for path in folder.rglob("*") if path.is_file()] if folder.is_dir() else []
    return [sum(path.stat().st_size for path in files), len(files)]


def keep_preview(job_id, rel, body):
    """Store one file of a worker's run in progress, replacing the last copy in one step.

    What a job may keep here is counted in bytes and in files (an empty file still takes a place on the disk),
    and so is what all jobs keep together."""
    folder = LIVE / job_id
    target = folder / rel
    with _previewing:
        size, files = _previews.get(job_id) or measure(folder)
        try:
            old = target.stat().st_size if target.is_file() else None
        except OSError:
            old = None
        size, files = size - (old or 0) + len(body), files + (old is None)
        others = sum(entry[0] for job, entry in _previews.items() if job != job_id)
        if (size > MAX_PREVIEWS or files > MAX_PREVIEW_FILES or size + others > MAX_ALL_PREVIEWS
                or shutil.disk_usage(STORE).free < len(body) + FREE_MARGIN):
            raise HTTPException(507, "No room for more previews of this job.")
        part = target.with_name(f".{target.name}.part")
        try:
            if target.is_dir():
                raise IsADirectoryError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            part.write_bytes(body)
            os.replace(part, target)
        except OSError as error:      # the name is taken by a folder, or lies under a file
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass                  # there was nowhere to write it in the first place
            raise HTTPException(422, "That file cannot be kept under that name.") from error
        _previews[job_id] = [size, files]


@router.put("/jobs/{job_id}/live")
async def worker_preview(request: Request, job_id: str, path: str = Query(..., max_length=200)):
    """One file of a worker's run as it stands: its state, the frame a stage is on, a preview, a small copy of a clip.

    The job page reads these until the finished run arrives. They are never part of the catalog."""
    with _lock:
        job = worker_job(job_id)
        if job["status"] not in ("running", "stopping"):
            raise HTTPException(409, "This job is not running any more.")
        heard(job)
    rel = shown_path(path)
    limit = READ_LIMITS.get(rel.as_posix(), MAX_PREVIEW)
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:      # said to be too large: not read at all
        raise HTTPException(413, "That file is too large for a preview.")
    body = await request.body()
    if not body or len(body) > limit:
        raise HTTPException(413 if body else 422, "That file is empty or too large for a preview.")
    await run_in_threadpool(keep_preview, job_id, rel, body)
    return {"kept": rel.as_posix()}


def unpack(archive, run_dir):
    """Take a worker's result apart into the run folder: plain media and label files only, nothing outside it.

    The files go to a staging folder first and the folder is moved into place at the end, so a result that
    turns out to be damaged never shows up half-published. A plain tar cannot hold more than its own size;
    one that claims to (sparse members can) is refused. So is one with more files than a run has (an empty
    file costs disk too), and one whose state or catalog is larger than this site reads."""
    if shutil.disk_usage(STORE).free < archive.stat().st_size + FREE_MARGIN:      # the archive and its files, side by side
        raise tarfile.TarError("not enough free disk space to unpack it")
    staging = INCOMING / f"{run_dir.name}.unpacked"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    root, budget, files = staging.resolve(), archive.stat().st_size, 0
    try:
        with tarfile.open(archive, "r:") as tar:   # plain tar only: a compressed one could unpack to any size
            for seen, member in enumerate(tar):
                if seen >= 20 * MAX_RESULT_FILES:      # going through an endless list of entries is work too
                    raise tarfile.TarError("the archive has too many entries")
                if not member.isfile() or member.issparse() or not plain(member.name):
                    continue
                target = (root / member.name).resolve()
                if not target.is_relative_to(root):
                    continue
                files, budget = files + 1, budget - member.size
                if budget < 0:
                    raise tarfile.TarError("the archive unpacks to more than its own size")
                if files > MAX_RESULT_FILES:
                    raise tarfile.TarError(f"the archive holds more than {MAX_RESULT_FILES} files")
                if member.size > READ_LIMITS.get(member.name, MAX_RESULT):
                    raise tarfile.TarError(f"{member.name} is larger than this site reads")
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, open(target, "wb") as file:
                    shutil.copyfileobj(source, file)
        if not (staging / "catalog.json").is_file():
            raise tarfile.TarError("the archive holds no catalog.json")
        (staging / library.HOLD).touch()   # its clips stay out of the catalog until the job is published
        discard(run_dir, run_dir.parent)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, run_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@router.put("/jobs/{job_id}/result")
async def worker_result(request: Request, job_id: str, total: int = Query(..., ge=1, le=MAX_RESULT),
                        offset: int = Query(0, ge=0)):
    """One piece of the finished run, as a tar archive. With the last piece the clips join the catalog."""
    with _lock:
        job = worker_job(job_id)
        if job["status"] != "running":
            raise HTTPException(409, "This job is not running any more.")
        heard(job)
        run = job["run"]
    part = INCOMING / f"{job_id}.tar.part"
    arrived = await take_piece(request, part, total, offset)
    if arrived < total:
        return {"done": False, "offset": arrived}
    try:
        await run_in_threadpool(unpack, part, RUNS / run)   # off the event loop: the site keeps answering
    except (tarfile.TarError, OSError) as error:
        raise HTTPException(422, f"The result could not be read: {error}") from error
    finally:
        part.unlink(missing_ok=True)
    with _lock:
        if job["status"] != "running" or job not in _jobs:   # stopped or removed while the result was arriving
            if job not in _jobs:
                discard(RUNS / run, RUNS)
            raise HTTPException(409, "This job is not running any more.")
        job.update(status="done", finished_at=now(), error=None)
        job.pop("remote", None)
        save_jobs()
        _previews.pop(job_id, None)
        seen = dict(job)
    discard(LIVE / job_id, LIVE)   # the run folder has it all now
    return {"done": True, "job": job_view(seen)}


load_jobs()
threading.Thread(target=worker, daemon=True, name="engine-jobs").start()
