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
import hmac
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
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

from . import library
from .catalog import secret
from .library import ROOT, RUNS, STORE

UPLOADS = STORE / "uploads"
PARTIAL = UPLOADS / ".partial"
JOBS_FILE = STORE / "jobs.json"
INCOMING = STORE / "incoming"     # results a worker is still sending
LIVE = STORE / "live"             # a worker's run as it stands (state, previews), until its result arrives
FILM = UPLOADS / ".film"          # one strip of small frames per upload
RESULT_TYPES = {".mp4", ".jpg", ".json", ".log"}
MAX_PREVIEW = 8 * 1024 * 1024         # one file of a worker's run in progress
MAX_PREVIEWS = 400 * 1024 * 1024      # all of them, per job
NAME_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}")
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


def media_pass(token):
    """What the admin page's browser holds, as a cookie, to load pictures and videos: an <img> or a <video>
    cannot send the token. It is made from the token, opens the two read-only routes below and nothing else,
    and stops working when the token changes."""
    return hmac.new(token.encode(), b"peelset-media-v1", "sha256").hexdigest()


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
    response.delete_cookie(MEDIA_COOKIE, path="/api/admin")
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


def film_strip(path):
    """Ten frames of a video side by side in one small picture, made once per upload. Returns its path, or None.

    Each frame is 128 pixels wide: enough to tell one piece of footage from another, and far too little to
    tell who is in it. (No face has been blurred at this point: the engine does that when it runs.)"""
    strip = FILM / f"{path.name}.jpg"
    if strip.is_file() and strip.stat().st_mtime >= path.stat().st_mtime:
        return strip
    tool, seconds = ffmpeg(), probe(path).get("seconds")
    if not tool or not seconds:
        return None
    command = [tool, "-hide_banner", "-loglevel", "error", "-y"]
    for index in range(10):
        command += ["-ss", f"{seconds * (index + 0.5) / 10:.2f}", "-i", str(path)]
    small = ";".join(f"[{index}:v]scale=128:72,setsar=1[v{index}]" for index in range(10))
    command += ["-filter_complex", f"{small};{''.join(f'[v{index}]' for index in range(10))}hstack=inputs=10[out]",
                "-map", "[out]", "-frames:v", "1", "-q:v", "5", str(strip)]
    FILM.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command, capture_output=True, timeout=90)
    except (OSError, subprocess.SubprocessError):
        pass
    return strip if strip.is_file() else None


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
    for folder in LIVE.glob("*") if LIVE.is_dir() else []:   # previews of jobs that are gone
        if folder.name not in known:
            shutil.rmtree(folder, ignore_errors=True)


def run_folder(job):
    """Where a job's run can be read: its run folder, or the copy a worker keeps here while its run goes on."""
    folder = RUNS / job["run"]
    return folder if (folder / "state.json").is_file() or not (LIVE / job["id"]).is_dir() else LIVE / job["id"]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def run_state(job):
    state = read_json(run_folder(job) / "state.json") if job.get("run") else {}
    return state if isinstance(state, dict) else {}


def log_tail(job, lines=12):
    try:
        text = (RUNS / job["run"] / "run.log").read_text(errors="replace")
    except (OSError, KeyError, TypeError):
        return ""
    keep = [line for line in text.splitlines() if line.strip() and "Warning" not in line]
    return "\n".join(keep[-lines:])


def rows(value):
    """The dicts in a list from a state file. A state can come from a worker: it is read, never trusted."""
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def record(value):
    return value if isinstance(value, dict) else {}


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def summary(state):
    """The short picture of a run: stages, clip counts, what it cost. A worker reports the first of these itself."""
    clips, costs, source = rows(state.get("clips")), record(state.get("costs")), record(state.get("source"))
    size = source.get("size")
    labeler = record(state.get("engine")).get("labeler")
    return {"stages": [{key: stage.get(key) for key in ("key", "title", "status", "progress", "detail", "seconds")}
                       for stage in rows(state.get("stages"))],
            "clips": {"planned": len(clips), "accepted": sum(clip.get("status") == "accepted" for clip in clips),
                      "rejected": sum(clip.get("status") == "rejected" for clip in clips)},
            "claude_usd": number(costs.get("anthropic_usd")), "fal_usd": number(costs.get("fal_usd")),
            "variants": sum(bool(variant.get("passed")) for clip in clips for variant in rows(clip.get("variants"))),
            "labeler": labeler if isinstance(labeler, str) else None, "elapsed": number(state.get("elapsed")),
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
        _pictures[path] = (stamp, summary(run_state(job)))
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
    view.update(seen if seen["stages"] else {key: remote.get(key, seen[key]) for key in seen})
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


def versions(folder):
    """Every file of a run the job page may ask for, with a number that changes when the file does."""
    found = {}
    if folder.is_dir():
        for path in folder.rglob("*"):
            rel = path.relative_to(folder)
            if path.suffix.lower() in RESULT_TYPES and not any(part.startswith(".") for part in rel.parts) and path.is_file():
                found[rel.as_posix()] = path.stat().st_mtime_ns // 1_000_000
    return found


def shown_path(rel):
    """A path inside a run folder, as the job page or a worker names it: plain parts, a known kind of file."""
    parts = rel.split("/")
    if not 1 <= len(parts) <= 5 or not all(NAME_PART.fullmatch(part) for part in parts) \
            or Path(rel).suffix.lower() not in RESULT_TYPES:
        raise HTTPException(404, "No such file.")
    return Path(*parts)


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


@router.get("/jobs")
def list_jobs():
    with _lock:
        sweep()
        jobs = [dict(job) for job in _jobs]
    return {"jobs": [job_view(job) for job in reversed(jobs)]}


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
    at_work = {stage.get("key") for stage in rows(state.get("stages")) if stage.get("status") == "running"}
    now = read_json(folder / "live" / "now.json") if job["status"] in ("running", "stopping") else {}
    if not isinstance(now, dict) or now.get("stage") not in at_work:
        now = {}
    elif f"{now.get('stage')}:{now.get('seq')}" == have:
        now = {"stage": now.get("stage"), "seq": now.get("seq"), "same": True}
    return {"job": job_view(job), "state": shown_state(state), "kit": record(read_json(folder / "live" / "kit.json")),
            "now": now or None, "files": versions(folder)}


@media_router.get("/jobs/{job_id}/file/{path:path}")
def job_file(job_id: str, path: str):
    """One file of the job's run: a preview, a clip, its labels, the engine's log."""
    with _lock:
        job = dict(find(job_id))
    rel = shown_path(path)
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
    if on and not picture(job)["clips"]["accepted"]:
        raise HTTPException(409, "This job has no accepted clips to publish.")
    if on:
        (folder / library.HOLD).unlink(missing_ok=True)
        os.utime(folder / "catalog.json")   # its clips are the newest in the catalog from now
    else:
        (folder / library.HOLD).touch()
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
        if sum(other["status"] in ("queued", "running") for other in _jobs) >= 10:
            raise HTTPException(429, "Ten jobs are already waiting. Let some finish first.")
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
        return job_view(dict(job))


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
    for root, name in ((RUNS, job["run"]), (LIVE, job["id"])):
        folder = (root / name).resolve()
        if folder.is_relative_to(root.resolve()) and folder != root.resolve() and folder.is_dir():
            shutil.rmtree(folder)
    return {"deleted": job_id}


# ---------- workers ----------

class Fal(BaseModel):
    """What a worker can spend on object-swap variants: its own key, and the cap its engine keeps to."""
    key: bool = False
    spent_usd: float = Field(0, ge=0, le=1e6)
    cap_usd: float = Field(0, ge=0, le=1e6)


class Hello(BaseModel):
    worker: str = Field(min_length=1, max_length=80)
    device: str = Field("", max_length=20)
    labeler: str = Field("", max_length=60)
    fal: Optional[Fal] = None


class Progress(BaseModel):
    stages: list = Field(default_factory=list, max_length=20)
    clips: dict = Field(default_factory=dict)
    claude_usd: Optional[float] = None
    source: dict = Field(default_factory=dict)
    log: str = Field("", max_length=6000)


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
    if job["worker"] in _workers:      # it asks for work only between jobs; while it has one, this keeps it listed
        _workers[job["worker"]]["at"] = time.time()


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
            shutil.rmtree(LIVE / job["id"], ignore_errors=True)
            save_jobs()
        return {"job": {key: job[key] for key in ("id", "footage", "run", "options")} if job else None}


@router.post("/jobs/{job_id}/progress")
def worker_progress(job_id: str, progress: Progress):
    """How the worker's run is going. The reply tells it whether the job was stopped from the admin page."""
    with _lock:
        job = worker_job(job_id)
        job["remote"] = progress.model_dump()
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
        return job_view(dict(job))


def keep_preview(folder, rel, body):
    """Store one file of a worker's run in progress, replacing the last copy in one step."""
    target = folder / rel
    held = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file() and path != target) if folder.is_dir() else 0
    if held + len(body) > MAX_PREVIEWS or shutil.disk_usage(STORE).free < len(body) + FREE_MARGIN:
        raise HTTPException(507, "No room for more previews of this job.")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(f".{target.name}.part")
    part.write_bytes(body)
    os.replace(part, target)


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
    body = await request.body()
    if not body or len(body) > MAX_PREVIEW:
        raise HTTPException(413 if body else 422, "That file is empty or too large for a preview.")
    await run_in_threadpool(keep_preview, LIVE / job_id, rel, body)
    return {"kept": rel.as_posix()}


def unpack(archive, run_dir):
    """Take a worker's result apart into the run folder: plain media and label files only, nothing outside it.

    The files go to a staging folder first and the folder is moved into place at the end, so a result that
    turns out to be damaged never shows up half-published. A plain tar cannot hold more than its own size;
    one that claims to (sparse members can) is refused."""
    staging = INCOMING / f"{run_dir.name}.unpacked"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    root, budget = staging.resolve(), archive.stat().st_size
    try:
        with tarfile.open(archive, "r:") as tar:   # plain tar only: a compressed one could unpack to any size
            for member in tar:
                target = (root / member.name).resolve()
                if (not member.isfile() or member.issparse() or not target.is_relative_to(root)
                        or target.suffix.lower() not in RESULT_TYPES):
                    continue
                budget -= member.size
                if budget < 0:
                    raise tarfile.TarError("the archive unpacks to more than its own size")
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, open(target, "wb") as file:
                    shutil.copyfileobj(source, file)
        if not (staging / "catalog.json").is_file():
            raise tarfile.TarError("the archive holds no catalog.json")
        (staging / library.HOLD).touch()   # its clips stay out of the catalog until the job is published
        shutil.rmtree(run_dir, ignore_errors=True)
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
            shutil.rmtree(RUNS / run, ignore_errors=True)
            raise HTTPException(409, "This job is not running any more.")
        job.update(status="done", finished_at=now(), error=None)
        job.pop("remote", None)
        save_jobs()
    shutil.rmtree(LIVE / job_id, ignore_errors=True)   # the run folder has it all now
    return {"done": True, "job": job_view(dict(job))}


load_jobs()
threading.Thread(target=worker, daemon=True, name="engine-jobs").start()
