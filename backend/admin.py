"""Admin API: upload raw footage and run the engine on it. Every route needs the admin token.

Footage is uploaded in pieces to <store>/uploads/. A job runs the engine (engine/) on one upload as a
separate process, one job at a time, and writes a run folder under <store>/runs/. When the run
finishes, its accepted clips appear in the catalog on their own (see library.py).
"""
import hmac
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import library
from .catalog import secret
from .library import ROOT, RUNS, STORE

UPLOADS = STORE / "uploads"
PARTIAL = UPLOADS / ".partial"
JOBS_FILE = STORE / "jobs.json"
ENGINE = ROOT / "engine"
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
MB = 1024 * 1024
MAX_UPLOAD = int(os.environ.get("PEELSET_MAX_UPLOAD_MB", "2048")) * MB
CHUNK = 8 * MB            # the admin page sends a file in pieces of this size
MAX_CHUNK = 32 * MB
FREE_MARGIN = 500 * MB    # never fill the disk to the last byte: other things live on it

def admin(authorization: str = Header("")):
    token = secret("ADMIN_TOKEN") or ""
    if not token:
        raise HTTPException(503, "The admin panel is off. Set ADMIN_TOKEN on the server to turn it on.")
    given = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    if not hmac.compare_digest(given.encode(), token.encode()):
        raise HTTPException(401, "That admin token is not right.")


router = APIRouter(prefix="/api/admin", dependencies=[Depends(admin)])


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
    return {
        "engine": {"ready": bool(python), "device": os.environ.get("RELENTLESS_DEVICE") or "auto",
                   "claude_key": bool(secret("ANTHROPIC_API_KEY")), "claude_usd": ledger_usd()},
        "storage": {"path": str(STORE), "kind": storage_kind(), "free_bytes": disk.free, "total_bytes": disk.total},
        "machine": {"cpus": os.cpu_count(), "arch": platform.machine(), "system": platform.system(),
                    "memory_bytes": total, "memory_available_bytes": available},
        "library": {"clips": len(clips), "processed_here": sum(clip["origin"] == "processed here" for clip in clips)},
        "limits": {"upload_bytes": MAX_UPLOAD, "chunk_bytes": CHUNK},
    }


# ---------- footage ----------

def safe_name(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).name).strip("-.")
    if not name or Path(name).suffix.lower() not in VIDEO_TYPES:
        raise HTTPException(422, f"Upload a video file: {', '.join(sorted(VIDEO_TYPES))}.")
    return name[:120]


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
                if length and size:
                    hours, minutes, seconds = length.groups()
                    info = {"seconds": round(int(hours) * 3600 + int(minutes) * 60 + float(seconds), 1),
                            "width": int(size.group(1)), "height": int(size.group(2)),
                            "fps": round(float(rate.group(1)), 2) if rate else None}
                else:
                    info = {"unreadable": True}
            except (OSError, subprocess.SubprocessError):
                pass
        _probed[key] = info
    return _probed[key]


def footage_entry(path):
    stat = path.stat()
    return {"name": path.name, "bytes": stat.st_size,
            "uploaded_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(), **probe(path)}


@router.get("/footage")
def list_footage():
    files = [path for path in UPLOADS.glob("*") if path.is_file() and path.suffix.lower() in VIDEO_TYPES]
    return {"footage": [footage_entry(path) for path in sorted(files, key=lambda path: -path.stat().st_mtime)]}


@router.put("/footage")
async def upload(request: Request, name: str = Query(..., max_length=300), total: int = Query(..., ge=1),
                 offset: int = Query(0, ge=0)):
    """One piece of a file. `offset` is where the piece starts; the reply says how much has arrived."""
    name = safe_name(name)
    if total > MAX_UPLOAD:
        raise HTTPException(413, f"Files can be up to {MAX_UPLOAD // MB} MB.")
    PARTIAL.mkdir(parents=True, exist_ok=True)
    part = PARTIAL / f"{name}.part"
    have = part.stat().st_size if part.exists() else 0
    if offset == 0:
        if shutil.disk_usage(STORE).free < total + FREE_MARGIN:
            raise HTTPException(507, "Not enough free disk space for this file.")
        have = 0
    elif offset != have:
        raise HTTPException(409, f"Expected the piece starting at byte {have}.")
    piece = await request.body()
    if not piece or len(piece) > MAX_CHUNK or offset + len(piece) > total:
        raise HTTPException(413 if piece else 422, "That piece is empty or too large.")
    with open(part, "r+b" if offset else "wb") as file:
        file.seek(offset)
        file.write(piece)
    arrived = offset + len(piece)
    if arrived < total:
        return {"done": False, "offset": arrived}
    final = UPLOADS / name
    os.replace(part, final)
    entry = footage_entry(final)
    if entry.get("unreadable"):
        final.unlink()
        raise HTTPException(422, "That file does not look like a video ffmpeg can read.")
    return {"done": True, "footage": entry}


@router.delete("/footage/{name}")
def delete_footage(name: str):
    path = UPLOADS / safe_name(name)
    with _lock:
        if any(job["footage"] == path.name and job["status"] in ("queued", "running") for job in _jobs):
            raise HTTPException(409, "This footage is being processed. Stop the job first.")
    if not path.is_file():
        raise HTTPException(404, "No such footage.")
    path.unlink()
    return {"deleted": path.name}


# ---------- jobs ----------

class JobRequest(BaseModel):
    footage: str = Field(max_length=300)
    start: float = Field(0, ge=0, le=86400)
    seconds: Optional[float] = Field(None, gt=0, le=3600)       # None: to the end of the video
    max_clips: int = Field(4, ge=1, le=24)
    quality: Literal["fast", "full"] = "fast"                   # fast: 720p clips; full: 1080p, about twice as slow


_lock = threading.Lock()
_wake = threading.Event()
_jobs = []        # oldest first; kept in JOBS_FILE
_running = {}     # job id -> its process


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
        if job["status"] == "running":   # this process is new, so whatever ran before is gone
            job.update(status="failed", finished_at=now(), error="The server restarted while this was running.")
    save_jobs()


def run_state(job):
    try:
        return json.loads((RUNS / job["run"] / "state.json").read_text()) if job.get("run") else {}
    except (OSError, ValueError):
        return {}


def log_tail(job, lines=12):
    try:
        text = (RUNS / job["run"] / "run.log").read_text(errors="replace")
    except (OSError, KeyError, TypeError):
        return ""
    keep = [line for line in text.splitlines() if line.strip() and "Warning" not in line]
    return "\n".join(keep[-lines:])


def job_view(job):
    """A job with the live picture of its run: stages, clip counts and what it has cost."""
    state = run_state(job)
    clips = state.get("clips", [])
    view = {**job,
            "stages": [{key: stage.get(key) for key in ("key", "title", "status", "progress", "detail")}
                       for stage in state.get("stages", [])],
            "clips": {"planned": len(clips), "accepted": sum(clip["status"] == "accepted" for clip in clips),
                      "rejected": sum(clip["status"] == "rejected" for clip in clips)},
            "claude_usd": (state.get("costs") or {}).get("anthropic_usd"),
            "source": {key: (state.get("source") or {}).get(key) for key in ("start_s", "seconds")}}
    if job["status"] in ("running", "failed", "stopped"):
        view["log"] = log_tail(job)
    return view


def engine_command(job, python):
    options = job["options"]
    command = [python, "-m", "relentless.pipeline", str(UPLOADS / job["footage"]), "--out", str(RUNS / job["run"]),
               "--start", str(options["start"]), "--max-clips", str(options["max_clips"])]
    if options.get("seconds"):
        command += ["--seconds", str(options["seconds"])]
    if options["quality"] == "fast":
        command.append("--fast")
    return command


def run_job(job):
    python = engine_python()
    run_dir = RUNS / job["run"]
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "RELENTLESS_DATA": str(STORE), "PYTHONUNBUFFERED": "1"}
    if secret("ANTHROPIC_API_KEY"):   # the engine reads the project's .env itself; this covers a server's environment
        env["RELENTLESS_ANTHROPIC_API_KEY"] = secret("ANTHROPIC_API_KEY")
    command = engine_command(job, python)
    if shutil.which("nice"):   # leave the processor to whatever else the machine serves
        command = ["nice", "-n", "10", *command]
    with open(run_dir / "run.log", "w") as log:
        process = subprocess.Popen(command, cwd=ENGINE, stdout=log, stderr=subprocess.STDOUT, env=env)
        with _lock:
            _running[job["id"]] = process
        code = process.wait()
    with _lock:
        _running.pop(job["id"], None)
        if job["status"] == "stopping":
            job.update(status="stopped", error=None)
        elif code == 0:
            job.update(status="done", error=None)
        else:
            job.update(status="failed", error=log_tail(job, 4) or f"The engine exited with code {code}.")
        job["finished_at"] = now()
        save_jobs()


def worker():
    """Runs queued jobs one after another: the engine wants the whole machine."""
    while True:
        with _lock:
            job = next((job for job in _jobs if job["status"] == "queued"), None)
            if job:
                job.update(status="running", started_at=now())
                save_jobs()
        if not job:
            _wake.wait(5)
            _wake.clear()
            continue
        try:
            run_job(job)
        except Exception as error:   # a job must never take the queue down with it
            with _lock:
                job.update(status="failed", finished_at=now(), error=f"{type(error).__name__}: {error}")
                save_jobs()


@router.get("/jobs")
def list_jobs():
    with _lock:
        jobs = [dict(job) for job in _jobs]
    return {"jobs": [job_view(job) for job in reversed(jobs)]}


@router.post("/jobs", status_code=201)
def start_job(request: JobRequest):
    name = safe_name(request.footage)
    if not (UPLOADS / name).is_file():
        raise HTTPException(404, "No such footage. Upload it first.")
    if not engine_python():
        raise HTTPException(503, "The engine is not installed on this machine, so footage can be stored but not processed.")
    if not secret("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "The engine needs ANTHROPIC_API_KEY to label clips. Set it on the server.")
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
        elif job["status"] == "running" and job_id in _running:
            job["status"] = "stopping"
            _running[job_id].terminate()
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
    run_dir = (RUNS / job["run"]).resolve()
    if run_dir.is_relative_to(RUNS.resolve()) and run_dir != RUNS.resolve() and run_dir.is_dir():
        shutil.rmtree(run_dir)
    return {"deleted": job_id}


load_jobs()
threading.Thread(target=worker, daemon=True, name="engine-jobs").start()
