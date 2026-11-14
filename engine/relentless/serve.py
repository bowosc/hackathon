"""Local dashboard for the engine: watch a run live, browse past runs, start a new one.

  uv run python -m relentless.serve            # then open http://127.0.0.1:8765

The page polls data/runs/<name>/state.json while a run is going. "New run" starts the pipeline
as a subprocess of this server (one at a time: there is one GPU).
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from . import ledger, media, pipeline, screen
from .config import CLAUDE_MODEL, CLAUDE_PRICES, DATA, FAL_BUDGET_USD, FAL_MODELS, REPO, RUNS

VIEWER = Path(__file__).parent / "viewer" / "index.html"
UPLOADS = DATA / "uploads"
VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".m4v", ".webm"}
TYPES = {".html": "text/html; charset=utf-8", ".json": "application/json", ".mp4": "video/mp4",
         ".jpg": "image/jpeg", ".png": "image/png", ".js": "text/javascript", ".css": "text/css",
         ".log": "text/plain; charset=utf-8"}
MAX_UPLOAD = 4 << 30

# Tokens per call (input, output) measured on the MEVA clips. Used to estimate a model's cost before it has
# any calls in the ledger.
TYPICAL_TOKENS = {"label": (10_600, 2_000), "screen": (2_700, 260), "verify": (4_200, 500)}

_active: dict = {"proc": None, "name": None}
_lock = threading.Lock()
_video_cache: dict[tuple, dict] = {}


def list_runs() -> list[dict]:
    out = []
    for p in sorted(RUNS.glob("*/state.json"), key=lambda p: -p.stat().st_mtime):
        try:
            s = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        clips = s.get("clips", [])
        out.append({"name": s.get("name", p.parent.name), "status": s.get("status"), "started": s.get("started"),
                    "source": s.get("source", {}).get("file"), "start_s": s.get("source", {}).get("start_s"),
                    "seconds": s.get("source", {}).get("seconds"),
                    "accepted": sum(1 for c in clips if c.get("status") == "accepted"), "clips": len(clips)})
    return out


def list_videos() -> list[dict]:
    groups = [("Raw CCTV", DATA / "raw"), ("Uploads", UPLOADS), ("Highlights", DATA / "highlights")]
    out = []
    for group, root in groups:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in VIDEO_EXT or not p.is_file():
                continue
            key = (str(p), p.stat().st_mtime_ns)
            if key not in _video_cache:
                cap = cv2.VideoCapture(str(p))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                _video_cache[key] = {
                    "path": str(p.relative_to(DATA)), "name": p.name, "group": group,
                    "seconds": round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps, 1),
                    "size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
                    "mb": round(p.stat().st_size / 1e6, 1)}
                cap.release()
            out.append(_video_cache[key])
    return out


def claude_costs() -> dict:
    """USD per label, per screening sheet (six candidates) and per variant check on the configured model:
    the ledger's average for that model, or an estimate from typical token counts if it has no calls yet."""
    models = {"label": CLAUDE_MODEL, "screen": screen.SCREEN_MODEL or CLAUDE_MODEL, "verify": CLAUDE_MODEL}
    paid = [r for r in ledger.rows() if r["service"] == "anthropic"]
    out = {"model": CLAUDE_MODEL}
    for kind, model in models.items():
        seen = [r["usd"] for r in paid if r.get("model") == model and str(r.get("purpose", "")).split(":")[0] == kind]
        if seen:
            out[kind] = sum(seen) / len(seen)
        else:
            p_in, p_out, _ = CLAUDE_PRICES.get(model, CLAUDE_PRICES["claude-opus-5-5"])
            tok_in, tok_out = TYPICAL_TOKENS[kind]
            out[kind] = (tok_in * p_in + tok_out * p_out) / 1e6
    return out


def catalog() -> list[dict]:
    """Every sellable clip from every finished run, with URLs this server can play."""
    items = []
    for p in sorted(RUNS.glob("*/catalog.json")):
        try:
            for it in json.loads(p.read_text()):
                it["files"] = {k: f"/runs/{p.parent.name}/{v}" for k, v in it.get("files", {}).items()}
                items.append(it)
        except (OSError, json.JSONDecodeError):
            continue
    return items


def inside(root: Path, rel: str) -> Path | None:
    """Resolve rel under root; None if it escapes."""
    p = (root / rel).resolve()
    return p if p.is_relative_to(root.resolve()) else None


def running() -> str | None:
    with _lock:
        proc = _active["proc"]
        if proc is not None and proc.poll() is None:
            return _active["name"]
        return None


def start_run(body: dict) -> tuple[int, dict]:
    video = inside(DATA, str(body.get("video", "")))
    if video is None or not video.is_file() or video.suffix.lower() not in VIDEO_EXT:
        return 400, {"error": "unknown video"}
    busy = running()
    if busy:
        return 409, {"error": f"run {busy} is still going", "name": busy}
    with media.Reader(video) as rd:
        duration = rd.count / rd.fps
    start = max(0.0, float(body.get("start") or 0))
    seconds = float(body.get("seconds") or 0) or None
    end = min(duration, start + seconds) if seconds else duration
    if end - start < 2:
        return 400, {"error": "window too short"}
    out = pipeline.unique_dir(RUNS, pipeline.run_name(video, start, end))
    out.mkdir(parents=True)
    cmd = [sys.executable, "-m", "relentless.pipeline", str(video), "--start", str(start), "--out", str(out),
           "--max-clips", str(int(body.get("max_clips", 8)))]
    if seconds:
        cmd += ["--seconds", str(seconds)]
    if body.get("fast", True):
        cmd.append("--fast")
    if not body.get("screen", True):
        cmd.append("--no-screen")
    if not body.get("cache", True):
        cmd.append("--no-cache")
    multiply = int(body.get("multiply") or 0)
    if multiply > 0:
        cmd += ["--multiply", str(multiply), "--variants", str(max(1, int(body.get("variants") or 1)))]
    log = open(out / "run.log", "w")
    with _lock:
        _active["proc"] = subprocess.Popen(cmd, cwd=REPO / "engine", stdout=log, stderr=subprocess.STDOUT)
        _active["name"] = out.name
    return 200, {"name": out.name}


def stop_run() -> tuple[int, dict]:
    with _lock:
        proc, name = _active["proc"], _active["name"]
    if proc is None or proc.poll() is not None:
        return 200, {"stopped": None}
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    state = RUNS / name / "state.json"
    if state.exists():
        s = json.loads(state.read_text())
        if s.get("status") == "running":
            s["status"] = "stopped"
            for st in s.get("stages", []):
                if st["status"] == "running":
                    st["status"] = "stopped"
            media.dump(s, state)
    return 200, {"stopped": name}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the terminal quiet; errors still print
        if args and str(args[1])[:1] in "45":
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def send_json(self, obj, code: int = 200) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            return self.send_json({"error": "not found"}, 404)
        size = path.stat().st_size
        start, end, code = 0, size - 1, HTTPStatus.OK
        rng = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            code = HTTPStatus.PARTIAL_CONTENT
        self.send_response(code)
        self.send_header("Content-Type", TYPES.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store" if path.suffix in (".json", ".html", ".log") else "no-cache")
        if code == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with path.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 18, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser cancelled (seek, tab closed)

    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(url.path)
        if path in ("/", "/index.html"):
            return self.send_file(VIEWER)
        if path == "/api/runs":
            return self.send_json({"runs": list_runs(), "running": running()})
        if path == "/api/videos":
            return self.send_json({"videos": list_videos(), "claude": claude_costs(), "fal": {
                "spent": round(ledger.spent("fal"), 2), "cap": FAL_BUDGET_USD,
                "per_second": FAL_MODELS["gemini"]["usd_per_s"]}})
        if path == "/api/catalog":
            return self.send_json(catalog())
        if path.startswith("/runs/"):
            p = inside(RUNS, path[len("/runs/"):])
            return self.send_file(p) if p else self.send_json({"error": "bad path"}, 400)
        self.send_json({"error": "not found"}, 404)

    def do_HEAD(self) -> None:
        self.do_GET()

    def _body(self, limit: int) -> bytes | None:
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            self.send_json({"error": "too large"}, 413)
            return None
        return self.rfile.read(n)

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/runs":
            raw = self._body(1 << 20)
            if raw is None:
                return
            try:
                code, out = start_run(json.loads(raw or b"{}"))
            except (ValueError, TypeError) as e:
                code, out = 400, {"error": str(e)}
            return self.send_json(out, code)
        if path == "/api/stop":
            self._body(1 << 10)
            return self.send_json(*reversed(stop_run()))
        self.send_json({"error": "not found"}, 404)

    def do_PUT(self) -> None:
        url = urllib.parse.urlparse(self.path)
        if url.path != "/api/upload":
            return self.send_json({"error": "not found"}, 404)
        name = urllib.parse.parse_qs(url.query).get("name", ["upload.mp4"])[0]
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)
        if Path(name).suffix.lower() not in VIDEO_EXT:
            return self.send_json({"error": "not a video file"}, 400)
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_UPLOAD:
            return self.send_json({"error": "missing or too large"}, 400)
        UPLOADS.mkdir(parents=True, exist_ok=True)
        dst = UPLOADS / name
        with dst.open("wb") as f:
            left = n
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        self.send_json({"path": str(dst.relative_to(DATA)), "name": name})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let teammates on the same network watch")
    ap.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    a = ap.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    server.daemon_threads = True
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}/"
    print(f"Relentless dashboard: {url}  (Ctrl+C to stop)", flush=True)
    if a.open:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        stop_run()


if __name__ == "__main__":
    main()
