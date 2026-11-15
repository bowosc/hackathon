"""Process a website's queued jobs on this machine.

  uv run python -m relentless.worker https://peelset.com

The site's admin page can queue a job "for a worker" instead of running it on the web server.
This program asks the site for such jobs. For each one it downloads the footage, runs the
pipeline here (on this machine's GPU), reports the stages as they go, and sends the finished
clips back, where they wait on the job's page to be published to the catalog. While a job runs it also
keeps the site's copy of the run's previews up to date (live.py), so the job page can show each stage
at work. Leave it running; stop it with Ctrl+C.

The admin token comes from --token, or from PEELSET_ADMIN_TOKEN in the repo's .env.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import ledger, media
from .config import CLAUDE_MODEL, DATA, FAL_BUDGET_USD, REPO, RUNS, device, setting

CHUNK = 8 * 1024 * 1024
SENT_WITH_RUN = ("state.json", "catalog.json", "run.log")
NOT_SENT = {"compare"}    # the side-by-side video: the site switches between the two clips itself
CLIP_FILES = ("raw", "upscaled", "poster", "hands", "card")      # what the job page shows of every clip
VARIANT_FILES = ("video", "original", "card")
PREVIEW_LIMIT = 8 * 1024 * 1024   # the site takes preview files up to this size


class SiteError(RuntimeError):
    """The site answered, and the answer was no."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"{status} from the site: {detail}")
        self.status = status


class Site:
    """The admin API of one site."""

    def __init__(self, base: str, token: str):
        self.base, self.token = base.rstrip("/") + "/api/admin", token

    def call(self, path: str, method: str = "GET", body=None, timeout: float = 120):
        data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {self.token}", "User-Agent": "relentless-worker"}
        if body is not None and not isinstance(body, bytes):
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            try:
                detail = json.load(error).get("detail")
            except ValueError:
                detail = None
            raise SiteError(error.code, detail or error.reason) from error

    def download(self, name: str, dst: Path, beat=None) -> Path:
        """Fetch a footage file. `beat` is called every 20 s, so the site knows a long download is still alive."""
        request = urllib.request.Request(f"{self.base}/footage/{urllib.parse.quote(name)}/file",
                                         headers={"Authorization": f"Bearer {self.token}", "User-Agent": "relentless-worker"})
        with urllib.request.urlopen(request, timeout=120) as response:
            size = int(response.headers.get("Content-Length") or 0)
            if dst.exists() and size and dst.stat().st_size == size:
                return dst      # already here from an earlier job
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp, last = dst.with_name(dst.name + ".part"), time.time()
            with open(tmp, "wb") as file:
                while piece := response.read(1024 * 1024):
                    file.write(piece)
                    if beat and time.time() - last > 20:
                        last = time.time()
                        beat()
            tmp.replace(dst)
        return dst

    def send_result(self, job_id: str, archive: Path) -> dict:
        total, reply = archive.stat().st_size, {}
        with open(archive, "rb") as file:
            for offset in range(0, total, CHUNK):
                query, piece = urllib.parse.urlencode({"total": total, "offset": offset}), file.read(CHUNK)
                for attempt in range(4):
                    try:
                        reply = self.call(f"/jobs/{job_id}/result?{query}", "PUT", piece, timeout=300)
                        break
                    except OSError:      # the connection dropped: the same piece again, a little later
                        if attempt == 3:
                            raise
                        time.sleep(3 * (attempt + 1))
        return reply


def summary(state: dict, log: str = "") -> dict:
    """What the admin page shows of a run (the same picture the site builds for its own jobs)."""
    clips = state.get("clips", [])
    return {"stages": [{key: stage.get(key) for key in ("key", "title", "status", "progress", "detail")}
                       for stage in state.get("stages", [])],
            "clips": {"planned": len(clips), "accepted": sum(c["status"] == "accepted" for c in clips),
                      "rejected": sum(c["status"] == "rejected" for c in clips)},
            "claude_usd": (state.get("costs") or {}).get("anthropic_usd"),
            "source": {key: (state.get("source") or {}).get(key) for key in ("start_s", "seconds")},
            "log": log[-5000:]}


def shown_files(run_dir: Path) -> dict[str, Path]:
    """The files a job page shows of a run, by their path in the run folder: the previews, and every clip
    with its variants. Only finished files: the state and the kit name a file once it is complete."""
    names = [part[key] for part in read_json(run_dir / "live" / "kit.json").values() if isinstance(part, dict)
             for key in ("file", "tracks") if part.get(key)]
    for clip in read_json(run_dir / "state.json").get("clips", []):
        names += [clip.get("files", {}).get(kind) for kind in CLIP_FILES]
        for variant in clip.get("variants", []):
            names += [variant.get("files", {}).get(kind) for kind in VARIANT_FILES]
    return {name: run_dir / name for name in names if name and (run_dir / name).is_file()}


def result_files(run_dir: Path) -> list[Path]:
    """What the site needs of a finished run: its state and catalog, the files of every sellable item,
    and what the job page shows (previews, rejected clips)."""
    files = [run_dir / name for name in SENT_WITH_RUN if (run_dir / name).exists()]
    for item in read_json(run_dir / "catalog.json") or []:
        files += [run_dir / rel for kind, rel in item.get("files", {}).items()
                  if kind not in NOT_SENT and (run_dir / rel).exists()]
    if (run_dir / "live" / "kit.json").is_file():
        files.append(run_dir / "live" / "kit.json")
    files += shown_files(run_dir).values()
    return list(dict.fromkeys(files))


class Previews(threading.Thread):
    """Keeps the site's copy of a run's previews up to date while the run goes on.

    The state and the frame a stage is on go first, every round. The other files follow as they appear, a few
    per round, so a large one never holds the state back for long. An upscaled clip is sent as a small copy:
    the full one travels once, with the result."""

    ALWAYS = ("state.json", "live/now.json", "live/kit.json")
    REFUSED = {400, 404, 413, 415, 422, 507}      # the site will not take this file, however often it is offered

    def __init__(self, site: Site, job_id: str, run_dir: Path):
        super().__init__(daemon=True, name="previews")
        self.site, self.job_id, self.run_dir = site, job_id, run_dir
        self.sent: dict[str, tuple] = {}
        self.copies: dict[str, tuple] = {}        # which version of a clip its small copy in the scratch folder is of
        self.over = False                         # the site says the job is not running any more
        self.done = threading.Event()
        self.scratch = tempfile.TemporaryDirectory(prefix="relentless-previews-")

    def run(self) -> None:
        while not self.done.wait(1.0):
            try:
                self.round()
            except Exception as error:      # previews are a courtesy: the run goes on without them
                print(f"  previews: {type(error).__name__}: {error}", flush=True)

    def finish(self, last: bool) -> None:
        """The run is over. With `last`, bring the site's copy up to the run's final state first: a run that
        failed or was stopped has no result to send, and its last state says how far it got and why it ended."""
        self.done.set()
        if self.ident is not None:      # it was started
            self.join(timeout=60)
        try:
            if last and not self.is_alive():
                self.round(budget=30)
        except Exception as error:
            print(f"  previews: {type(error).__name__}: {error}", flush=True)
        self.scratch.cleanup()

    def round(self, budget: float = 1.5) -> None:
        if self.over:
            return
        started = time.time()
        files = {name: self.run_dir / name for name in self.ALWAYS}
        files.update(shown_files(self.run_dir))
        for name, path in files.items():
            try:
                stat = path.stat()
            except OSError:
                continue
            mark = (stat.st_size, stat.st_mtime_ns)
            if self.sent.get(name) == mark:
                continue
            body = self.body(name, path, mark)
            if body is not None and len(body) <= PREVIEW_LIMIT:
                query = urllib.parse.urlencode({"path": name})
                try:
                    self.site.call(f"/jobs/{self.job_id}/live?{query}", "PUT", body, timeout=120)
                except SiteError as error:
                    if error.status == 409:
                        self.over = True
                        return
                    if error.status not in self.REFUSED:
                        return      # a passing trouble: the next round offers the file again
                except OSError:
                    return          # the connection dropped: the next round offers the file again
            self.sent[name] = mark      # sent, or refused for good, or too large to offer: either way, not again
            if name not in self.ALWAYS and time.time() - started > budget:
                return

    def body(self, name: str, path: Path, mark: tuple) -> bytes | None:
        try:
            if name.endswith("/upscaled.mp4"):
                small = Path(self.scratch.name) / name.replace("/", "-")
                if self.copies.get(name) != mark or not small.exists():      # encoded once per version of the clip
                    media.web_copy(path, small, crf=30, max_height=540, preset="veryfast")
                    self.copies[name] = mark
                return small.read_bytes()
            if name == "state.json":      # the engine writes it for reading; the site only needs the content
                return json.dumps(json.loads(path.read_text()), separators=(",", ":")).encode()
            return path.read_bytes()
        except (OSError, ValueError, RuntimeError):
            return None


def pack(run_dir: Path) -> Path:
    archive = run_dir / "result.tar"
    with tarfile.open(archive, "w") as tar:
        for path in result_files(run_dir):
            tar.add(path, arcname=str(path.relative_to(run_dir)))
    return archive


def pipeline_command(job: dict, footage: Path, run_dir: Path) -> list[str]:
    options = job["options"]
    command = [sys.executable, "-m", "relentless.pipeline", str(footage), "--out", str(run_dir),
               "--start", str(options.get("start", 0)), "--max-clips", str(options.get("max_clips", 4))]
    if options.get("seconds"):
        command += ["--seconds", str(options["seconds"])]
    if options.get("quality", "fast") == "fast":
        command.append("--fast")
    if options.get("multiply"):
        command += ["--multiply", str(int(options["multiply"]))]
    return command + ["--live"]      # the previews the job page shows


def read_json(path: Path):
    """The file's content, or an empty dict while it does not exist yet."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def tail(path: Path, lines: int = 12) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return "\n".join([line for line in text.splitlines() if line.strip() and "Warning" not in line][-lines:])


def run_job(site: Site, job: dict) -> str:
    """Do one job from download to upload. Returns a line for the console."""
    # Names come from the site. Only their last part is used, so they cannot point outside the data folder.
    name, run_dir = Path(job["footage"]).name, RUNS / Path(job["run"]).name

    def downloading() -> None:
        try:
            site.call(f"/jobs/{job['id']}/progress", "POST", summary({}, "downloading the footage"))
        except (RuntimeError, OSError):
            pass

    footage = site.download(job["footage"], DATA / "uploads" / name, beat=downloading)
    run_dir.mkdir(parents=True, exist_ok=True)
    stopped = False
    previews = Previews(site, job["id"], run_dir)
    with open(run_dir / "run.log", "w") as log:
        process = subprocess.Popen(pipeline_command(job, footage, run_dir), cwd=REPO / "engine", stdout=log,
                                   stderr=subprocess.STDOUT)
        previews.start()
        try:
            while process.poll() is None:
                time.sleep(2)
                try:
                    reply = site.call(f"/jobs/{job['id']}/progress", "POST",
                                      summary(read_json(run_dir / "state.json"), tail(run_dir / "run.log")))
                except (RuntimeError, OSError):
                    continue      # a report that did not get through is not a reason to give up the run
                if reply.get("stop") and not stopped:      # stopped from the admin page
                    stopped = True
                    process.terminate()
        finally:
            previews.finish(last=stopped or process.poll() != 0)
    if stopped or process.returncode != 0:
        site.call(f"/jobs/{job['id']}/failed", "POST", {"error": tail(run_dir / "run.log", 4)})
        return "stopped from the admin page" if stopped else f"failed (exit code {process.returncode})"
    site.call(f"/jobs/{job['id']}/progress", "POST", summary(read_json(run_dir / "state.json"), "sending the clips to the site"))
    archive = pack(run_dir)
    megabytes = archive.stat().st_size / 1e6
    try:
        done = site.send_result(job["id"], archive)
    finally:
        archive.unlink(missing_ok=True)
    accepted = done.get("job", {}).get("clips", {}).get("accepted", "?")
    return f"done: {accepted} clip(s) sent to the site, ready to publish from the job's page ({megabytes:.0f} MB)"


def hello(name: str) -> dict:
    """What this worker tells the site about itself each time it asks for work."""
    return {"worker": name, "device": device(), "labeler": CLAUDE_MODEL,
            "fal": {"key": bool(setting("FAL_KEY", "")), "spent_usd": round(ledger.spent("fal"), 3),
                    "cap_usd": FAL_BUDGET_USD}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", help="the site's address, e.g. https://peelset.com")
    ap.add_argument("--token", help="the site's admin token (default: PEELSET_ADMIN_TOKEN in .env)")
    ap.add_argument("--name", default=platform.node().split(".")[0] or "worker", help="how this worker shows up on the admin page")
    ap.add_argument("--once", action="store_true", help="do one job, then exit")
    a = ap.parse_args(argv)
    token = a.token or setting("PEELSET_ADMIN_TOKEN", "")
    if not token:
        raise SystemExit("No admin token: pass --token, or put PEELSET_ADMIN_TOKEN in the repo's .env")
    site = Site(a.site, token)
    site.call("/status")      # fails right away on a wrong address or token
    print(f"{a.name} ({device()}) is waiting for jobs from {a.site}. Ctrl+C stops it.", flush=True)
    while True:
        try:
            job = site.call("/worker/next", "POST", hello(a.name)).get("job")
        except (RuntimeError, OSError) as error:      # the site is restarting or the network dropped: keep asking
            print(f"  could not reach the site ({error}); trying again", flush=True)
            time.sleep(10)
            continue
        if not job:
            time.sleep(4)
            continue
        print(f"job {job['id']}: {job['footage']} {job['options']}", flush=True)
        started = time.time()
        try:
            outcome = run_job(site, job)
        except Exception as error:      # one bad job must not end the worker
            outcome = f"failed: {type(error).__name__}: {error}"
            try:
                site.call(f"/jobs/{job['id']}/failed", "POST", {"error": str(error)[:2000]})
            except (RuntimeError, OSError):
                pass
        print(f"  {outcome} in {time.time() - started:.0f} s", flush=True)
        if a.once:
            return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
