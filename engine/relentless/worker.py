"""Process a website's queued jobs on this machine.

  uv run python -m relentless.worker https://peelset.com

The site's admin page can queue a job "for a worker" instead of running it on the web server.
This program asks the site for such jobs. For each one it downloads the footage, runs the
pipeline here (on this machine's GPU), reports the stages as they go, and sends the finished
clips back, where they join the site's catalog. Leave it running; stop it with Ctrl+C.

The admin token comes from --token, or from PEELSET_ADMIN_TOKEN in the repo's .env.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import DATA, REPO, RUNS, device, setting

CHUNK = 8 * 1024 * 1024
SENT_WITH_RUN = ("state.json", "catalog.json", "run.log")
NOT_SENT = {"compare"}    # the side-by-side video: the site switches between the two clips itself


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
            raise RuntimeError(f"{error.code} from the site: {detail or error.reason}") from error

    def download(self, name: str, dst: Path) -> Path:
        request = urllib.request.Request(f"{self.base}/footage/{urllib.parse.quote(name)}/file",
                                         headers={"Authorization": f"Bearer {self.token}", "User-Agent": "relentless-worker"})
        with urllib.request.urlopen(request, timeout=120) as response:
            size = int(response.headers.get("Content-Length") or 0)
            if dst.exists() and size and dst.stat().st_size == size:
                return dst      # already here from an earlier job
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            with open(tmp, "wb") as file:
                while piece := response.read(1024 * 1024):
                    file.write(piece)
            tmp.replace(dst)
        return dst

    def send_result(self, job_id: str, archive: Path) -> dict:
        total, reply = archive.stat().st_size, {}
        with open(archive, "rb") as file:
            for offset in range(0, total, CHUNK):
                query = urllib.parse.urlencode({"total": total, "offset": offset})
                reply = self.call(f"/jobs/{job_id}/result?{query}", "PUT", file.read(CHUNK), timeout=300)
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


def result_files(run_dir: Path) -> list[Path]:
    """What the site needs of a finished run: its state and catalog, and the files of every sellable item."""
    files = [run_dir / name for name in SENT_WITH_RUN if (run_dir / name).exists()]
    for item in read_json(run_dir / "catalog.json") or []:
        files += [run_dir / rel for kind, rel in item.get("files", {}).items()
                  if kind not in NOT_SENT and (run_dir / rel).exists()]
    return list(dict.fromkeys(files))


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
    return command


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
    footage = site.download(job["footage"], DATA / "uploads" / job["footage"])
    run_dir = RUNS / job["run"]
    run_dir.mkdir(parents=True, exist_ok=True)
    stopped = False
    with open(run_dir / "run.log", "w") as log:
        process = subprocess.Popen(pipeline_command(job, footage, run_dir), cwd=REPO / "engine", stdout=log,
                                   stderr=subprocess.STDOUT)
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
    return f"done: {accepted} clip(s) added to the site's catalog ({megabytes:.0f} MB sent)"


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
    site, hello = Site(a.site, token), {"worker": a.name, "device": device()}
    site.call("/status")      # fails right away on a wrong address or token
    print(f"{a.name} ({hello['device']}) is waiting for jobs from {a.site}. Ctrl+C stops it.", flush=True)
    while True:
        try:
            job = site.call("/worker/next", "POST", hello).get("job")
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
