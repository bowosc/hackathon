"""The previews a job page shows (live.py): what they hold, and that no face survives in them."""
import base64
import json
import tarfile

from pathlib import Path

import cv2
import numpy as np

from conftest import FPS, person
from relentless import live, media, render, worker


def picture(now: dict) -> np.ndarray:
    data = base64.b64decode(now["image"].split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def test_the_motion_picture_is_too_coarse_for_a_face():
    """Before tracking nobody's face has been found, so the frame itself is shown at 64 pixels across."""
    small = np.full((270, 480), 40, np.uint8)
    small[100:130, 200:225] = np.indices((30, 25)).sum(axis=0) % 2 * 215 + 40      # a "face": fine detail, 25 px wide
    changed = np.zeros((270, 480), np.uint8)
    changed[200:220, 50:90] = 1
    img = live.motion_picture(small, changed)
    assert img.shape == (270, 480, 3)
    face = img[100:130, 200:225].astype(int)
    assert np.abs(np.diff(face[:, :, 0], axis=1)).max() < 12        # the pixel-level pattern is gone
    assert img[210, 70][0] > 150 and img[210, 70][2] < 80          # changed pixels are lit, in blue
    assert img[20, 20].max() < 60                                   # the rest stays dim


def test_the_motion_view_follows_the_scan(tmp_path):
    view = live.MotionView(tmp_path, first=300, last=900, fps=30.0, every=0)
    small = np.zeros((90, 160), np.uint8)
    for f, score in ((300, 0.0), (303, 0.004), (450, 0.02)):
        view(f, small, None if f == 300 else np.zeros_like(small), score)
    now = json.loads((tmp_path / "now.json").read_text())
    assert now["stage"] == "motion" and now["seq"] == 3 and now["t"] == 5.0
    assert len(now["timeline"]) == live.BINS and now["timeline"][0] == 0.0 and now["timeline"][1] == 0.004
    assert now["timeline"][60] == 0.02 and now["timeline"][61] is None      # nothing is claimed past the scan
    assert picture(now).shape == (90, 160, 3)


def detail(img) -> float:
    """How much a picture varies from pixel to pixel, per colour channel: high for texture, low for a blur."""
    return float(img.astype(float).std(axis=(0, 1)).mean())


def walk(view, people_at, n, frame, step=3):
    """Feed `n` samples to a tracking view. `people_at(i)` gives (tracked, others) for sample i."""
    for i in range(n):
        tracked, others = people_at(i)
        view(i * step, frame, tracked, others, samples=i + 1)


def test_the_tracking_view_blurs_every_head_it_shows(tmp_path):
    rng = np.random.default_rng(1)
    frame = np.repeat(np.repeat(rng.integers(0, 255, (68, 120, 3), dtype=np.uint8), 16, axis=0), 16, axis=1)   # 1920x1088
    followed, loose = person(1, x=600.0, y=300.0, height=400.0), person(1, x=1300.0, y=300.0, height=400.0)
    view, before = live.TrackingView(tmp_path, first=0, last=300, fps=FPS, every=0), frame.copy()
    both = lambda i: ([(7, followed.box[0], followed.kp[0])], [(loose.box[0], loose.kp[0])])       # noqa: E731
    walk(view, both, live.REACH, frame)
    assert not (tmp_path / "now.json").exists()        # nothing is shown before the samples after a frame are known
    walk(view, both, live.REACH + 5, frame)
    now = json.loads((tmp_path / "now.json").read_text())
    # the frame shown is REACH samples behind the newest one (sample 10, frame 30), not the newest
    assert now["stage"] == "people" and now["t"] == round((10 - live.REACH) * 3 / FPS, 2) and now["size"] == [1920, 1088]
    assert now["samples"] == live.REACH + 5 and now["seen"] == 1 and now["counts"][24] == 1 and now["counts"][25] is None
    assert [p["id"] for p in now["people"]] == [7] and now["people"][0]["box"] == [532, 300, 668, 700]
    assert len(now["people"][0]["kp"]) == 17
    shown, s = picture(now), 768 / 1920
    assert shown.shape[1] == 768
    plain = detail(shown[int(600 * s):int(640 * s), int(900 * s):int(960 * s)])
    for x in (600.0, 1300.0):                                       # the person followed, and the one that is not
        assert detail(shown[int(322 * s):int(342 * s), int((x - 10) * s):int((x + 10) * s)]) < plain / 3
    assert (frame == before).all()                                  # the frame the tracker works on is left alone


def test_a_person_the_model_misses_for_a_moment_stays_blurred_in_the_live_frame(tmp_path):
    """The outputs blur a person between two sightings a few samples apart. The live frame must do the same,
    so it is the frame from REACH samples ago, blurred with what was seen before and after it."""
    from relentless import people

    rng = np.random.default_rng(2)
    frame = np.repeat(np.repeat(rng.integers(0, 255, (68, 120, 3), dtype=np.uint8), 16, axis=0), 16, axis=1)
    n, step, missed = 24, 3, {8, 9, 10, 11, 16}
    x_at = lambda i: 400.0 + 30.0 * i                               # walking to the right      # noqa: E731
    body = [person(1, x=x_at(i), y=300.0, height=400.0) for i in range(n)]
    stray = person(1, x=1500.0, y=500.0, height=300.0)

    def people_at(i):
        tracked = [] if i in missed else [(3, body[i].box[0], body[i].kp[0])]
        return tracked, ([(stray.box[0], stray.kp[0])] if i == 9 else [])

    # what the outputs blur, computed from the whole stretch
    seen = [i for i in range(n) if i not in missed]
    raw = {"frame": np.array([i * step for i in seen]), "id": np.full(len(seen), 3), "step": step,
           "box": np.stack([body[i].box[0] for i in seen]), "kp": np.stack([body[i].kp[0] for i in seen]),
           "loose": {9 * step: [people.head_circle(stray.box[0], stray.kp[0])]}}
    outputs = people.head_circles(raw)

    view, checked = live.TrackingView(tmp_path, first=0, last=n * step, fps=FPS, every=0), 0
    for i in range(n):
        view(i * step, frame, *people_at(i), samples=i + 1)
        shown_sample = i - live.REACH
        if shown_sample < 0:
            continue
        mine = view.circles(len(view._window) - 1 - live.REACH)
        theirs = outputs.get(shown_sample * step, np.zeros((0, 3), np.float32))
        assert sorted(map(tuple, np.round(mine, 2).tolist())) == sorted(map(tuple, np.round(theirs, 2).tolist())), shown_sample
        checked += 1
        if shown_sample in (9, 10):     # nobody was detected as followed here, yet their head is blurred in the picture
            now, s = json.loads((tmp_path / "now.json").read_text()), 768 / 1920
            assert now["people"] == [] and len(mine) >= 1
            cx = x_at(shown_sample)
            head = detail(picture(now)[int(322 * s):int(342 * s), int((cx - 10) * s):int((cx + 10) * s)])
            assert head < detail(picture(now)[int(900 * s):int(940 * s), int(200 * s):int(260 * s)]) / 3
            # and it is the blur that does it: the same spot in the frame as the tracker saw it is full of detail
            assert detail(frame[322:342, int(cx - 10):int(cx + 10)]) > 3 * head
    assert checked == n - live.REACH and len(outputs[10 * step]) == 1 and len(outputs[9 * step]) == 2


def test_the_preview_is_the_busiest_stretch():
    fps = 30.0
    moments = [(1500, 1650), (1560, 1700), (4000, 4100)]
    a, b = live.busiest(0, 9000, fps, moments, people=[(0, 9000)])
    assert b - a == 14 * 30 and a <= 1500 and b >= 1700            # both overlapping moments are inside
    assert live.busiest(60, 360, fps, moments, []) == (60, 360)     # a stretch shorter than the preview is shown whole
    assert live.busiest(0, 9000, fps, [], [(6000, 6900)])[0] >= 5580   # nobody's hands busy: where the people are


def test_a_cut_pass_writes_what_the_page_shows(tmp_path):
    """Scan and preview videos, the people of the preview, and one still per hand moment, from one pass."""
    src = tmp_path / "cam.mp4"
    with media.Writer(src, 320, 180, FPS) as writer:
        for f in range(90):
            frame = np.full((180, 320, 3), 30, np.uint8)
            frame[:, : 3 * f + 10] = (40, 120, 200)                 # something that changes from frame to frame
            writer.write(frame)
    t = person(30, x=160.0, y=20.0, height=150.0)                    # sampled every 3rd frame over all 90
    t.id = 4
    moments = [{"k": 0, "track": 4, "start": 15, "end": 75, "crop": (64, 36, 160, 90), "hands": ["right"], "score": 0.9, "clip": "c000"},
               {"k": 1, "track": 4, "start": 30, "end": 84, "crop": (0, 0, 160, 90), "hands": ["left"], "score": 0.4, "clip": None}]
    kit = live.Kit(tmp_path / "run" / "live", 0, 90, FPS, (320, 180))
    jobs = kit.cut_jobs([t], {4: [(30, 60)]}, moments)
    assert render.run(src, jobs, heads={}) > 0
    kit.after_cut()
    parts = json.loads((tmp_path / "run" / "live" / "kit.json").read_text())
    assert parts["scan"] == {"file": "live/scan.mp4", "fps": 5.0, "seconds": 3.0, "size": [640, 360]}
    assert parts["preview"]["start_s"] == 0.0 and parts["preview"]["seconds"] == 3.0 and parts["preview"]["tracks"] == "live/tracks.json"
    assert parts["cands"] == {"file": "live/cands.jpg", "tile": [160, 90], "cols": 16, "index": [0, 1]}
    with media.Reader(tmp_path / "run" / "live" / "scan.mp4") as scan, media.Reader(tmp_path / "run" / "live" / "preview.mp4") as preview:
        assert (scan.count, scan.width) == (15, 640) and (preview.count, preview.width) == (90, 960)
    sheet = cv2.imread(str(tmp_path / "run" / "live" / "cands.jpg"))
    assert sheet.shape == (90, 16 * 160, 3)
    assert sheet[45, 80].astype(int).sum() > 250 > sheet[45, 800].astype(int).sum()     # a still, then empty tiles
    tracks = json.loads((tmp_path / "run" / "live" / "tracks.json").read_text())
    assert len(tracks["samples"]) == 30 and tracks["samples"][10] == {"t": 1.0, "people": tracks["samples"][10]["people"]}
    assert [p["busy"] for s in tracks["samples"] for p in s["people"]] == [10 <= i < 20 for i in range(30)]
    assert [(c["k"], c["clip"], c["start_s"], c["seconds"]) for c in tracks["candidates"]] == [(0, "c000", 0.5, 2.0), (1, None, 1.0, 1.8)]


def test_the_screen_sheet_has_one_row_per_candidate(tmp_path):
    kit = live.Kit(tmp_path / "live", 0, 300, FPS, (1920, 1080))
    stills = {k: [np.full((180, 320, 3), 10 * k, np.uint8)] * 4 for k in (9, 2, 5)}
    kit.screen_sheet(stills)
    parts = json.loads((tmp_path / "live" / "kit.json").read_text())
    assert parts["screen"] == {"file": "live/screen.jpg", "tile": [160, 90], "frames": 4, "index": [2, 5, 9]}
    sheet = cv2.imread(str(tmp_path / "live" / "screen.jpg"))
    assert sheet.shape == (3 * 90, 4 * 160, 3) and abs(int(sheet[225, 300, 0]) - 90) < 6     # the third row is candidate 9
    kit.quiet()                                                     # with no frame on show, there is nothing to remove
    assert not (tmp_path / "live" / "now.json").exists()


def run_folder(tmp_path):
    """A run as the engine leaves it with --live: an accepted clip, a rejected one, and the previews."""
    run = tmp_path / "shop-abc"
    for clip in ("c000", "c001"):
        (run / "clips" / clip).mkdir(parents=True)
        for name in ("upscaled.mp4", "raw.mp4", "compare.mp4", "poster.jpg", "card.json", "hands.json", "context.mp4"):
            (run / "clips" / clip / name).write_bytes(b"x")
    (run / "live").mkdir()
    for name in ("scan.mp4", "preview.mp4", "tracks.json", "cands.jpg", "now.json", "unfinished.mp4"):
        (run / "live" / name).write_bytes(b"y")
    (run / "live" / "kit.json").write_text(json.dumps({
        "scan": {"file": "live/scan.mp4"}, "preview": {"file": "live/preview.mp4", "tracks": "live/tracks.json"},
        "cands": {"file": "live/cands.jpg"}, "screen": {"file": "live/screen.jpg"}}))      # screen.jpg does not exist
    files = lambda clip: {kind: f"clips/{clip}/{name}" for kind, name in (               # noqa: E731
        ("raw", "raw.mp4"), ("upscaled", "upscaled.mp4"), ("poster", "poster.jpg"), ("hands", "hands.json"),
        ("card", "card.json"), ("compare", "compare.mp4"), ("context", "context.mp4"))}
    (run / "state.json").write_text(json.dumps({
        "stages": [{"key": "label", "title": "Label", "status": "done", "progress": 1.0}],
        "clips": [{"id": "c000", "status": "accepted", "files": files("c000"), "variants": []},
                  {"id": "c001", "status": "rejected", "files": files("c001"), "variants": []}],
        "costs": {"anthropic_usd": 0.004}, "source": {"start_s": 0, "seconds": 20}}, indent=2))
    (run / "catalog.json").write_text(json.dumps([{"id": "shop-abc-c000", "files": {
        "video": "clips/c000/upscaled.mp4", "original": "clips/c000/raw.mp4", "compare": "clips/c000/compare.mp4",
        "poster": "clips/c000/poster.jpg", "card": "clips/c000/card.json", "context": "clips/c000/context.mp4"}}]))
    return run


def test_a_worker_sends_back_what_the_site_and_the_job_page_need(tmp_path):
    """The archive: the run's state and catalog, each sellable item's files, every clip the page shows, the previews."""
    run = run_folder(tmp_path)
    with tarfile.open(worker.pack(run)) as tar:
        names = set(tar.getnames())
    clip = lambda c, *names: {f"clips/{c}/{n}" for n in names}                           # noqa: E731
    assert names == ({"catalog.json", "state.json", "live/kit.json", "live/scan.mp4", "live/preview.mp4",
                      "live/tracks.json", "live/cands.jpg"}
                     | clip("c000", "upscaled.mp4", "raw.mp4", "poster.jpg", "card.json", "hands.json", "context.mp4")
                     | clip("c001", "upscaled.mp4", "raw.mp4", "poster.jpg", "card.json", "hands.json"))
    picture = worker.summary(json.loads((run / "state.json").read_text()), "last lines")
    assert picture["clips"] == {"planned": 2, "accepted": 1, "rejected": 1} and picture["claude_usd"] == 0.004
    assert picture["source"] == {"start_s": 0, "seconds": 20} and picture["log"] == "last lines"
    job = {"id": "abc", "footage": "shop.mp4", "run": "shop-abc",
           "options": {"start": 5.0, "seconds": 20.0, "max_clips": 3, "quality": "full", "multiply": 2}}
    command = worker.pipeline_command(job, tmp_path / "shop.mp4", run)
    assert command[1:3] == ["-m", "relentless.pipeline"] and "--fast" not in command
    assert command[3:] == [str(tmp_path / "shop.mp4"), "--out", str(run), "--start", "5.0", "--max-clips", "3",
                           "--seconds", "20.0", "--multiply", "2", "--live"]
    del job["options"]["multiply"]
    assert "--multiply" not in worker.pipeline_command(job, tmp_path / "shop.mp4", run)


def test_a_worker_keeps_the_sites_previews_up_to_date(tmp_path, monkeypatch):
    """While a run goes on: the state and the current frame every round, each finished file once, clips made small."""
    run = run_folder(tmp_path)
    sent, encoded = [], []

    class Site:
        refuse = {}          # file name -> what goes wrong when it is offered

        def call(self, path, method="GET", body=None, timeout=120):
            name = path.split("path=")[1].replace("%2F", "/")
            if name in self.refuse:
                raise self.refuse[name]
            sent.append((name, method, len(body)))
            return {}

    def small_copy(src, dst, **how):
        encoded.append(src.parent.name)
        dst.write_bytes(b"small")
        return dst

    monkeypatch.setattr(worker.media, "web_copy", small_copy)
    previews = worker.Previews(Site(), "abc", run)
    previews.round(budget=60)
    names = [name for name, _, _ in sent]
    assert names[:3] == ["state.json", "live/now.json", "live/kit.json"] and all(method == "PUT" for _, method, _ in sent)
    assert "live/unfinished.mp4" not in names and "clips/c000/compare.mp4" not in names      # only what is named, and finished
    assert {"live/scan.mp4", "live/tracks.json", "clips/c001/poster.jpg", "clips/c000/hands.json"} <= set(names)
    sizes = {name: size for name, _, size in sent}
    assert sizes["clips/c000/upscaled.mp4"] == 5                    # the small copy, not the clip
    assert sizes["state.json"] < (run / "state.json").stat().st_size
    sent.clear()
    previews.round()
    assert sent == []                                               # nothing changed, nothing sent
    (run / "live" / "now.json").write_bytes(b"a newer frame")
    previews.round()
    assert [name for name, _, _ in sent] == ["live/now.json"]

    # a dropped connection, or a site in trouble: the same file is offered again next round
    for trouble in (OSError("connection reset"), worker.SiteError(503, "starting up")):
        sent.clear()
        (run / "live" / "now.json").write_bytes(b"frame " + str(trouble).encode())
        Site.refuse = {"live/now.json": trouble}
        previews.round()
        assert sent == []
        Site.refuse = {}
        previews.round()
        assert [name for name, _, _ in sent] == ["live/now.json"]

    # a file the site will not take is not offered again, is not encoded again, and holds nothing else back
    sent.clear()
    encoded.clear()
    (run / "clips" / "c000" / "upscaled.mp4").write_bytes(b"a new version of the clip")
    (run / "clips" / "c001" / "poster.jpg").write_bytes(b"a new poster")
    Site.refuse = {"clips/c000/upscaled.mp4": worker.SiteError(507, "No room for more previews of this job.")}
    for _ in range(4):
        previews.round(budget=60)
    assert encoded == ["c000"] and [name for name, _, _ in sent] == ["clips/c001/poster.jpg"]

    # the job is over on the site (stopped from the page, say): nothing more is sent
    sent.clear()
    (run / "live" / "now.json").write_bytes(b"one more frame")
    Site.refuse = {"live/now.json": worker.SiteError(409, "This job is not running any more.")}
    previews.round()
    Site.refuse = {}
    (run / "clips" / "c001" / "poster.jpg").write_bytes(b"yet another poster")
    previews.round()
    assert sent == [] and previews.over
    previews.scratch.cleanup()


def test_a_run_that_fails_leaves_its_last_state_on_the_site(tmp_path, monkeypatch):
    """No result is sent for a failed run, so the last thing the worker does is bring the site's copy up to date."""
    run = run_folder(tmp_path)
    sent = []

    class Site:
        def call(self, path, method="GET", body=None, timeout=120):
            sent.append((path.split("path=")[1].replace("%2F", "/"), body))
            return {}

    monkeypatch.setattr(worker.media, "web_copy", lambda src, dst, **how: (dst.write_bytes(b"small"), dst)[1])
    previews = worker.Previews(Site(), "abc", run)
    previews.start()
    state = json.loads((run / "state.json").read_text())
    state.update(status="error", error="RuntimeError: no motion found in this footage")
    (run / "state.json").write_text(json.dumps(state))
    previews.finish(last=True)
    last = [json.loads(body) for name, body in sent if name == "state.json"][-1]
    assert last["status"] == "error" and "no motion" in last["error"]
    assert not previews.is_alive() and not Path(previews.scratch.name).exists()
    # a run that succeeded sends its result instead: no last round
    sent.clear()
    again = worker.Previews(Site(), "abc", run)
    again.finish(last=False)
    assert sent == []
