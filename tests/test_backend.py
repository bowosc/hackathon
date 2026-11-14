"""The site's backend on the real seed library: catalog records, search, the keyword interpreter and the admin API.

Run with:  .venv/bin/python -m pytest tests
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend import admin, catalog, datasets, library
from backend.main import app

client = TestClient(app)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point uploads, jobs and runs at an empty folder, with an admin token set."""
    for module, names in ((library, {"RUNS": "runs"}), (admin, {"STORE": "", "RUNS": "runs", "UPLOADS": "uploads",
                                                              "PARTIAL": "uploads/.partial", "JOBS_FILE": "jobs.json"})):
        for name, rel in names.items():
            monkeypatch.setattr(module, name, tmp_path / rel if rel else tmp_path)
    monkeypatch.setattr(admin, "secret", lambda name: {"ADMIN_TOKEN": "s3cret"}.get(name))
    monkeypatch.setattr(admin, "_jobs", [])
    return tmp_path


AUTH = {"Authorization": "Bearer s3cret"}


def test_the_catalog_is_the_real_library():
    clips = catalog.catalog()
    assert len(clips) >= 30 and len({clip["id"] for clip in clips}) == len(clips)
    for clip in clips:
        assert clip["video_url"].startswith("/api/media/") and clip["image"].endswith(".jpg")
        assert clip["actions"] and clip["segments"] and clip["variants"][:2] == ["source", "upscaled"]
        assert library.media_file(*clip["video_url"].removeprefix("/api/media/").split("/", 1)) is not None
    swapped = [clip for clip in clips if "augmented" in clip["variants"]]
    assert swapped and all(clip["versions"][-1]["key"] == "augmented" for clip in swapped)


def test_titles_and_families():
    item = {"main_object": "paper cup", "labels": [{"verb": "hold", "object": "paper cup"},
                                                    {"verb": "pick_up", "object": "paper cup"},
                                                    {"verb": "open", "object": "bag"}]}
    assert library.title(item) == "Picking up a paper cup"
    assert library.title({"main_object": "snack bags", "labels": [{"verb": "other", "object": "snack bags"}]}) == "Handling snack bags"
    assert library.families(item["labels"]) == ["Pick & place", "Open & close", "Holding & carrying"]
    assert library.environment("school hallway snack table beside lockers") == "Hallway"
    assert library.environment("cafe counter with snack basket") == "Café counter"
    assert library.lighting("dim indoor light") == "Low light" and library.lighting("even indoor fluorescent") == "Indoor"
    assert library.resolution([1912, 1072]) == "1080p" and library.resolution([1280, 712]) == "720p"


def test_media_stays_inside_the_library():
    assert library.media_file("seed", "../../config/settings.json") is None
    assert library.media_file("seed", "../../.env") is None
    assert library.media_file("elsewhere", "clips/x.mp4") is None
    assert client.get("/api/media/seed/clips/nope/upscaled.mp4").status_code == 404
    clip = catalog.catalog()[0]
    part = client.get(clip["video_url"], headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100 and part.headers["content-type"] == "video/mp4"


def test_search_words_meet_in_the_middle():
    assert catalog.search_words("Someone picking up the cups") == ["pick", "cup"]
    assert catalog.stem("putting") == catalog.stem("put") and catalog.stem("pulling") == catalog.stem("pull")
    clips = catalog.catalog()
    cups = catalog.filter_clips(clips, "picking up cups")
    assert cups and all("cup" in catalog.clip_text(clip).lower() for clip in cups)
    assert catalog.filter_clips(clips, "forklift pallets") == []
    # one unknown word among known ones still returns the clips that match the rest
    assert catalog.filter_clips(clips, "forklift snack bag")


def test_filters_and_facets_agree():
    groups = {group["key"]: group for group in client.get("/api/facets").json()["groups"]}
    assert set(groups) == {"environment", "action", "annotation", "resolution", "lighting"}
    for key, group in groups.items():
        for option in group["options"]:
            found = client.get("/api/clips", params={key: option["value"], "limit": 100}).json()
            assert found["total"] == option["count"] > 0


def test_keyword_plans_return_real_clips(monkeypatch):
    monkeypatch.setattr(datasets, "secret", lambda name: None)   # no AI key: the keyword interpreter answers
    def ask(prompt):
        compiled = client.post("/api/compile", json={"prompt": prompt}).json()
        preview = client.post("/api/dataset-preview?limit=50", json={"plan": compiled["plan"]}).json()
        return compiled, preview
    compiled, preview = ask("someone picking up a cup")
    assert compiled["mode"] == "demo" and compiled["plan"]["actions"] == ["Pick & place"]
    assert 0 < preview["total"] < preview["catalog_total"]
    assert all(any("cup" in name for name in clip["objects"]) for clip in preview["clips"])
    assert "cup" in preview["clips"][0]["title"]          # the cup being picked up ranks first
    compiled, preview = ask("I need hand-object interactions for a banana plantation robot. Adapt the manipulated objects to bananas.")
    assert compiled["plan"]["modifications"] == ["object_swap", "background"] and compiled["plan"]["target_object"] == "bananas"
    assert preview["total"] == preview["catalog_total"]   # the swap target is not looked for in the source footage
    assert ask("forklift moving pallets")[1]["total"] == 0
    assert ask("handing over money at a counter")[1]["total"] > 0


def test_ai_plans_keep_only_what_the_catalog_has():
    draft = datasets.Draft(summary="x", actions=["Pick & place", "Teleporting"], objects=["paper cup", "unicorn"],
                           environments=["Moon base"], annotations=[], keywords="", modifications=["object_swap"],
                           target_object="", lighting="varied", background="", resolution="4K", variants_per_clip=99)
    plan = datasets.tidy(draft)
    assert plan.actions == ["Pick & place"] and plan.objects == ["paper cup"] and plan.environments == []
    assert plan.modifications == [] and plan.variants_per_clip == catalog.SETTINGS["max_variants_per_clip"]


def test_admin_needs_the_token(store, monkeypatch):
    assert client.get("/api/admin/status").status_code == 401
    assert client.get("/api/admin/status", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/admin/status", headers=AUTH).status_code == 200
    monkeypatch.setattr(admin, "secret", lambda name: None)
    assert client.get("/api/admin/status", headers=AUTH).status_code == 503   # no token set: the panel is off


def test_upload_in_pieces(store):
    data = bytes(range(256)) * 40
    first = client.put("/api/admin/footage", params={"name": "../../Shop cam 1.MP4", "total": len(data)},
                       content=data[:6000], headers=AUTH)
    assert first.json() == {"done": False, "offset": 6000}
    wrong = client.put("/api/admin/footage", params={"name": "Shop cam 1.MP4", "total": len(data), "offset": 9000},
                       content=data[9000:], headers=AUTH)
    assert wrong.status_code == 409
    last = client.put("/api/admin/footage", params={"name": "Shop cam 1.MP4", "total": len(data), "offset": 6000},
                      content=data[6000:], headers=AUTH)
    assert last.json()["done"] and last.json()["footage"]["name"] == "Shop-cam-1.MP4"
    assert (store / "uploads" / "Shop-cam-1.MP4").read_bytes() == data      # the name cannot leave the folder
    assert [item["name"] for item in client.get("/api/admin/footage", headers=AUTH).json()["footage"]] == ["Shop-cam-1.MP4"]
    assert client.put("/api/admin/footage", params={"name": "notes.txt", "total": 3}, content=b"abc", headers=AUTH).status_code == 422
    assert client.put("/api/admin/footage", params={"name": "big.mp4", "total": admin.MAX_UPLOAD + 1}, content=b"x",
                      headers=AUTH).status_code == 413
    assert client.delete("/api/admin/footage/Shop-cam-1.MP4", headers=AUTH).status_code == 200
    assert not (store / "uploads" / "Shop-cam-1.MP4").exists()


def test_a_job_runs_the_engine_on_the_upload(store, monkeypatch):
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"video")
    monkeypatch.setattr(admin, "secret", lambda name: {"ADMIN_TOKEN": "s3cret", "ANTHROPIC_API_KEY": "key"}.get(name))
    monkeypatch.setattr(admin, "engine_python", lambda: None)
    refused = client.post("/api/admin/jobs", json={"footage": "cam.mp4"}, headers=AUTH)
    assert refused.status_code == 503 and "not installed" in refused.json()["detail"]
    monkeypatch.setattr(admin, "engine_python", lambda: "/engine/python")
    assert client.post("/api/admin/jobs", json={"footage": "missing.mp4"}, headers=AUTH).status_code == 404
    job = {"id": "abc12345", "footage": "cam.mp4", "run": "cam-abc12345",
           "options": {"start": 30.0, "seconds": 60.0, "max_clips": 4, "quality": "fast"}}
    command = admin.engine_command(job, "/engine/python")
    assert command[:3] == ["/engine/python", "-m", "relentless.pipeline"] and command[3] == str(store / "uploads" / "cam.mp4")
    assert command[4:] == ["--out", str(store / "runs" / "cam-abc12345"), "--start", "30.0", "--max-clips", "4",
                           "--seconds", "60.0", "--fast"]


def test_a_finished_run_joins_the_catalog(store):
    before = len(catalog.catalog())
    seed = json.loads((library.SEED / "catalog.json").read_text())[0]
    run = store / "runs" / "shop-cam-1"
    run.mkdir(parents=True)
    item = {**seed, "id": "shop-cam-1-c000", "run": "shop-cam-1", "caption": "A hand lifts a tin of paint off a shelf.",
            "objects": ["tin of paint"], "main_object": "tin of paint",
            "labels": [{"verb": "pick_up", "action": "lifts the tin", "object": "tin of paint", "hand": "right",
                        "start": 0.0, "end": 1.0}],
            "files": {"video": "clips/c000/upscaled.mp4", "poster": "clips/c000/poster.jpg"},
            "lineage": {"source": "shop-cam-1.mp4", "frames": [0, 90], "crop_xywh": [0, 0, 320, 180]}}
    (run / "catalog.json").write_text(json.dumps([item]))
    clips = catalog.catalog()
    assert len(clips) == before + 1
    new = next(clip for clip in clips if clip["id"] == "shop-cam-1-c000")
    assert new["title"] == "Picking up a tin of paint" and new["origin"] == "processed here"
    assert new["video_url"] == "/api/media/runs/shop-cam-1/clips/c000/upscaled.mp4"
    assert catalog.filter_clips(clips, "tin of paint")[0]["id"] == "shop-cam-1-c000"


def test_a_job_gives_way_when_memory_is_short(store, monkeypatch):
    """The engine shares its machine: a job does not start without room, and is stopped if room runs out."""
    job = {"id": "abc12345", "footage": "cam.mp4", "run": "cam-abc12345", "status": "running",
           "options": {"start": 0.0, "seconds": 10.0, "max_clips": 1, "quality": "fast"}}
    monkeypatch.setattr(admin, "engine_python", lambda: "/engine/python")
    monkeypatch.setattr(admin, "memory", lambda: (8 * 1024 * admin.MB, 500 * admin.MB))
    with pytest.raises(admin.JobError, match="Only 500 MB"):
        admin.run_job(job)
    assert not (store / "runs" / "cam-abc12345").exists()          # nothing was started

    readings = iter([4000, 4000, 200])                               # free MB: at the start, then while it runs
    monkeypatch.setattr(admin, "memory", lambda: (8 * 1024 * admin.MB, next(readings, 200) * admin.MB))
    monkeypatch.setattr(admin, "engine_command", lambda job, python: ["sleep", "60"])
    monkeypatch.setattr(admin, "ENGINE", store)
    monkeypatch.setattr(admin.time, "sleep", lambda seconds: None)
    admin.run_job(job)
    assert job["status"] == "failed" and "running out of memory" in job["error"]


def test_a_worker_takes_a_job_and_sends_the_clips_back(store, monkeypatch):
    """A job queued for a worker: claimed, reported on, and finished by uploading the run as an archive."""
    import io
    import tarfile

    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"raw footage")
    monkeypatch.setattr(admin, "engine_python", lambda: None)             # no engine on the site itself
    queued = client.post("/api/admin/jobs", json={"footage": "cam.mp4", "where": "worker", "max_clips": 2}, headers=AUTH)
    assert queued.status_code == 201 and queued.json()["status"] == "queued"
    assert client.post("/api/admin/worker/next", json={"worker": "mac"}).status_code == 401
    job = client.post("/api/admin/worker/next", json={"worker": "mac", "device": "mps"}, headers=AUTH).json()["job"]
    assert job["footage"] == "cam.mp4" and job["options"]["max_clips"] == 2
    assert client.post("/api/admin/worker/next", json={"worker": "mac"}, headers=AUTH).json() == {"job": None}
    assert client.get("/api/admin/status", headers=AUTH).json()["workers"] == [{"name": "mac", "device": ""}]
    assert client.get("/api/admin/footage/cam.mp4/file", headers=AUTH).content == b"raw footage"

    report = {"stages": [{"key": "people", "title": "Person tracking", "status": "running", "progress": 0.5}],
              "clips": {"planned": 0, "accepted": 0, "rejected": 0}, "source": {"start_s": 0, "seconds": 20}, "log": "tracking"}
    assert client.post(f"/api/admin/jobs/{job['id']}/progress", json=report, headers=AUTH).json() == {"stop": False}
    seen = client.get("/api/admin/jobs", headers=AUTH).json()["jobs"][0]
    assert seen["status"] == "running" and seen["worker"] == "mac" and seen["stages"][0]["progress"] == 0.5
    assert seen["log"] == "tracking" and "remote" not in seen

    seed = json.loads((library.SEED / "catalog.json").read_text())[0]
    item = {**seed, "id": f"{job['run']}-c000", "run": job["run"], "files": {"video": "clips/c000/upscaled.mp4"},
            "lineage": {"source": "cam.mp4", "frames": [0, 90], "crop_xywh": [0, 0, 320, 180]}}
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for name, data in {"catalog.json": json.dumps([item]).encode(), "state.json": b'{"clips": [{"status": "accepted"}]}',
                           "clips/c000/upscaled.mp4": b"video", "../escaped.json": b"{}", "clips/c000/run.sh": b"#!/bin/sh"}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    data, before = archive.getvalue(), len(catalog.catalog())
    first = client.put(f"/api/admin/jobs/{job['id']}/result", params={"total": len(data)}, content=data[:4000], headers=AUTH)
    assert first.json() == {"done": False, "offset": 4000}
    last = client.put(f"/api/admin/jobs/{job['id']}/result", params={"total": len(data), "offset": 4000}, content=data[4000:],
                      headers=AUTH)
    assert last.json()["done"] and last.json()["job"]["status"] == "done" and last.json()["job"]["clips"]["accepted"] == 1
    run = store / "runs" / job["run"]
    assert (run / "clips/c000/upscaled.mp4").read_bytes() == b"video"
    assert not (store / "runs" / "escaped.json").exists() and not (run / "clips/c000/run.sh").exists()   # only media and labels, only inside
    assert len(catalog.catalog()) == before + 1
    assert client.put(f"/api/admin/jobs/{job['id']}/result", params={"total": 5}, content=b"again", headers=AUTH).status_code == 409


def test_stopping_a_worker_job_reaches_the_worker(store, monkeypatch):
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"raw footage")
    job_id = client.post("/api/admin/jobs", json={"footage": "cam.mp4", "where": "worker"}, headers=AUTH).json()["id"]
    client.post("/api/admin/worker/next", json={"worker": "mac"}, headers=AUTH)
    assert client.post(f"/api/admin/jobs/{job_id}/stop", headers=AUTH).json()["status"] == "stopping"
    assert client.post(f"/api/admin/jobs/{job_id}/progress", json={}, headers=AUTH).json() == {"stop": True}
    assert client.post(f"/api/admin/jobs/{job_id}/failed", json={"error": "terminated"}, headers=AUTH).json()["status"] == "stopped"
    # a worker that goes quiet loses its job
    other = client.post("/api/admin/jobs", json={"footage": "cam.mp4", "where": "worker"}, headers=AUTH).json()["id"]
    client.post("/api/admin/worker/next", json={"worker": "mac"}, headers=AUTH)
    lost = next(job for job in admin._jobs if job["id"] == other)
    lost["heard"] -= admin.WORKER_SILENCE + 1
    for _ in range(100):
        if lost["status"] == "failed":
            break
        admin._wake.set()
        import time
        time.sleep(0.1)
    assert lost["status"] == "failed" and "stopped reporting" in lost["error"]
