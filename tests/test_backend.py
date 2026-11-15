"""The site's backend on the real seed library: catalog records, search, the keyword interpreter and the admin API.

Run with:  .venv/bin/python -m pytest tests
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend import admin, catalog, datasets, documents, library
from backend.main import app

client = TestClient(app)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point uploads, jobs and runs at an empty folder, with an admin token set."""
    for module, names in ((library, {"RUNS": "runs"}), (admin, {"STORE": "", "RUNS": "runs", "UPLOADS": "uploads",
                                                              "PARTIAL": "uploads/.partial", "JOBS_FILE": "jobs.json",
                                                              "INCOMING": "incoming", "LIVE": "live",
                                                              "FILM": "uploads/.film", "TRASH": "trash"})):
        for name, rel in names.items():
            monkeypatch.setattr(module, name, tmp_path / rel if rel else tmp_path)
    monkeypatch.setattr(admin, "secret", lambda name: {"ADMIN_TOKEN": "s3cret"}.get(name))
    for name, empty in (("_jobs", []), ("_workers", {}), ("_previews", {}), ("_pictures", {}), ("_documents", {}),
                        ("_listed", {}), ("_film_failed", {})):
        monkeypatch.setattr(admin, name, empty)
    client.cookies.clear()
    return tmp_path


def archive_of(files):
    """A plain tar, as a worker sends its finished run."""
    import io
    import tarfile

    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return packed.getvalue()


def worker_job(footage="cam.mp4", **options):
    """A job queued for a worker and taken by one. Returns what the worker was handed."""
    queued = client.post("/api/admin/jobs", json={"footage": footage, "where": "worker", **options}, headers=AUTH)
    assert queued.status_code == 201, queued.text
    return client.post("/api/admin/worker/next", json={"worker": "mac", "device": "mps"}, headers=AUTH).json()["job"]


def seed_item(run, clip="c000", **changes):
    """A catalog item as the engine writes it, for a clip of `run`."""
    seed = json.loads((library.SEED / "catalog.json").read_text())[0]
    first = 100 * int(clip[1:])      # each clip is its own stretch of the footage: the catalog keeps one clip per stretch
    return {**seed, "id": f"{run}-{clip}", "run": run, "files": {"video": f"clips/{clip}/upscaled.mp4"},
            "lineage": {"source": f"{run}.mp4", "frames": [first, first + 90], "crop_xywh": [0, 0, 320, 180]}, **changes}


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


def test_a_plan_that_asks_too_much_gives_up_the_least():
    """No single clip has the action, the object and the place: the plan drops the fewest parts that leave a match."""
    clips = catalog.catalog()
    handover = {clip["id"] for clip in clips if "Handover" in clip["actions"]}
    counter = {clip["id"] for clip in clips if clip["environment"] == "Café counter"}
    money = [name for name in datasets.inventory()["objects"] if "money" in name or "banknote" in name or "note" in name]
    with_money = {clip["id"] for clip in clips if set(money) & set(clip["objects"])}
    assert handover and counter and with_money and not (handover & counter & with_money)      # the premise
    asked = datasets.Plan(summary="Handing over money at a counter.", actions=["Handover"], objects=money,
                          environments=["Café counter"])
    plan = datasets.loosen(asked)
    found = {clip["id"] for clip in datasets.resolve_selection(datasets.Selection(plan=plan))[0]}
    dropped = [part for part in ("environments", "actions", "objects") if getattr(asked, part) and not getattr(plan, part)]
    assert found and len(dropped) == 1 and "left open" in plan.summary
    # it kept the pair of parts that matches the most clips
    pairs = {"environments": handover & with_money, "actions": counter & with_money, "objects": handover & counter}
    assert found == pairs[dropped[0]] and len(found) == max(len(ids) for ids in pairs.values())
    # a plan that already matches is left alone, and one that only has keywords nobody matches stays empty
    fine = datasets.Plan(summary="Cups.", actions=["Pick & place"])
    assert datasets.loosen(fine) == fine
    nothing = datasets.loosen(datasets.Plan(summary="Forklifts.", keywords="forklift pallet"))
    assert not datasets.resolve_selection(datasets.Selection(plan=nothing))[0]


def test_ai_plans_keep_only_what_the_catalog_has():
    draft = datasets.Draft(summary="x", actions=["Pick & place", "Teleporting"], objects=["paper cup", "unicorn"],
                           environments=["Moon base"], annotations=[], keywords="", modifications=["object_swap"],
                           target_object="", lighting="varied", background="", resolution="4K", variants_per_clip=99)
    plan = datasets.tidy(draft)
    assert plan.actions == ["Pick & place"] and plan.objects == ["paper cup"] and plan.environments == []
    assert plan.modifications == [] and plan.variants_per_clip == catalog.SETTINGS["max_variants_per_clip"]
    # the library has nothing like the request: the selection is empty rather than everything
    missing = datasets.tidy(draft.model_copy(update={"actions": [], "objects": [], "unavailable": "forklift, pallets"}))
    assert missing.keywords == "forklift pallets"
    # only the words no clip mentions are kept, so words the catalog does know cannot let clips back in
    wordy = datasets.tidy(draft.model_copy(update={"actions": [], "objects": [], "unavailable": "a person holding a hammer at a table"}))
    assert wordy.keywords == "hammer" and datasets.resolve_selection(datasets.Selection(plan=wordy))[0] == []
    assert datasets.resolve_selection(datasets.Selection(plan=missing))[0] == []
    # but with something real to go on, the closest clips are still shown
    partly = datasets.tidy(draft.model_copy(update={"unavailable": "a wrench"}))
    assert partly.keywords == "" and datasets.resolve_selection(datasets.Selection(plan=partly))[0]


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
    assert admin.safe_name("a" * 250 + ".mp4") == "a" * 100 + ".mp4"      # a long name keeps its extension
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
                           "--seconds", "60.0", "--fast", "--live"]      # --live: the previews the job page shows
    # object-swap variants cost fal credits, and only a worker has a fal key
    job["options"]["multiply"] = 2
    assert admin.engine_command(job, "/engine/python")[-3:] == ["--multiply", "2", "--live"]
    refused = client.post("/api/admin/jobs", json={"footage": "cam.mp4", "multiply": 1}, headers=AUTH)
    assert refused.status_code == 422 and "worker" in refused.json()["detail"]
    assert client.post("/api/admin/jobs", json={"footage": "cam.mp4", "multiply": 9, "where": "worker"}, headers=AUTH).status_code == 422
    # the engine's default labeler is named here too, for the page: the two must not drift apart
    assert f'"{admin.ENGINE_LABELER}"' in (admin.ENGINE / "relentless" / "config.py").read_text()


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
    # what was processed here is newer than the seed library, whatever clock the machine that made it runs on
    assert catalog.filter_clips(clips, sort="newest")[0]["id"] == "shop-cam-1-c000" and clips[0]["id"] == "shop-cam-1-c000"
    # a run that is held back (one started from the admin page, until it is published) is not in the catalog
    (run / library.HOLD).touch()
    assert len(catalog.catalog()) == before and client.get("/api/media/runs/shop-cam-1/clips/c000/upscaled.mp4").status_code == 404
    (run / library.HOLD).unlink()
    assert len(catalog.catalog()) == before + 1


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

    # an engine that ignores the request to stop is killed
    monkeypatch.setattr(admin, "memory", lambda: (8 * 1024 * admin.MB, 4000 * admin.MB))
    monkeypatch.setattr(admin, "engine_command", lambda job, python: ["sh", "-c", "trap '' TERM; sleep 60"])
    import itertools
    clock = itertools.count(0, 6)                                    # every look at the clock is 6 s later
    monkeypatch.setattr(admin.time, "time", lambda: next(clock))
    job.update(status="stopping", error=None)
    admin.run_job(job)
    assert job["status"] == "stopped"


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
    assert client.get("/api/admin/status", headers=AUTH).json()["workers"] == [
        {"name": "mac", "device": "", "labeler": "", "fal": {}, "busy": True}]
    assert client.get("/api/admin/footage/cam.mp4/file", headers=AUTH).content == b"raw footage"

    report = {"stages": [{"key": "people", "title": "Person tracking", "status": "running", "progress": 0.5}],
              "clips": {"planned": 0, "accepted": 0, "rejected": 0}, "source": {"start_s": 0, "seconds": 20}, "log": "tracking"}
    assert client.post(f"/api/admin/jobs/{job['id']}/progress", json=report, headers=AUTH).json() == {"stop": False}
    # a worker asks for work only between jobs: while it has one, its reports keep it on the list of workers
    admin._workers["mac"]["at"] -= 3600
    assert client.get("/api/admin/status", headers=AUTH).json()["workers"] == []
    client.post(f"/api/admin/jobs/{job['id']}/progress", json=report, headers=AUTH)
    assert [worker["name"] for worker in client.get("/api/admin/status", headers=AUTH).json()["workers"]] == ["mac"]
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
    # the clips are the job's dataset now; they join the catalog when the job is published, and can be taken back
    assert len(catalog.catalog()) == before and last.json()["job"]["published"] is False
    assert client.get(f"/api/media/runs/{job['run']}/clips/c000/upscaled.mp4").status_code == 404
    assert client.post(f"/api/admin/jobs/{job['id']}/publish").status_code == 401
    assert client.post(f"/api/admin/jobs/{job['id']}/publish", headers=AUTH).json()["published"] is True
    assert len(catalog.catalog()) == before + 1 and catalog.catalog()[0]["id"] == f"{job['run']}-c000"
    assert client.get(f"/api/media/runs/{job['run']}/clips/c000/upscaled.mp4").status_code == 200
    assert client.get("/api/admin/status", headers=AUTH).json()["totals"]["clips"] == 1
    assert client.post(f"/api/admin/jobs/{job['id']}/unpublish", headers=AUTH).json()["published"] is False
    assert len(catalog.catalog()) == before
    assert client.put(f"/api/admin/jobs/{job['id']}/result", params={"total": 5}, content=b"again", headers=AUTH).status_code == 409

    # a compressed archive is refused: it could unpack to any size
    other = client.post("/api/admin/jobs", json={"footage": "cam.mp4", "where": "worker"}, headers=AUTH).json()["id"]
    client.post("/api/admin/worker/next", json={"worker": "mac"}, headers=AUTH)
    squeezed = io.BytesIO()
    with tarfile.open(fileobj=squeezed, mode="w:gz") as tar:
        info = tarfile.TarInfo("catalog.json")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"[]"))
    refused = client.put(f"/api/admin/jobs/{other}/result", params={"total": len(squeezed.getvalue())},
                         content=squeezed.getvalue(), headers=AUTH)
    assert refused.status_code == 422 and not (store / "incoming" / f"{other}.tar.part").exists()
    assert client.post(f"/api/admin/jobs/{other}/publish", headers=AUTH).status_code == 409      # not finished: nothing to publish


def test_the_job_page_follows_a_run_through_its_previews(store, monkeypatch):
    """A worker keeps a copy of its run's state and previews on the site while it works; the job page reads them."""
    import io
    import tarfile

    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"raw footage")
    job_id = client.post("/api/admin/jobs", json={"footage": "cam.mp4", "where": "worker", "multiply": 1}, headers=AUTH).json()["id"]
    waiting = client.get(f"/api/admin/jobs/{job_id}", headers=AUTH).json()
    assert waiting["job"]["status"] == "queued" and waiting["state"] == {} and waiting["files"] == {} and waiting["now"] is None
    put = lambda path, data: client.put(f"/api/admin/jobs/{job_id}/live", params={"path": path}, content=data, headers=AUTH)   # noqa: E731
    assert put("state.json", b"{}").status_code == 404                           # nobody has taken the job yet
    hello = {"worker": "mac", "device": "mps", "labeler": "claude-haiku-5-5", "fal": {"key": True, "spent_usd": 0.93, "cap_usd": 3.0}}
    job = client.post("/api/admin/worker/next", json=hello, headers=AUTH).json()["job"]
    assert job["options"]["multiply"] == 1
    seen = client.get("/api/admin/status", headers=AUTH).json()["workers"][0]
    assert seen["fal"] == {"key": True, "spent_usd": 0.93, "cap_usd": 3.0} and seen["labeler"] == "claude-haiku-5-5" and seen["busy"]

    state = {"name": job["run"], "status": "running", "elapsed": 12.5,
             "source": {"file": "cam.mp4", "path": "/Users/someone/data/uploads/cam.mp4", "seconds": 20.0, "size": [1920, 1080], "fps": 30.0},
             "stages": [{"key": "motion", "title": "Motion filter", "status": "done", "progress": 1.0, "seconds": 0.7},
                        {"key": "people", "title": "Person tracking", "status": "running", "progress": 0.4}],
             "motion": {"seconds": 20.0, "kept_seconds": 20.0, "timeline": [0.01, 0.02], "frame_spans": [[0, 600]]},
             "clips": [], "candidates": [], "people": [], "costs": {"anthropic_usd": 0.001, "fal_usd": 0.0},
             "log": [{"t": i, "msg": f"line {i} -> /Users/someone/data/runs/x"} for i in range(400)], "_private": "x",
             "eval": {"big": True}}
    now = {"stage": "people", "seq": 3, "t": 8.0, "people": [{"id": 1, "box": [1, 2, 3, 4]}], "image": "data:image/jpeg;base64,AAAA"}
    assert put("state.json", json.dumps(state).encode()).json() == {"kept": "state.json"}
    assert put("live/now.json", json.dumps(now).encode()).status_code == 200
    assert put("live/kit.json", json.dumps({"scan": {"file": "live/scan.mp4"}}).encode()).status_code == 200
    assert put("live/scan.mp4", b"scan video").status_code == 200
    assert put("clips/c000/poster.jpg", b"poster").status_code == 200
    # only plain names of known kinds, inside the job's own folder, and not without end
    for bad in ("../../jobs.json", "/etc/passwd.json", "live/../../x.json", ".hidden.json", "live/.part.json", "run.sh",
                "a/b/c/d/e/f.json", "live//x.json", "catalog", "live\\..\\x.json"):
        assert put(bad, b"{}").status_code == 404, bad
    assert put("live/empty.json", b"").status_code == 422
    assert put("live/huge.mp4", b"x" * (admin.MAX_PREVIEW + 1)).status_code == 413
    # what a job keeps here is counted: in bytes, and in files (an empty-looking file costs disk too)
    held_bytes, held_files = admin._previews[job_id]
    assert held_files == 5 and held_bytes == sum(path.stat().st_size for path in (store / "live" / job_id).rglob("*") if path.is_file())
    with monkeypatch.context() as tight:
        tight.setattr(admin, "MAX_PREVIEWS", held_bytes + 100)                   # room for 100 more bytes
        assert put("live/more.mp4", b"x" * 200).status_code == 507
        assert put("live/more.mp4", b"x" * 60).status_code == 200
        assert put("live/more.mp4", b"x" * 90).status_code == 200                # the copy it replaces does not count twice
        tight.setattr(admin, "MAX_PREVIEW_FILES", 6)
        assert put("live/one-more.jpg", b"x").status_code == 507 and put("live/more.mp4", b"y").status_code == 200
        tight.setattr(admin, "MAX_PREVIEW_FILES", 400)
        tight.setattr(admin, "MAX_ALL_PREVIEWS", admin._previews[job_id][0] + 5)  # all jobs together
        tight.setitem(admin._previews, "another-job", [1000, 3])
        assert put("live/one-more.jpg", b"x").status_code == 507
    assert put("state.json", b"[" + b"1," * admin.READ_LIMITS["state.json"] + b"1]").status_code == 413   # more than this site reads
    assert put("live/more.mp4/inner.json", b"{}").status_code == 422             # a name that lies under a file
    assert put("odd.json/inner.json", b"{}").status_code == 200 and put("odd.json", b"{}").status_code == 422   # or is a folder
    assert put("held/x.json", b"{}").status_code == 404                          # nothing may stand where the hold marker goes
    assert not [path for path in (store / "live" / job_id).rglob(".*")]          # no half-written file is left behind
    (store / "live" / job_id / "live" / ".hidden.json").write_text("{}")
    assert sorted(path.name for path in (store / "live" / job_id / "live").iterdir()) == [".hidden.json", "kit.json", "more.mp4", "now.json", "scan.mp4"]
    # nothing of it is in the run folder yet: only the marker that holds the run back, there since the job was made
    assert [path.name for path in (store / "runs" / job["run"]).iterdir()] == [library.HOLD]

    page = client.get(f"/api/admin/jobs/{job_id}", headers=AUTH).json()
    assert page["job"]["status"] == "running" and page["job"]["stages"][1]["progress"] == 0.4 and page["job"]["elapsed"] == 12.5
    assert page["job"]["device"] == "mps" and page["job"]["published"] is False
    assert "path" not in page["state"]["source"] and "frame_spans" not in page["state"]["motion"]      # no paths of the worker's machine
    assert "_private" not in page["state"] and "eval" not in page["state"] and len(page["state"]["log"]) == 150
    assert page["state"]["log"][-1] == {"t": 399, "msg": "line 399"}
    assert page["now"]["seq"] == 3 and page["now"]["image"].startswith("data:image/jpeg") and page["kit"]["scan"]["file"] == "live/scan.mp4"
    assert set(page["files"]) == {"state.json", "live/now.json", "live/kit.json", "live/scan.mp4", "live/more.mp4",
                                  "clips/c000/poster.jpg", "odd.json/inner.json"}        # no hidden file among them
    again = client.get(f"/api/admin/jobs/{job_id}", params={"have": "people:3"}, headers=AUTH).json()
    assert again["now"] == {"stage": "people", "seq": 3, "same": True}           # the frame the page already has is not sent twice
    # a frame of a stage that is over is not shown as the current one
    state["stages"][1]["status"] = "done"
    put("state.json", json.dumps(state).encode())
    assert client.get(f"/api/admin/jobs/{job_id}", headers=AUTH).json()["now"] is None

    file = lambda path, **how: client.get(f"/api/admin/jobs/{job_id}/file/{path}", **how)     # noqa: E731
    assert file("live/scan.mp4").status_code == 401
    assert file("live/scan.mp4", headers=AUTH).content == b"scan video"
    # a <video> cannot send the token: the page's browser gets a cookie that opens files, and only files
    assert client.post("/api/admin/session").status_code == 401
    opened = client.post("/api/admin/session", headers={**AUTH, "X-Forwarded-Proto": "https"})
    cookie = opened.headers["set-cookie"]
    assert all(part in cookie for part in ("HttpOnly", "SameSite=strict", "Secure", "Path=/api/admin")) and "s3cret" not in cookie
    ticket = {"Cookie": cookie.split(";")[0]}
    assert file("live/scan.mp4", headers=ticket).content == b"scan video"
    assert client.get("/api/admin/status", headers=ticket).status_code == 401
    assert client.post(f"/api/admin/jobs/{job_id}/stop", headers=ticket).status_code == 401
    assert file("live/scan.mp4", headers={"Cookie": f"{admin.MEDIA_COOKIE}=s3cret"}).status_code == 401
    with monkeypatch.context() as changed:                                       # a new token ends the old cookie
        changed.setattr(admin, "secret", lambda name: {"ADMIN_TOKEN": "another"}.get(name))
        assert file("live/scan.mp4", headers=ticket).status_code == 401
    # signing out ends every copy of the cookie, not only this browser's; the page is told to ask for a new one
    before = client.get("/api/admin/status", headers=AUTH).json()["session"]
    assert client.delete("/api/admin/session", headers=ticket).status_code == 401
    signed_out = client.delete("/api/admin/session", headers=AUTH)
    assert signed_out.status_code == 200 and signed_out.headers["clear-site-data"] == '"cache"'      # and what the browser kept
    assert file("live/scan.mp4", headers=ticket).status_code == 401
    assert client.get("/api/admin/status", headers=AUTH).json()["session"] != before
    client.cookies.clear()
    assert file("clips/c000/poster.jpg", headers=AUTH).headers["content-type"] == "image/jpeg"
    for bad in ("..%2F..%2Fjobs.json", "live/..%2F..%2F..%2Fjobs.json", "%2Fetc%2Fhosts.json", "live/missing.mp4", "state.txt",
                "state.json", "live/now.json", "live/kit.json", "live/.hidden.json", "held"):
        assert file(bad, headers=AUTH).status_code == 404, bad      # the state is given read and trimmed, never as a file
    assert client.get("/api/admin/jobs/nope/file/live/scan.mp4", headers=AUTH).status_code == 404
    (store / "outside.json").write_text('{"secret": true}')
    (store / "live" / job_id / "live" / "link.json").symlink_to(store / "outside.json")
    assert file("live/link.json", headers=AUTH).status_code == 404               # a link out of the job's folder is not followed

    # the finished run arrives: the page reads the run folder from now on, and the copy is gone
    data = archive_of({"catalog.json": b"[]", "state.json": json.dumps({**state, "status": "done"}).encode(),
                       "live/kit.json": b'{"scan": {"file": "live/scan.mp4"}}', "live/scan.mp4": b"the full scan",
                       "held": b"", "held/x.json": b"{}", "run.log": b"done"})
    assert client.put(f"/api/admin/jobs/{job_id}/result", params={"total": len(data)}, content=data, headers=AUTH).json()["done"]
    assert not (store / "live" / job_id).exists() and job_id not in admin._previews
    assert (store / "runs" / job["run"] / library.HOLD).is_file()                # the marker is the site's own, a plain file
    assert file("live/scan.mp4", headers=AUTH).content == b"the full scan" and file("run.log", headers=AUTH).content == b"done"
    assert put("state.json", b"{}").status_code == 409                           # the run is over
    assert client.post(f"/api/admin/jobs/{job_id}/publish", headers=AUTH).status_code == 409    # no accepted clip in it
    assert client.delete(f"/api/admin/jobs/{job_id}", headers=AUTH).status_code == 200
    assert not (store / "runs" / job["run"]).exists() and not list((store / "trash").iterdir())


def test_a_job_taken_again_starts_with_no_previews(store):
    """What an earlier attempt left behind is not shown as the new attempt's, and a deleted job leaves nothing."""
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"raw footage")
    job_id = client.post("/api/admin/jobs", json={"footage": "cam.mp4", "where": "worker"}, headers=AUTH).json()["id"]
    stale = store / "live" / job_id / "live"
    stale.mkdir(parents=True)
    (stale / "scan.mp4").write_bytes(b"from before")
    client.post("/api/admin/worker/next", json={"worker": "mac"}, headers=AUTH)
    assert not (store / "live" / job_id).exists()
    put = lambda path, data: client.put(f"/api/admin/jobs/{job_id}/live", params={"path": path}, content=data, headers=AUTH)   # noqa: E731
    assert put("live/scan.mp4", b"new").status_code == 200
    assert client.post(f"/api/admin/jobs/{job_id}/failed", json={"error": "it broke"}, headers=AUTH).json()["status"] == "failed"
    assert (store / "live" / job_id / "live" / "scan.mp4").read_bytes() == b"new"      # a failed job keeps what it showed
    assert client.delete(f"/api/admin/jobs/{job_id}", headers=AUTH).status_code == 200
    assert not (store / "live" / job_id).exists() and job_id not in admin._previews


def test_a_state_that_is_not_the_engines_does_not_take_the_console_down(store):
    """A worker's files are read, never trusted. Whatever is in them, every route keeps answering, in JSON."""
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"raw footage")
    job = worker_job()
    job_id = job["id"]
    put = lambda path, data: client.put(f"/api/admin/jobs/{job_id}/live", params={"path": path}, content=data, headers=AUTH)   # noqa: E731

    def answers():
        listed = client.get("/api/admin/jobs", headers=AUTH)
        page = client.get(f"/api/admin/jobs/{job_id}", headers=AUTH)
        assert listed.status_code == 200 and page.status_code == 200 and client.get("/api/admin/status", headers=AUTH).status_code == 200
        return listed.json()["jobs"][0], page.json()

    odd = [json.dumps(value) for value in (
        [1, 2, 3], "text", {"stages": ["x", 7, None], "clips": "many", "costs": [1], "source": "cam", "engine": 5},
        {"stages": [{"key": "people", "status": "running"}, {"key": ["a", "list"], "status": "running"}],
         "clips": [{"variants": "none"}, {"variants": [3]}], "costs": {"anthropic_usd": "a lot", "fal_usd": True},
         "elapsed": "long", "source": {"size": "big", "seconds": []}})]
    odd += ['{"elapsed": NaN, "costs": {"anthropic_usd": Infinity, "fal_usd": -Infinity}, "source": {"seconds": 1e999}}',
            '{"stages": [{"key": "people", "status": "running", "progress": NaN}], "clips": [{"status": "accepted", "seconds": NaN}]}',
            '{"deep": ' + "[" * 500 + "]" * 500 + "}", "[" * 100_000, '{"a": ' * 30 + "1" + "}" * 30,
            '{"clips": ' + "[" * 600 + "]" * 600 + "}", '{"log": [{"t": 1, "msg": ' + '{"a": ' * 400 + "1" + "}" * 400 + "}]}",
            "[" + "[]," * 200_000 + "[]]", "not json at all", '{"big": ' + "9" * 5000 + "}"]
    for text in odd:
        assert put("state.json", text.encode()).status_code == 200
        listed, page = answers()
        assert listed["claude_usd"] is None and listed["elapsed"] is None
        assert all(isinstance(stage["key"], str) for stage in listed["stages"])
    # a number that is not one becomes "no number", wherever it sits, and the rest of the state is still shown
    put("state.json", b'{"name": "x", "elapsed": 4.5, "clips": [{"id": "c000", "seconds": NaN, "label": {"quality": Infinity}}]}')
    listed, page = answers()
    assert listed["elapsed"] == 4.5 and page["state"]["clips"] == [{"id": "c000", "seconds": None, "label": {"quality": None}}]

    for path, text in (("live/now.json", "[1]"), ("live/kit.json", "not json"), ("live/now.json", '{"stage": ["people"], "seq": 1}'),
                       ("live/now.json", '{"stage": "people", "seq": NaN}'), ("live/kit.json", '{"scan": NaN}')):
        assert put(path, text.encode()).status_code == 200
        listed, page = answers()
        assert page["now"] is None and page["kit"] in ({}, {"scan": None})

    # the worker's short report is kept with the job and sent on: it is taken apart, not passed through
    report = lambda body: client.post(f"/api/admin/jobs/{job_id}/progress", content=body, headers={**AUTH, "Content-Type": "application/json"})   # noqa: E731
    assert report('{"claude_usd": NaN}').status_code == 422
    assert report('{"stages": [3]}').status_code == 422
    assert report('{"stages": [{"key": "motion", "status": "done", "progress": NaN, "detail": ["x"]}, {"key": 5}], '
                  '"clips": {"planned": NaN, "accepted": "many", "rejected": 2}, "source": {"seconds": Infinity, "start_s": 3}}').status_code == 200
    kept = next(entry for entry in admin._jobs if entry["id"] == job_id)["remote"]
    assert kept["stages"] == [{"key": "motion", "title": None, "status": "done", "progress": None, "detail": None, "seconds": None}]
    assert kept["clips"] == {"planned": 0, "accepted": 0, "rejected": 2} and kept["source"] == {"start_s": 3, "seconds": None}
    json.dumps(admin._jobs, allow_nan=False)                                     # nothing that cannot be written down as JSON
    answers()

    # the same in a finished result: it is taken, the job is done, and everything still answers
    data = archive_of({"catalog.json": b'[{"id": "x", "seconds": NaN}, ' + b"[" * 3000 + b"]" * 3000 + b"]",
                       "state.json": b'{"elapsed": NaN, "costs": {"anthropic_usd": Infinity}, "clips": [{"status": "accepted"}]}'})
    done = client.put(f"/api/admin/jobs/{job_id}/result", params={"total": len(data)}, content=data, headers=AUTH)
    assert done.status_code == 200 and done.json()["job"]["status"] == "done"
    listed, page = answers()
    assert listed["status"] == "done" and listed["clips"]["accepted"] == 1 and listed["claude_usd"] is None
    assert client.get("/api/admin/status", headers=AUTH).json()["totals"]["usd"] == 0
    # a catalog nothing can be made of is not published, and the public catalog never notices it
    before = len(catalog.catalog())
    assert client.post(f"/api/admin/jobs/{job_id}/publish", headers=AUTH).status_code == 409
    assert len(catalog.catalog()) == before and client.get("/api/clips").status_code == 200
    assert client.delete(f"/api/admin/jobs/{job_id}", headers=AUTH).status_code == 200


def test_one_job_that_cannot_be_read_is_still_listed(store, monkeypatch):
    """Whatever goes wrong reading one job's run, the list shows the job, so it can be deleted from the page."""
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"raw footage")
    first, second = worker_job()["id"], worker_job()["id"]
    real = admin.picture
    monkeypatch.setattr(admin, "picture", lambda job: real(job) if job["id"] != first else (_ for _ in ()).throw(RuntimeError("unreadable")))
    listed = client.get("/api/admin/jobs", headers=AUTH)
    assert listed.status_code == 200 and [job["id"] for job in listed.json()["jobs"]] == [second, first]
    broken = listed.json()["jobs"][1]
    assert "cannot be read" in broken["error"] and broken["stages"] == [] and broken["published"] is False


def test_footage_says_where_it_comes_from(store):
    assert admin.described("2018-03-07.17-30-00.17-35-00.school.G421.avi") == {
        "dataset": "MEVA", "camera": "G421", "recorded": "2018-03-07 17:30", "license": "CC BY 4.0"}
    assert admin.described("hallway-snack-table.mp4") is None
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"not really a video")
    assert client.get("/api/admin/footage/cam.mp4/film", headers=AUTH).status_code == 404      # nothing ffmpeg can read
    assert client.get("/api/admin/footage/cam.mp4/film").status_code == 401


def test_the_strip_of_small_frames_is_made_once_and_gently(store, monkeypatch):
    """Ten frames, one at a time, by one ffmpeg at a time. A file that gives no strip is not tried again at once:
    asking for the picture (which the page's cookie alone can do) must not keep the machine busy."""
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"footage")
    (store / "uploads" / "bad.mp4").write_bytes(b"footage that ffmpeg cannot read")
    runs = store / "ffmpeg-runs.txt"
    tool = store / "ffmpeg"
    # stands in for ffmpeg: notes each run, and writes its last argument unless the input is bad.mp4
    tool.write_text(f"""#!/bin/sh
echo "$@" >> {runs}
for last; do :; done
case "$*" in *bad.mp4*) exit 1;; esac
printf picture > "$last"
""")
    tool.chmod(0o755)
    monkeypatch.setattr(admin, "ffmpeg", lambda: str(tool))
    monkeypatch.setattr(admin, "probe", lambda path: {"seconds": 100.0})
    monkeypatch.setattr(admin, "memory", lambda: (8 * 1024 * admin.MB, 4000 * admin.MB))
    count = lambda: len(runs.read_text().splitlines()) if runs.exists() else 0      # noqa: E731
    film = lambda name: client.get(f"/api/admin/footage/{name}/film", headers=AUTH)   # noqa: E731

    assert film("cam.mp4").content == b"picture" and count() == 11               # ten frames, then one strip of them
    lines = runs.read_text().splitlines()
    assert all(line.count(" -i ") == 1 and "-threads 1" in line for line in lines[:10])     # one decoder per run, one thread
    assert [line.split(" -ss ")[1].split()[0] for line in lines[:10]] == [f"{5 + 10 * i:.2f}" for i in range(10)]
    assert lines[10].count(" -i ") == 10 and "cam.mp4" not in lines[10]          # the strip is made of the ten small frames
    assert film("cam.mp4").status_code == 200 and count() == 11                  # made once
    assert not [path for path in (store / "uploads" / ".film").iterdir() if path.name != "cam.mp4.jpg"]   # nothing left over

    assert film("bad.mp4").status_code == 404 and count() == 12                  # it stops at the first frame that fails
    assert [film("bad.mp4").status_code for _ in range(5)] == [404] * 5 and count() == 12      # and is not tried again
    # with little memory free, nothing is started at all
    (store / "uploads" / "later.mp4").write_bytes(b"footage")
    monkeypatch.setattr(admin, "memory", lambda: (8 * 1024 * admin.MB, 200 * admin.MB))
    assert film("later.mp4").status_code == 404 and count() == 12
    assert client.delete("/api/admin/footage/cam.mp4", headers=AUTH).status_code == 200
    assert not (store / "uploads" / ".film" / "cam.mp4.jpg").exists()


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


def test_a_request_body_cannot_be_larger_than_its_route_needs(store):
    """Anyone can post to the public routes, and a JSON body is read before the admin token is checked."""
    big = "x" * (2 * 1024 * 1024)
    assert client.post("/api/compile", json={"prompt": big}).status_code == 413
    assert client.post("/api/admin/jobs", json={"footage": big}).status_code == 413          # even without a token
    assert client.post("/api/admin/jobs", content=iter([b"x" * 600_000] * 3)).status_code == 413   # no length declared
    assert client.post("/api/dataset-preview", json={"plan": None}).status_code == 200       # ordinary requests pass
    piece = b"v" * (3 * 1024 * 1024)                                                         # pieces of an upload may be large
    assert client.put("/api/admin/footage", params={"name": "cam.mp4", "total": len(piece) + 1}, content=piece,
                      headers=AUTH).json() == {"done": False, "offset": len(piece)}
    assert client.put("/api/admin/footage", params={"name": "cam.mp4", "total": 10**9}, content=b"v" * (40 * 1024 * 1024),
                      headers=AUTH).status_code == 413


def test_values_in_a_request_are_names_not_documents(store, monkeypatch, tmp_path):
    long = "x" * 5000
    assert client.post("/api/dataset-preview", json={"plan": {"summary": "s", "objects": ["paper cup", long]}}).status_code == 422
    assert client.post("/api/dataset-preview", json={"excluded_ids": [long]}).status_code == 422
    # the public request box does not grow without end
    monkeypatch.setattr(datasets, "ROOT", tmp_path)
    monkeypatch.setitem(datasets.SETTINGS, "max_saved_requests", 2)
    body = {"prompt": "cups", "selection": {"plan": {"summary": "Cups.", "actions": ["Pick & place"]}}, "email": "a@lab.com"}
    assert [client.post("/api/dataset-requests", json=body).status_code for _ in range(3)] == [201, 201, 503]


def test_only_files_a_catalog_item_names_are_public(store):
    """A run folder also holds the run's state, its log and rejected clips. None of that is handed out."""
    seed = json.loads((library.SEED / "catalog.json").read_text())[0]
    run = store / "runs" / "shop"
    (run / "clips" / "c000").mkdir(parents=True)
    (run / "clips" / "c001").mkdir()
    for name in ("clips/c000/upscaled.mp4", "clips/c001/upscaled.mp4", "state.json", "run.log"):
        (run / name).write_bytes(b"{}")
    good = {**seed, "id": "shop-c000", "run": "shop", "files": {"video": "clips/c000/upscaled.mp4"},
            "lineage": {"source": "shop.mp4", "frames": [0, 90], "crop_xywh": [0, 0, 320, 180]}}
    broken = {"id": "shop-c009", "kind": "original", "labels": [{"verb": "pick_up"}], "files": {},
              "lineage": {"source": "shop.mp4", "frames": [90, 180], "crop_xywh": [0, 0, 320, 180]}}
    (run / "catalog.json").write_text(json.dumps([good, broken, "not an item"]))
    ids = {clip["id"] for clip in client.get("/api/clips", params={"limit": 100}).json()["clips"]}
    assert "shop-c000" in ids and "shop-c009" not in ids                 # the damaged item is left out, the site stays up
    assert client.get("/api/media/runs/shop/clips/c000/upscaled.mp4").status_code == 200
    for private in ("state.json", "catalog.json", "run.log", "clips/c001/upscaled.mp4"):
        assert client.get(f"/api/media/runs/shop/{private}").status_code == 404
    assert client.get("/api/media/seed/catalog.json").status_code == 404


def test_a_keyword_answer_given_while_the_ai_was_down_does_not_stick(monkeypatch):
    calls = []

    def flaky(prompt, key):
        calls.append(prompt)
        if len(calls) == 1:
            raise RuntimeError("the AI service is having a moment")
        return datasets.Plan(summary="From the AI.", actions=["Pick & place"])

    monkeypatch.setattr(datasets, "secret", lambda name: "key" if name == "ANTHROPIC_API_KEY" else None)
    monkeypatch.setattr(datasets, "claude_plan", flaky)
    monkeypatch.setattr(datasets, "_compiled", {})
    monkeypatch.setattr(datasets, "_ai_calls", [])
    ask = lambda: client.post("/api/compile", json={"prompt": "picking up a cup"}).json()   # noqa: E731
    assert ask()["mode"] == "demo"                                    # the keyword interpreter answered
    assert ask()["mode"] == "ai" and ask()["plan"]["summary"] == "From the AI."
    assert len(calls) == 2                                            # asked again after the failure, then answered from memory
    monkeypatch.setitem(datasets.SETTINGS, "ai_calls_per_hour", 2)   # the hourly cap is reached: keywords, and not remembered
    assert client.post("/api/compile", json={"prompt": "something else"}).json()["mode"] == "demo" and len(datasets._compiled) == 1


def test_a_damaged_result_is_never_half_published(store, monkeypatch):
    import io
    import tarfile

    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"raw footage")
    job_id = client.post("/api/admin/jobs", json={"footage": "cam.mp4", "where": "worker"}, headers=AUTH).json()["id"]
    job = client.post("/api/admin/worker/next", json={"worker": "mac"}, headers=AUTH).json()["job"]
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for name, data in {"clips/c000/upscaled.mp4": b"video" * 1000, "state.json": b"{}"}.items():   # no catalog.json
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    data = archive.getvalue()
    refused = client.put(f"/api/admin/jobs/{job_id}/result", params={"total": len(data)}, content=data, headers=AUTH)
    assert refused.status_code == 422
    assert [path.name for path in (store / "runs" / job["run"]).iterdir()] == [library.HOLD]    # nothing of it got in
    assert not list((store / "incoming").iterdir())                   # nothing left behind
    # the job thread outlives a disk that cannot be written
    monkeypatch.setattr(admin, "save_jobs", lambda: (_ for _ in ()).throw(OSError("disk full")))
    assert client.get("/api/admin/jobs", headers=AUTH).status_code == 200
    assert any(thread.name == "engine-jobs" and thread.is_alive() for thread in __import__("threading").enumerate())


def finished(store, job, files):
    """Send `files` as the finished run of a worker's job. Returns the reply."""
    data = archive_of(files)
    return client.put(f"/api/admin/jobs/{job['id']}/result", params={"total": len(data)}, content=data, headers=AUTH)


def test_a_run_is_private_from_the_moment_its_job_exists(store, monkeypatch):
    """Wherever the engine writes the run (this machine, or a worker sharing its data folder), nothing of it is in
    the catalog before the job is published."""
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"raw footage")
    before = len(catalog.catalog())
    for where in ("worker", "server"):
        monkeypatch.setattr(admin, "secret", lambda name: {"ADMIN_TOKEN": "s3cret", "ANTHROPIC_API_KEY": "key"}.get(name))
        monkeypatch.setattr(admin, "engine_python", lambda: "/no/such/python")
        job = client.post("/api/admin/jobs", json={"footage": "cam.mp4", "where": where}, headers=AUTH).json()
        run = store / "runs" / job["run"]
        assert (run / library.HOLD).is_file()                        # held before anything can be written into it
        # the engine finishes and leaves its catalog in the run folder, as it does for a run made by hand
        (run / "clips" / "c000").mkdir(parents=True)
        (run / "clips" / "c000" / "upscaled.mp4").write_bytes(b"video")
        (run / "catalog.json").write_text(json.dumps([seed_item(job["run"])]))
        assert len(catalog.catalog()) == before
        assert client.get(f"/api/media/runs/{job['run']}/clips/c000/upscaled.mp4").status_code == 404
        assert client.post(f"/api/admin/jobs/{job['id']}/publish", headers=AUTH).status_code == 409     # not finished
    with admin._lock:
        entry = next(item for item in admin._jobs if item["id"] == job["id"])
        entry.update(status="done", error=None)
    assert client.post(f"/api/admin/jobs/{job['id']}/publish", headers=AUTH).json()["published"] is True
    assert len(catalog.catalog()) == before + 1
    served = client.get(f"/api/media/runs/{job['run']}/clips/c000/upscaled.mp4")
    assert served.status_code == 200 and served.headers["cache-control"] == "public, max-age=300"   # it may be taken back
    assert client.get(catalog.catalog()[-1]["video_url"]).headers["cache-control"] == "public, max-age=3600"   # the seed library stays


def test_a_published_run_hands_out_its_own_files_only(store):
    """A catalog can come from a worker. Naming a file of another run in it, published or held, gets nobody that file."""
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"raw footage")
    private = worker_job()
    assert finished(store, private, {"catalog.json": json.dumps([seed_item(private["run"])]).encode(),
                                     "state.json": b'{"clips": [{"status": "accepted"}, {"status": "rejected"}]}',
                                     "clips/c000/upscaled.mp4": b"a clip nobody published",
                                     "clips/c001/upscaled.mp4": b"a rejected clip"}).json()["done"]
    public = worker_job()
    reach = f"../{private['run']}/clips/c000/upscaled.mp4"
    items = [seed_item(public["run"]),
             seed_item(public["run"], "c001", files={"video": reach}),
             seed_item(public["run"], "c002", files={"video": "clips/c002/upscaled.mp4", "card": f"../{private['run']}/state.json"}),
             seed_item(public["run"], "c003", files={"video": "/etc/hosts"}),
             seed_item(public["run"], "c004", files={"video": "clips\\c004\\upscaled.mp4"}),
             seed_item(public["run"], "c005", files={"video": ["clips/c005/upscaled.mp4"]}),
             seed_item(public["run"], "c006", files="clips/c006/upscaled.mp4")]
    assert finished(store, public, {"catalog.json": json.dumps(items).encode(), "state.json": b'{"clips": [{"status": "accepted"}]}',
                                    "clips/c000/upscaled.mp4": b"its own clip"}).json()["done"]
    before = len(catalog.catalog())
    assert client.post(f"/api/admin/jobs/{public['id']}/publish", headers=AUTH).json()["published"] is True
    ids = {clip["id"] for clip in catalog.catalog()}
    assert len(catalog.catalog()) == before + 1 and f"{public['run']}-c000" in ids      # only the clip with plain paths of its own
    assert client.get(f"/api/media/runs/{public['run']}/clips/c000/upscaled.mp4").content == b"its own clip"
    for path in (f"{public['run']}/..%2F{private['run']}/clips/c000/upscaled.mp4", f"{public['run']}/../{private['run']}/clips/c000/upscaled.mp4",
                 f"{public['run']}/..%2F{private['run']}/state.json", f"{private['run']}/clips/c000/upscaled.mp4",
                 f"{private['run']}/clips/c001/upscaled.mp4", f"{public['run']}/state.json"):
        assert client.get(f"/api/media/runs/{path}").status_code == 404, path
    # even a name that did get on the list is not served out of another run's folder, or out of a held run
    library._cache["files"].add(("runs", f"{public['run']}/{reach}"))
    assert library.media_file("runs", f"{public['run']}/{reach}") is None
    assert client.post(f"/api/admin/jobs/{private['id']}/publish", headers=AUTH).json()["published"] is True
    assert client.get(f"/api/media/runs/{private['run']}/clips/c000/upscaled.mp4").status_code == 200
    assert client.get(f"/api/media/runs/{private['run']}/clips/c001/upscaled.mp4").status_code == 404      # rejected: never public
    assert client.post(f"/api/admin/jobs/{private['id']}/unpublish", headers=AUTH).json()["published"] is False
    library.clips()                                                              # the list of files is up to date...
    library._cache["files"].add(("runs", f"{private['run']}/clips/c000/upscaled.mp4"))      # ...and then, as if it were stale
    assert library.media_file("runs", f"{private['run']}/clips/c000/upscaled.mp4") is None


def test_a_catalog_with_the_wrong_kind_of_values_loses_those_clips_only(store):
    """What the public routes send, sort and filter on is checked once, when the catalog is built."""
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"raw footage")
    job = worker_job()
    run, good = job["run"], seed_item(job["run"])
    label = good["labels"][0]
    odd = [dict(seed_item(run, f"c{n:03d}"), **change) for n, change in enumerate([
        {"seconds": "long"}, {"seconds": None}, {"fps": [30]}, {"quality": {"a": 1}}, {"caption": ["a", "list"]},
        {"objects": [["nested"]]}, {"labels": [{**label, "start": "soon"}]}, {"labels": [[label]]}, {"labels": "none"},
        {"scene": {"setting": ["x"]}}, {"tiers": [1, 2]}, {"lineage": {"source": "x", "frames": [0.5, "end"], "crop_xywh": [0, 0, 1, 1]}},
        {"main_object": 5}, {"resolution": [1920]}, {"caption": "x" * 5000}, {"labels": [label] * 200}], start=1)]
    text = json.dumps([good, *odd])[:-1] + ', {"id": "nan", "seconds": NaN, "files": {"video": "clips/c900/upscaled.mp4"}}]'
    assert finished(store, job, {"catalog.json": text.encode(), "state.json": b'{"clips": [{"status": "accepted"}]}',
                                 "clips/c000/upscaled.mp4": b"video"}).json()["done"]
    before = len(catalog.catalog())
    assert client.post(f"/api/admin/jobs/{job['id']}/publish", headers=AUTH).json()["published"] is True
    clips = catalog.catalog()
    assert len(clips) == before + 1 and clips[0]["id"] == f"{run}-c000"
    for route in ("/api/clips?limit=100", "/api/clips?sort=duration", "/api/clips?sort=newest", "/api/facets", f"/api/clips/{run}-c000",
                  "/api/clips?q=cup"):
        assert client.get(route).status_code == 200, route
    assert client.post("/api/dataset-preview", json={"plan": None}).status_code == 200
    # a record keeps the shape the explorer reads: a version without action segments of its own has no such key
    record = client.get(f"/api/clips/{run}-c000").json()
    assert "segments" not in record["versions"][0] and record["segments"] and record["duration_seconds"] == good["seconds"]


def test_the_catalog_reads_one_listing_of_its_sources(store, monkeypatch):
    """What is cached is exactly what its stamp stands for: a run published in between cannot slip in under an old stamp."""
    calls = []
    real = library.sources
    monkeypatch.setattr(library, "sources", lambda: (calls.append(1), real())[1])
    monkeypatch.setitem(library._cache, "stamp", None)
    library.clips()
    assert len(calls) == 1


def test_a_run_is_removed_in_one_step(store, monkeypatch):
    """A run folder taken apart file by file could lose its hold marker before its catalog, and be public meanwhile."""
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"raw footage")
    job = worker_job()
    assert finished(store, job, {"catalog.json": json.dumps([seed_item(job["run"])]).encode(), "state.json": b"{}",
                                 "clips/c000/upscaled.mp4": b"video"}).json()["done"]
    run, seen = store / "runs" / job["run"], []
    real = admin.shutil.rmtree

    def watched(folder, *args, **kwargs):
        seen.append((str(folder), run.exists()))      # is the run still where the catalog looks, while it is taken apart?
        return real(folder, *args, **kwargs)

    monkeypatch.setattr(admin.shutil, "rmtree", watched)
    assert client.delete(f"/api/admin/jobs/{job['id']}", headers=AUTH).status_code == 200
    removed = [entry for entry in seen if job["run"] in entry[0]]
    assert removed and all(str(store / "trash") in folder and not still_there for folder, still_there in removed)
    assert not run.exists() and not list((store / "trash").iterdir())


def test_a_result_is_refused_when_it_is_not_what_a_run_looks_like(store, monkeypatch):
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"raw footage")
    base = {"catalog.json": b"[]", "state.json": b"{}"}

    def refused(files, **limits):
        job = worker_job()
        with monkeypatch.context() as tight:
            for name, value in limits.items():
                tight.setattr(admin, name, value)
            reply = finished(store, job, files)
        assert [path.name for path in (store / "runs" / job["run"]).iterdir()] == [library.HOLD] and not list((store / "incoming").iterdir())
        client.post(f"/api/admin/jobs/{job['id']}/failed", json={"error": "refused"}, headers=AUTH)
        return reply

    # more files than a run has: empty ones cost disk too, and no size budget notices them
    many = refused({**base, **{f"clips/c{n:03d}/poster.jpg": b"" for n in range(6)}}, MAX_RESULT_FILES=5)
    assert many.status_code == 422 and "more than 5 files" in many.json()["detail"]
    # a state or a catalog larger than this site reads
    large = refused({**base, "state.json": b" " * 5000}, READ_LIMITS={**admin.READ_LIMITS, "state.json": 4000})
    assert large.status_code == 422 and "state.json is larger" in large.json()["detail"]
    # no room for the archive and its files side by side
    packed = store / "result.tar"
    packed.write_bytes(archive_of(base))
    with monkeypatch.context() as tight, pytest.raises(admin.tarfile.TarError, match="free disk space"):
        tight.setattr(admin, "FREE_MARGIN", 10 ** 18)
        admin.unpack(packed, store / "runs" / "elsewhere")
    assert not (store / "runs" / "elsewhere").exists()
    # names that are not a run's: left out, the rest is taken
    job = worker_job()
    taken = finished(store, job, {**base, "deep/a/b/c/d/e/f.json": b"{}", ".hidden.json": b"{}", "clips/../../x.json": b"{}",
                                  "held/x.json": b"{}", "clips/c000/.part.mp4": b"x", "clips/c000/poster.jpg": b"ok"})
    assert taken.json()["done"]
    kept = sorted(path.relative_to(store / "runs" / job["run"]).as_posix() for path in (store / "runs" / job["run"]).rglob("*") if path.is_file())
    assert kept == ["catalog.json", "clips/c000/poster.jpg", library.HOLD, "state.json"]


def test_json_from_outside_is_read_with_limits(tmp_path, monkeypatch):
    """documents.py: what a worker's files may hold, and what becomes of the rest."""
    assert documents.load('{"a": [1, 2.5, "x", null, true]}') == {"a": [1, 2.5, "x", None, True]}
    # numbers that are none: they could not be sent on as JSON
    assert documents.load('[NaN, Infinity, -Infinity, 1e999, -1e999, 1e308]') == [None, None, None, None, None, 1e308]
    for text in ("[" * 30 + "]" * 30, '{"a": ' * 30 + "1" + "}" * 30, "[" * 100_000, "not json", '{"a": 1', "9" * 5000):
        with pytest.raises(ValueError):
            documents.load(text)
    assert documents.load("[" * 24 + "]" * 24)                                    # as deep as is read, and no deeper
    monkeypatch.setattr(documents, "MAX_PARTS", 50)
    assert len(documents.load("[" + "[]," * 48 + "[]]")) == 49
    with pytest.raises(documents.Unreadable, match="too many parts"):
        documents.load("[" + "[]," * 50 + "[]]")
    file = tmp_path / "state.json"
    file.write_text('{"elapsed": 3}')
    assert documents.read(file, 100) == {"elapsed": 3} and documents.read(file, 5) is None        # larger than is read
    assert documents.read(tmp_path / "missing.json", 100) is None
    file.write_bytes(b"\xff\xfe not text")
    assert documents.read(file, 100) is None
    assert documents.rows([{"a": 1}, 3, "x", None, [1]]) == [{"a": 1}] and documents.rows("text") == [] and documents.record([1]) == {}
    assert [documents.number(value) for value in (3, 2.5, True, "3", None, float("nan"), float("inf"))] == [3, 2.5, None, None, None, None, None]
    part = admin.NAME_PART
    assert all(documents.plain_path(rel, part) for rel in ("state.json", "clips/c000/upscaled.mp4", "a/b/c/d/e/f.json"))
    assert not any(documents.plain_path(rel, part) for rel in ("../x.json", "a/../x.json", "/etc/hosts", ".hidden", "a//b", "a\\b", "",
                                                               "a/b/c/d/e/f/g.json", "x" * 100, 5, None, ["a"]))
