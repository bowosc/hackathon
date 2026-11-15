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
                                                              "PARTIAL": "uploads/.partial", "JOBS_FILE": "jobs.json",
                                                              "INCOMING": "incoming", "LIVE": "live",
                                                              "FILM": "uploads/.film"})):
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
    with monkeypatch.context() as tight:
        tight.setattr(admin, "MAX_PREVIEWS", 100)
        assert put("live/more.mp4", b"x" * 200).status_code == 507
    assert sorted(path.name for path in (store / "live" / job_id / "live").iterdir()) == ["kit.json", "now.json", "scan.mp4"]
    assert not (store / "runs").exists()                                         # nothing of it is in a run folder yet

    page = client.get(f"/api/admin/jobs/{job_id}", headers=AUTH).json()
    assert page["job"]["status"] == "running" and page["job"]["stages"][1]["progress"] == 0.4 and page["job"]["elapsed"] == 12.5
    assert page["job"]["device"] == "mps" and page["job"]["published"] is False
    assert "path" not in page["state"]["source"] and "frame_spans" not in page["state"]["motion"]      # no paths of the worker's machine
    assert "_private" not in page["state"] and "eval" not in page["state"] and len(page["state"]["log"]) == 150
    assert page["state"]["log"][-1] == {"t": 399, "msg": "line 399"}
    assert page["now"]["seq"] == 3 and page["now"]["image"].startswith("data:image/jpeg") and page["kit"]["scan"]["file"] == "live/scan.mp4"
    assert set(page["files"]) == {"state.json", "live/now.json", "live/kit.json", "live/scan.mp4", "clips/c000/poster.jpg"}
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
    client.cookies.clear()
    assert file("clips/c000/poster.jpg", headers=AUTH).headers["content-type"] == "image/jpeg"
    for bad in ("..%2F..%2Fjobs.json", "live/..%2F..%2F..%2Fjobs.json", "%2Fetc%2Fhosts.json", "live/missing.mp4", "state.txt"):
        assert file(bad, headers=AUTH).status_code == 404, bad
    assert client.get("/api/admin/jobs/nope/file/state.json", headers=AUTH).status_code == 404

    # the finished run arrives: the page reads the run folder from now on, and the copy is gone
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for name, data in {"catalog.json": b"[]", "state.json": json.dumps({**state, "status": "done"}).encode(),
                           "live/kit.json": b'{"scan": {"file": "live/scan.mp4"}}', "live/scan.mp4": b"the full scan",
                           "held": b"", "run.log": b"done"}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    data = archive.getvalue()
    assert client.put(f"/api/admin/jobs/{job_id}/result", params={"total": len(data)}, content=data, headers=AUTH).json()["done"]
    assert not (store / "live" / job_id).exists() and (store / "runs" / job["run"] / library.HOLD).is_file()
    assert file("live/scan.mp4", headers=AUTH).content == b"the full scan" and file("run.log", headers=AUTH).content == b"done"
    assert put("state.json", b"{}").status_code == 409                           # the run is over
    assert client.post(f"/api/admin/jobs/{job_id}/publish", headers=AUTH).status_code == 409    # no accepted clip in it
    assert client.delete(f"/api/admin/jobs/{job_id}", headers=AUTH).status_code == 200
    assert not (store / "runs" / job["run"]).exists()


def test_a_state_that_is_not_the_engines_does_not_take_the_jobs_list_down(store):
    """A worker's state file is read, never trusted: whatever is in it, the console keeps answering."""
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"raw footage")
    job_id = client.post("/api/admin/jobs", json={"footage": "cam.mp4", "where": "worker"}, headers=AUTH).json()["id"]
    client.post("/api/admin/worker/next", json={"worker": "mac"}, headers=AUTH)
    for odd in ([1, 2, 3], "text", {"stages": ["x", 7, None], "clips": "many", "costs": [1], "source": "cam", "engine": 5},
                {"stages": [{"key": "people", "status": "running"}], "clips": [{"variants": "none"}, {"variants": [3]}],
                 "costs": {"anthropic_usd": "a lot", "fal_usd": True}, "elapsed": "long", "source": {"size": "big", "seconds": []}}):
        assert client.put(f"/api/admin/jobs/{job_id}/live", params={"path": "state.json"}, content=json.dumps(odd).encode(),
                          headers=AUTH).status_code == 200
        listed = client.get("/api/admin/jobs", headers=AUTH)
        assert listed.status_code == 200 and listed.json()["jobs"][0]["claude_usd"] is None
        assert client.get(f"/api/admin/jobs/{job_id}", headers=AUTH).status_code == 200
        assert client.get("/api/admin/status", headers=AUTH).status_code == 200
    assert client.put(f"/api/admin/jobs/{job_id}/live", params={"path": "live/now.json"}, content=b"[1]", headers=AUTH).status_code == 200
    assert client.put(f"/api/admin/jobs/{job_id}/live", params={"path": "live/kit.json"}, content=b"not json", headers=AUTH).status_code == 200
    page = client.get(f"/api/admin/jobs/{job_id}", headers=AUTH).json()
    assert page["now"] is None and page["kit"] == {}


def test_footage_says_where_it_comes_from(store):
    assert admin.described("2018-03-07.17-30-00.17-35-00.school.G421.avi") == {
        "dataset": "MEVA", "camera": "G421", "recorded": "2018-03-07 17:30", "license": "CC BY 4.0"}
    assert admin.described("hallway-snack-table.mp4") is None
    (store / "uploads").mkdir()
    (store / "uploads" / "cam.mp4").write_bytes(b"not really a video")
    assert client.get("/api/admin/footage/cam.mp4/film", headers=AUTH).status_code == 404      # nothing ffmpeg can read
    assert client.get("/api/admin/footage/cam.mp4/film").status_code == 401


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
    assert refused.status_code == 422 and not (store / "runs" / job["run"]).exists()
    assert not list((store / "incoming").iterdir())                   # nothing left behind
    # the job thread outlives a disk that cannot be written
    monkeypatch.setattr(admin, "save_jobs", lambda: (_ for _ in ()).throw(OSError("disk full")))
    assert client.get("/api/admin/jobs", headers=AUTH).status_code == 200
    assert any(thread.name == "engine-jobs" and thread.is_alive() for thread in __import__("threading").enumerate())
