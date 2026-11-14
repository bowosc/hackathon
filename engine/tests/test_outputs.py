import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from conftest import FPS, STEP, person, reach
from relentless import activity, catalog, evaluate, interactions, label, meva, multiply, pipeline, render, serve


def test_window_card_shifts_and_clips_times():
    card = {"main_object": "cup", "labels": [{"action": "a", "object": "cup", "start": 0.5, "end": 2.0},
                                              {"action": "b", "object": "cup", "start": 3.0, "end": 6.5},
                                              {"action": "c", "object": "lid", "start": 7.0, "end": 8.0}]}
    out = pipeline.window_card(card, 2.5, 4.0)
    assert [(l["action"], l["start"], l["end"]) for l in out["labels"]] == [("b", 0.5, 4.0)]
    assert out["main_object"] == "cup" and len(card["labels"]) == 3          # the original is untouched
    assert pipeline.best_window(card, 8.0, 4.0) == 2.5                       # the window with most cup handling
    assert pipeline.best_window(card, 3.0, 4.0) == 0.0


def test_carry_labels_renames_the_object_everywhere():
    card = {"caption": "picks up a paper cup", "main_object": "paper cup", "objects": ["paper cup", "table"],
            "labels": [{"action": "picks up paper cup", "object": "paper cup", "start": 0, "end": 1}], "scene": {}}
    edit = multiply.presets(("paper cup", "apple"))["swap"]
    out = multiply.carry_labels(card, edit)
    assert out["main_object"] == "apple" and out["objects"] == ["apple", "table"]
    assert out["labels"][0] == {"action": "picks up apple", "object": "apple", "start": 0, "end": 1}
    assert out["caption"] == "picks up a apple" or "apple" in out["caption"]
    assert out["scene"]["object_swap"] == "paper cup -> apple"
    assert card["main_object"] == "paper cup"
    assert "with an apple" in edit.prompt and "with a soda can" in multiply.presets(("cup", "soda can"))["swap"].prompt


def test_widen_to_16x9_stays_in_frame():
    x, y, w, h = multiply.widen_to_16x9((1800, 900, 100, 150), 1920, 1080)
    assert x + w <= 1920 and y + h <= 1080 and abs(w / h - 16 / 9) < 0.05


def _label(**kw):
    base = dict(hands_visible=True, interaction=True, caption="c", objects=["cup"], main_object="cup",
                labels=[label.Action(verb="pick_up", action="picks up cup", object="cup", hand="right", start=0, end=1)],
                scene=label.Scene(setting="s", lighting="l", viewpoint="v"), quality=6, swap_candidates=["can"],
                issues="none")
    return label.ClipLabel(**{**base, **kw})


def test_acceptance_rules():
    assert label.accepted(_label()) == (True, "ok")
    assert label.accepted(_label(interaction=False))[0] is False
    assert label.accepted(_label(hands_visible=False))[0] is False
    assert label.accepted(_label(quality=3))[1] == "quality 3 below 4"
    assert label.accepted(_label(labels=[]))[0] is False


def test_tiers_and_swap_score():
    lab = _label().model_dump()
    lab["labels"].append({**lab["labels"][0], "start": 1, "end": 2})
    clip = {"label": lab, "files": {"hands": "h.json"}, "variants": []}
    assert pipeline.tiers(clip) == ["bulk", "labeled+"]
    clip["variants"] = [{"passed": False}, {"passed": True}]
    assert pipeline.tiers(clip) == ["bulk", "labeled+", "studio"]
    assert pipeline.tiers({"label": {**lab, "quality": 4}, "files": {}, "variants": []}) == ["bulk"]
    vague = {**lab, "main_object": "small dark item"}
    assert pipeline.swap_score(lab, 4.0) > pipeline.swap_score(vague, 4.0)
    assert pipeline.swap_score({**lab, "swap_candidates": []}, 4.0) < 0


def test_funnel_rows_follow_the_numbers():
    state = {"_numbers": {"raw_s": 90.0, "motion_s": 60.0, "tracklets": 4, "person_s": 100.0, "active_s": 40.0,
                          "candidates": 9, "candidate_s": 30.0, "refined": 3, "upscaled": 3, "labeled": 3,
                          "accepted": 2, "rejected": 1, "kept": {"motion": 0.66, "zoom": 0.05}}}
    rows = {r["key"]: r for r in pipeline.funnel(state)}
    assert list(rows) == ["raw", "motion", "people", "activity", "zoom", "refined", "accepted"]
    assert rows["raw"]["share"] == 1.0 and rows["motion"]["share"] == 0.66
    assert rows["zoom"]["share"] is None          # planned crops: no footage share claimed
    assert rows["accepted"]["value"] == 2 and "1 rejected" in rows["accepted"]["sub"]
    assert rows["refined"]["sub"] == "best 3 of 9"


def test_run_name():
    from pathlib import Path
    assert pipeline.run_name(Path("2018-03-07.17-30-00.17-35-00.school.G421.avi"), 60, 150) == "G421-0307-1730_60-150s"
    assert pipeline.run_name(Path("my store cam (1).mp4"), 0, 12.4) == "my-store-cam-1-_0-12s"


def test_evaluate_counts_an_event_through_the_funnel():
    t = reach(person(), 40, 60)
    t.id = 0
    sig = activity.signals(t)
    active = {0: activity.active_spans(t, sig)}
    cands = interactions.find(t, sig, active[0], (1920, 1080))
    box = t.box[0]
    gt = meva.GroundTruth(
        activities=[meva.Activity("person_talks_to_person", 0, 300, people=[7]),                 # not a grasp
                    meva.Activity("person_puts_down_object", 5 * STEP, 8 * STEP, people=[7]),     # nothing moving then
                    meva.Activity("person_picks_up_object", 45 * STEP, 55 * STEP, people=[7])],
        geom={7: (np.arange(0, 300), np.tile(box, (300, 1)))})
    clip = {"id": "c000", "track": 0, "start": cands[0].start, "end": cands[0].end, "crop": list(cands[0].crop),
            "status": "accepted", "label": {"labels": [{"verb": "pick_up", "start": 0.5, "end": 3.0}]}}
    common = dict(fps=FPS, first=0, last=300, motion_spans=[(0, 300)], tracklets=[t], active=active,
                  candidates=cands, clips=[clip])
    kept = evaluate.footage(frame_size=(1920, 1080), **common)
    assert kept["motion"] == 1.0 and 0 < kept["accepted"] <= kept["zoom"] < kept["people"] < 1
    result = evaluate.evaluate(gt, kept=kept, **common)
    found = {s["stage"]: s["found"] for s in result["stages"]}
    assert result["events"] == 2
    assert found == {"motion": 2, "people": 2, "activity": 1, "zoom": 1, "refined": 1, "accepted": 1}
    assert result["label_agreement"] == 1.0
    assert [e["reached"] for e in result["per_event"]] == ["people", "accepted"]


def test_blur_only_touches_the_head_circle():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    before = frame.copy()
    render.blur_heads(frame, np.array([[150.0, 100.0, 20.0]]))
    changed = np.argwhere((frame != before).any(axis=2))
    assert len(changed) > 500
    assert np.hypot(changed[:, 1] - 150, changed[:, 0] - 100).max() <= 20 * 1.1 + 2
    inside = frame[90:110, 140:160].astype(int)
    assert inside.std() < before[90:110, 140:160].astype(int).std() / 3      # detail is gone


def test_stills_take_evenly_spaced_frames():
    job = render.Stills(100, 220, (10, 20, 64, 36), n=4)
    wanted = [f for f in range(0, 400) if job.wants(f)]
    assert wanted == [115, 145, 175, 205]
    job.emit(115, np.zeros((200, 300, 3), np.uint8))
    assert job.frames[0].shape == (36, 64, 3)


@pytest.fixture
def server(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    (runs / "demo").mkdir(parents=True)
    (runs / "demo" / "state.json").write_text(json.dumps({"name": "demo", "status": "done", "clips": []}))
    (runs / "demo" / "clip.mp4").write_bytes(bytes(range(256)) * 4)
    (tmp_path / "secret.txt").write_text("no")
    monkeypatch.setattr(serve, "RUNS", runs)
    monkeypatch.setattr(serve, "DATA", tmp_path)
    srv = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_server_serves_runs_with_ranges_and_blocks_escapes(server):
    runs = json.load(urllib.request.urlopen(server + "/api/runs"))
    assert runs["runs"][0]["name"] == "demo" and runs["running"] is None
    req = urllib.request.Request(server + "/runs/demo/clip.mp4", headers={"Range": "bytes=10-19"})
    with urllib.request.urlopen(req) as r:
        assert r.status == 206 and r.read() == bytes(range(10, 20))
        assert r.headers["Content-Range"] == "bytes 10-19/1024"
    with urllib.request.urlopen(urllib.request.Request(server + "/runs/demo/clip.mp4", headers={"Range": "bytes=-4"})) as r:
        assert r.read() == bytes(range(252, 256))
    for bad in ("/runs/../secret.txt", "/runs/demo/../../secret.txt", "/runs/%2e%2e/secret.txt", "/runs/demo/nope.mp4"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(server + bad)
        assert e.value.code in (400, 404)
    body = json.dumps({"video": "../../etc/passwd"}).encode()
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(urllib.request.Request(server + "/api/runs", data=body, method="POST"))
    assert e.value.code == 400


def test_catalog_keeps_the_newest_run_of_the_same_clip(tmp_path, monkeypatch):
    import os
    def item(run, caption, frames=(100, 200)):
        return {"id": f"{run}-c000", "kind": "original", "run": run, "caption": caption, "seconds": 3.0,
                "files": {"video": "clips/c000/upscaled.mp4"},
                "lineage": {"source": "cam.avi", "frames": list(frames), "crop_xywh": [0, 0, 320, 180]}}
    # the older run sorts last by name, so name order would pick the wrong one
    for age, (run, caption) in enumerate([("zz-old-run", "old label"), ("aa-new-run", "new label")]):
        (tmp_path / run).mkdir()
        path = tmp_path / run / "catalog.json"
        path.write_text(json.dumps([item(run, caption)] + ([item(run, "only here", (300, 400))] if age == 0 else [])))
        os.utime(path, (1_000_000 + age, 1_000_000 + age))
    monkeypatch.setattr(catalog, "RUNS", tmp_path)
    items = catalog.dedupe(catalog.collect())
    assert sorted(i["caption"] for i in items) == ["new label", "only here"]
    assert [i["files"]["video"] for i in items if i["caption"] == "new label"] == ["runs/aa-new-run/clips/c000/upscaled.mp4"]
