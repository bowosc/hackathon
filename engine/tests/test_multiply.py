"""Object swaps: which clips may be swapped, with what, and what a variant must hold up to.

The case these rules exist for: a refuse collector gripping the grab handle of the truck, labeled (from foggy
footage) as holding "a crumpled yellow-white packet", swapped for a bag of crisps, and passed by the check.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from relentless import label, multiply, pipeline


def plan(**kw):
    base = dict(object_seen="white paper cup", is_main_object=True, clearly_visible=True, loose_item=True, in_hand=True,
                swappable=True, why="a cup held in one hand",
                replacements=[multiply.Replacement(name="ceramic mug", same_kind=True, fits_scene=3, same_grip=2)])
    return multiply.SwapPlan(**{**base, **kw})


def replacement(name, same_kind=True, fits_scene=3, same_grip=3):
    return multiply.Replacement(name=name, same_kind=same_kind, fits_scene=fits_scene, same_grip=same_grip)


# ---------- is the clip fit for a swap ----------

def test_a_cup_in_a_hand_is_fit():
    assert multiply.fit(plan(), "paper cup") == (True, "")
    assert multiply.good_replacements(plan(), "paper cup") == ["ceramic mug"]


@pytest.mark.parametrize("change, reason", [
    (dict(object_seen="cannot tell"), "cannot be made out"),
    (dict(object_seen="grab handle of the truck", loose_item=False), "grab handle of the truck: not a loose item"),
    (dict(object_seen="card terminal", in_hand=False), "card terminal: the hand does not hold it"),
    (dict(object_seen="small pale thing", clearly_visible=False), "too small, blurred or hidden"),
    (dict(object_seen="napkin", is_main_object=False), "handles napkin, not the labeled paper cup"),
    (dict(swappable=False, why="the cup is only touched"), "the cup is only touched"),
    (dict(replacements=[]), "no replacement of the same kind"),
])
def test_what_makes_a_clip_unfit(change, reason):
    ok, why = multiply.fit(plan(**change), "paper cup")
    assert ok is False and reason in why


def test_the_rules_are_checked_not_the_planners_own_summary():
    """A plan that calls a fixed handle swappable is still a no: the summary is one answer among several."""
    confused = plan(object_seen="yellow grab handle", loose_item=False, swappable=True,
                    replacements=[replacement("bag of crisps")])
    assert multiply.fit(confused, "yellow packet")[0] is False


def test_only_same_kind_replacements_that_fit_the_scene_and_the_grip_are_kept():
    p = plan(replacements=[replacement("bag of crisps", same_kind=False), replacement("soup bowl", fits_scene=1),
                           replacement("bucket", same_grip=1), replacement("a ceramic mug", fits_scene=2, same_grip=3),
                           replacement("takeaway cup"), replacement("Paper Cup"), replacement("takeaway cup")])
    # best first; the article is dropped; the object itself and repeats are not replacements
    assert multiply.good_replacements(p, "paper cup") == ["takeaway cup", "ceramic mug"]
    assert multiply.fit(plan(replacements=[replacement("bag of crisps", same_kind=False)]), "paper cup")[0] is False


# ---------- asking the planner ----------

def test_the_planner_sees_the_window_and_the_card(monkeypatch):
    seen = {}

    def fake_ask(content, schema, **kw):
        seen.update(content=content, schema=schema, **kw)
        return plan(replacements=[replacement("mug", fits_scene=7, same_grip=-2)]), {"usd": 0.001, "model": "claude-haiku-5-5"}

    monkeypatch.setattr(multiply.claude, "ask_full", fake_ask)
    monkeypatch.setattr(multiply, "window_frames", lambda clip, start, seconds, hands=None: [b"jpeg"] * 6)
    monkeypatch.setattr(multiply, "SWAP_MODEL", None)
    card = {"caption": "holds a paper cup", "main_object": "paper cup", "objects": ["paper cup"], "quality": 5,
            "labels": [{"verb": "hold", "action": "holds paper cup", "object": "paper cup", "hand": "right", "start": 0, "end": 4}]}
    out, info = multiply.plan_swap(Path("upscaled.mp4"), card, start=1.0, seconds=4.0, n=2, purpose="swap:run:c001")
    assert seen["schema"] is multiply.SwapPlan and seen["purpose"] == "swap:run:c001" and seen["model"] is None
    images = [block for block in seen["content"] if block["type"] == "image"]
    text = seen["content"][-1]["text"]
    assert len(images) == 6 and '"main_object": "paper cup"' in text and "at most 2" in text
    assert "When in doubt, the clip is not fit" in text and "boxes marked L and R" not in text   # no hand tracks given
    assert (out.replacements[0].fits_scene, out.replacements[0].same_grip) == (3, 0)             # kept on the 0-3 scale


# ---------- the edit and the labels of a variant ----------

def test_the_edit_touches_only_the_object():
    prompt = multiply.presets(("paper cup", "apple"))["swap"].prompt
    assert prompt.startswith("Replace only the paper cup in the person's hand with an apple")
    assert "glove" in prompt and "keep every blurred face blurred" in prompt
    assert "with a soda can" in multiply.presets(("cup", "soda can"))["swap"].prompt
    assert multiply.with_article("the mug") == "the mug" and multiply.with_article("orange") == "an orange"


def test_labels_are_carried_over_by_whole_words():
    card = {"caption": "A hand picks up a paper cup from the cupboard shelf, next to two paper cups.",
            "main_object": "paper cup", "objects": ["paper cup", "cupboard"], "swap_candidates": ["mug"],
            "labels": [{"action": "picks up paper cup", "object": "paper cup", "start": 0, "end": 1}], "scene": {}}
    out = multiply.carry_labels(card, multiply.presets(("paper cup", "apple"))["swap"])
    # the cup in the hand is renamed; the other cups on the shelf are still cups (the edit leaves them alone)
    assert out["caption"] == "A hand picks up an apple from the cupboard shelf, next to two paper cups."
    assert out["objects"] == ["apple", "cupboard"] and out["main_object"] == "apple"
    assert out["labels"][0] == {"action": "picks up apple", "object": "apple", "start": 0, "end": 1}
    assert "swap_candidates" not in out and out["scene"]["object_swap"] == "paper cup -> apple"
    assert card["caption"].startswith("A hand picks up a paper cup")          # the original card is untouched
    # a short name is not found inside longer words
    short = multiply.carry_labels({"caption": "moves a cup past the cupboard and the teacup-holder", "objects": ["cup"]},
                                  multiply.presets(("cup", "glass"))["swap"])
    assert short["caption"] == "moves a glass past the cupboard and the teacup-holder" and short["objects"] == ["glass"]


def test_a_colour_in_front_of_the_old_object_goes_with_it():
    swap = multiply.presets(("paper cup", "white paper coffee cup"))["swap"]
    out = multiply.carry_labels({"caption": "lifts a maroon paper cup, sets the dark red paper cup down by a green basket",
                                 "labels": [{"action": "lifts maroon paper cup", "object": "paper cup"}]}, swap)
    assert out["caption"] == "lifts a white paper coffee cup, sets the white paper coffee cup down by a green basket"
    assert out["labels"][0] == {"action": "lifts white paper coffee cup", "object": "white paper coffee cup"}
    # colours joined by "and", with or without hyphens, are one colour phrase
    packet = multiply.presets(("snack packet", "apple"))["swap"]
    for text, want in [("picks up red and blue snack packet", "picks up apple"),
                       ("holds a pink-and-white snack packet", "holds an apple"),
                       ("a yellow snack packet on a red tray", "an apple on a red tray")]:
        assert multiply.carry_labels({"caption": text}, packet)["caption"] == want


def test_only_the_objects_name_is_renamed_not_the_verb_or_the_start_of_a_sentence():
    """A label whose object is called like a verb ("drink"), and a caption that starts with the object."""
    card = {"caption": "Drink in hand, the person drinks and sets the drink down.", "main_object": "drink", "objects": ["drink"],
            "labels": [{"verb": "drink", "action": "sips from drink", "object": "drink", "hand": "right", "start": 0, "end": 1}]}
    out = multiply.carry_labels(card, multiply.presets(("drink", "water bottle"))["swap"])
    assert out["caption"] == "Water bottle in hand, the person drinks and sets the water bottle down."
    assert out["labels"][0] == {"verb": "drink", "action": "sips from water bottle", "object": "water bottle", "hand": "right",
                                "start": 0, "end": 1}
    # a name with characters that mean something to a pattern, and no name at all
    odd = multiply.carry_labels({"caption": "holds cup (large) and a cup"}, multiply.presets(("cup (large)", "mug"))["swap"])
    assert odd["caption"] == "holds mug and a cup"
    untouched = multiply.carry_labels({"caption": "holds a cup"}, multiply.Edit("swap", "p", {}, ("", "mug")))
    assert untouched["caption"] == "holds a cup"


def test_a_variant_in_the_catalog_carries_its_own_caption():
    variant = {"id": "c000-swap-apple", "passed": True, "seconds": 4.0, "edit": "paper cup -> apple", "verify": {}, "model": "m",
               "prompt": "p", "files": {"video": "v.mp4"},
               "labels": {"caption": "picks up an apple", "main_object": "apple", "objects": ["apple"], "labels": [], "scene": {}}}
    clip = {"id": "c000", "status": "accepted", "seconds": 5.0, "frames": [0, 150], "crop": [0, 0, 320, 180], "track": 1,
            "tiers": ["bulk", "studio"], "size": [1280, 720], "native_size": [320, 180],
            "files": {k: f"{k}.x" for k in ("upscaled", "raw", "context", "compare", "hands", "card")},
            "label": {"caption": "picks up a paper cup", "objects": ["paper cup"], "main_object": "paper cup", "labels": [],
                      "quality": 5, "scene": {}},
            "variants": [variant, {**variant, "id": "c000-swap-bad", "passed": False}]}
    run = SimpleNamespace(state={"name": "run", "clips": [clip], "source": {"fps": 30.0, "file": "cam.avi"},
                                 "engine": {"labeler": "claude-haiku-5-5"}})
    original, swapped = pipeline.build_catalog(run)                          # the variant that did not pass is not sold
    assert original["caption"] == "picks up a paper cup" and original["main_object"] == "paper cup"
    assert swapped["caption"] == "picks up an apple" and swapped["main_object"] == "apple" and swapped["tiers"] == ["studio"]
    assert swapped["parent"] == original["id"]


# ---------- the check ----------

def verdict(score=7, **flags):
    base = dict(edit_applied=True, motion_preserved=True, hands_intact=True, faces_blurred=True, only_edit_changed=True,
                plausible=True, actions_seen=["holds apple"], score=score, notes="ok")
    return multiply.Verdict(**{**base, **flags})


def test_what_a_variant_must_hold_up_to():
    assert multiply.passed(verdict())
    for flag in ("edit_applied", "motion_preserved", "hands_intact", "faces_blurred", "only_edit_changed", "plausible"):
        assert not multiply.passed(verdict(**{flag: False})), flag
    assert not multiply.passed(verdict(score=5))


def test_the_checker_has_to_answer_every_question():
    """A question with a default is one the checker may skip, and a skipped question would count as a yes."""
    required = set(multiply.Verdict.model_json_schema()["required"])
    assert {"edit_applied", "motion_preserved", "hands_intact", "faces_blurred", "only_edit_changed", "plausible",
            "score"} <= required
    with pytest.raises(ValueError):
        multiply.Verdict(edit_applied=True, motion_preserved=True, hands_intact=True, faces_blurred=True,
                         actions_seen=[], score=8, notes="ok")


def paths(before, after):
    hand = lambda same: {"n": 40, "found": 1.0, "same": same}      # noqa: E731
    return {"original": {"left": hand(before), "right": None}, "variant": {"left": hand(after) if after is not None else None,
                                                                             "right": None}}


def test_hands_that_moved_fail_the_variant_whatever_the_checker_says():
    assert multiply.hands_kept(None) and multiply.hands_kept({})                 # nothing measured: the checker decides
    assert multiply.hands_kept(paths(1.0, 0.95)) and multiply.hands_kept(paths(0.9, 0.7))
    assert not multiply.hands_kept(paths(1.0, 0.6))                              # redrawn somewhere else
    assert not multiply.hands_kept(paths(0.9, None))                             # gone
    assert multiply.hands_kept(paths(0.3, 0.0))                                  # the measurement does not work on this crop
    assert multiply.passed(verdict(), paths(1.0, 0.98)) and not multiply.passed(verdict(), paths(1.0, 0.4))


def test_the_measurement_is_taken_on_the_original_and_on_every_variant(tmp_path, monkeypatch):
    out = tmp_path / "variants"
    out.mkdir()
    (out / "original.mp4").write_bytes(b"original")
    (out / "swap-apple-gemini.mp4").write_bytes(b"variant")
    monkeypatch.setattr(multiply, "verify", lambda *a, **k: (verdict(score=8), "claude-haiku-5-5"))
    monkeypatch.setattr(multiply.media, "probe", lambda p: {"seconds": 4.0, "width": 1280, "height": 720, "fps": 30.0})
    monkeypatch.setattr(multiply.media, "side_by_side", lambda paths_, dst, **k: dst)
    monkeypatch.setattr(multiply.ledger, "spent", lambda service: 0.0)
    edit = multiply.presets(("cup", "apple"))["swap"]
    edit.name = "swap-apple"
    measured = []

    def hand_paths(video):
        measured.append(video.name)
        same = 1.0 if video.name == "original.mp4" else 0.2
        return {"left": {"n": 40, "found": 1.0, "same": same}, "right": None}

    rec = multiply.multiply(out / "upscaled.mp4", {"labels": []}, [edit], "gemini", start=0.0, seconds=4.0, crop=None,
                            out_root=tmp_path, verify_variants=True, dry_run=False, force=False, out_dir=out,
                            hand_paths=hand_paths)["variants"][0]
    assert measured == ["original.mp4", "swap-apple-gemini.mp4"]
    assert rec["passed"] is False and rec["verify"]["score"] == 8                # the checker liked it; the hands moved
    assert rec["hand_paths"]["variant"]["left"]["same"] == 0.2 and rec["check_version"] == multiply.SWAP_VERSION


def test_a_cached_variant_gets_its_labels_made_again(tmp_path, monkeypatch):
    """The video and its check are kept; the labels are derived from the card, and are derived again."""
    out = tmp_path / "variants"
    out.mkdir()
    (out / "original.mp4").write_bytes(b"original")
    (out / "swap-apple-gemini.mp4").write_bytes(b"variant")
    (out / "swap-apple-gemini.json").write_text(json.dumps({
        "variant": "swap-apple", "prompt": "p", "file": "swap-apple-gemini.mp4", "cost_usd": 0.11, "passed": True,
        "labels": {"caption": "lifts a maroon apple"}, "verify": verdict(score=8).model_dump(),
        "verify_model": multiply.CLAUDE_MODEL, "check_version": multiply.SWAP_VERSION}))
    monkeypatch.setattr(multiply, "verify", lambda *a, **k: pytest.fail("checked by this model and this check already"))
    monkeypatch.setattr(multiply, "run_edit", lambda *a, **k: pytest.fail("the cached video must not be made again"))
    monkeypatch.setattr(multiply.media, "probe", lambda p: {"seconds": 4.0, "width": 1280, "height": 720, "fps": 30.0})
    monkeypatch.setattr(multiply.media, "side_by_side", lambda paths_, dst, **k: dst)
    monkeypatch.setattr(multiply.ledger, "spent", lambda service: 0.0)
    edit = multiply.presets(("cup", "apple"))["swap"]
    edit.name = "swap-apple"
    rec = multiply.multiply(out / "upscaled.mp4", {"caption": "lifts a maroon cup", "labels": []}, [edit], "gemini", start=0.0,
                            seconds=4.0, crop=None, out_root=tmp_path, verify_variants=True, dry_run=False, force=False,
                            out_dir=out)["variants"][0]
    assert rec["cached"] and rec["passed"] and rec["labels"]["caption"] == "lifts an apple"


# ---------- which clips are considered at all ----------

def labeled(quality=5, main="paper cup", verb="hold", swap=("mug",), seconds=4.0):
    return {"quality": quality, "main_object": main, "swap_candidates": list(swap),
            "labels": [{"verb": verb, "object": main, "start": 0.0, "end": seconds}]}


def test_only_a_named_object_that_is_in_the_hand_is_a_candidate():
    good = labeled()
    assert pipeline.held_share(good, 4.0) == 1.0 and pipeline.swap_score(good, 4.0) == 8.0
    assert pipeline.swap_score(labeled(main="small dark item"), 4.0) < 0          # not named
    assert pipeline.swap_score(labeled(swap=()), 4.0) < 0                         # the labeler saw nothing to swap it for
    assert pipeline.swap_score(labeled(verb="press"), 4.0) < 0                    # pressed, not held: a screen, a button
    assert pipeline.swap_score(labeled(verb="reach"), 4.0) < 0
    assert pipeline.swap_score(labeled(main=None), 4.0) < 0
    briefly = labeled()
    briefly["labels"][0]["end"] = 0.5
    assert pipeline.swap_score(briefly, 4.0) < 0                                  # in the hand for an eighth of the clip


def test_the_checks_that_cost_nothing():
    clip = {"label": labeled(), "seconds": 4.0, "hands": ["right"],
            "hand_tracking": {"model": "RTMW-x 384x288", "located": {"left": 0.1, "right": 0.9}}}
    assert pipeline.swap_candidate(clip, 5) == (True, "")
    assert pipeline.swap_candidate({**clip, "label": labeled(quality=4)}, 5) == (False, "label quality 4 is below 5")
    assert "no clearly named object" in pipeline.swap_candidate({**clip, "label": labeled(main="small item")}, 5)[1]
    hidden = {**clip, "hand_tracking": {"model": "RTMW-x 384x288", "located": {"left": 0.9, "right": 0.3}}}
    assert "working hand is not located" in pipeline.swap_candidate(hidden, 5)[1]
    # a run on a machine without the whole-body model has nothing to say about the hands: not a reason to refuse
    assert pipeline.swap_candidate({**clip, "hand_tracking": {"model": None}}, 5)[0] is True


# ---------- the whole multiply step, with everything paid for faked ----------

class FakeUpscaler:
    def __init__(self, *a, **k):
        pass

    def video(self, src, dst, progress=None):
        dst.write_bytes(b"upscaled")
        return {"in_size": [320, 180], "out_size": [1280, 720], "seconds": 1.0}


@pytest.fixture
def refined(tmp_path, monkeypatch):
    """A run with three accepted clips ready to be refined: a cup in a hand (fit), the truck's grab handle
    labeled as a packet (what the planner must stop), and a clip the cheap checks already rule out."""
    from relentless import upscale

    labels = {
        "c000": dict(caption="A gloved hand holds a crumpled yellow-white packet.", main_object="crumpled yellow-white packet",
                     verb="hold", quality=5, swap=["snack bag"]),
        "c001": dict(caption="A hand lifts a paper cup.", main_object="paper cup", verb="pick_up", quality=5, swap=["mug"]),
        "c002": dict(caption="A hand presses a card terminal.", main_object="card terminal", verb="press", quality=5, swap=[]),
    }
    plans = {"c000": plan(object_seen="yellow grab handle of the truck, gripped by a glove", loose_item=False, swappable=False,
                          is_main_object=False, why="a fixed handle, not a packet", replacements=[]),
             "c001": plan(replacements=[replacement("ceramic mug"), replacement("takeaway cup", fits_scene=2)])}
    calls = {"planned": [], "edited": []}

    def fake_label(clip, context=None, **kw):
        spec = labels[clip.parent.name]
        lab = label.ClipLabel(hands_visible=True, interaction=True, caption=spec["caption"], objects=[spec["main_object"]],
                              main_object=spec["main_object"], scene=label.Scene(setting="s", lighting="l", viewpoint="v"),
                              labels=[label.Action(verb=spec["verb"], action=spec["caption"], object=spec["main_object"],
                                                   hand="right", start=0.0, end=4.0)],
                              quality=spec["quality"], swap_candidates=spec["swap"], issues="none")
        return lab, {"usd": 0.002, "model": "claude-haiku-5-5", "frames": 8}

    def fake_plan(clip, card, **kw):
        calls["planned"].append(clip.parent.name)
        return plans[clip.parent.name], {"usd": 0.001, "model": "claude-haiku-5-5"}

    def fake_multiply(clip, card, edits, model, **kw):
        calls["edited"].append((clip.parent.name, [e.swap for e in edits], [e.name for e in edits]))
        out = kw["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "original.mp4").write_bytes(b"o")
        records = []
        for e in edits:
            (out / f"{e.name}-{model}.mp4").write_bytes(b"v")
            records.append({"variant": e.name, "model": "fal", "prompt": e.prompt, "file": f"{e.name}-{model}.mp4",
                            "cost_usd": 0.11, "labels": multiply.carry_labels(card, e), "passed": True,
                            "verify": verdict(score=8).model_dump(), "verify_model": "claude-haiku-5-5"})
        return {"variants": records}

    monkeypatch.setattr(upscale, "Upscaler", FakeUpscaler)
    monkeypatch.setattr(upscale, "compare", lambda raw, up, dst: dst.write_bytes(b"c"))
    monkeypatch.setattr(pipeline.media, "poster", lambda src, dst: dst.write_bytes(b"p"))
    monkeypatch.setattr(pipeline.labeler, "label_clip", fake_label)
    monkeypatch.setattr(multiply, "plan_swap", fake_plan)
    monkeypatch.setattr(multiply, "multiply", fake_multiply)
    monkeypatch.setattr(multiply, "estimate", lambda secs, model, n: 0.11)
    monkeypatch.setattr(pipeline.ledger, "spent", lambda service: 0.0)
    monkeypatch.setattr(pipeline, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(pipeline.Run, "say", lambda self, msg: self.state["log"].append({"t": 0, "msg": msg}))

    def run_it(multiply_n=2):
        out = tmp_path / f"run{multiply_n}"
        clips, plan_ = [], []
        for n, cid in enumerate(labels):
            cdir = out / "clips" / cid
            cdir.mkdir(parents=True)
            (cdir / "raw.mp4").write_bytes(b"raw")
            (cdir / "hands.json").write_text(json.dumps({"fps": 30.0, "frames": 120, "busy": ["right"]}))
            clips.append({"id": cid, "track": n, "seconds": 4.0, "frames": [n * 200, n * 200 + 120], "crop": [0, 0, 320, 180],
                          "context_crop": [0, 0, 640, 360], "hands": ["right"], "status": "cut", "label": None, "variants": [],
                          "hand_tracking": {"model": "RTMW-x 384x288", "located": {"left": 0.0, "right": 1.0}},
                          "files": {"hands": f"clips/{cid}/hands.json", "raw": f"clips/{cid}/raw.mp4",
                                    "context": f"clips/{cid}/context.mp4"}})
            plan_.append((cid, SimpleNamespace(hands=["right"]), n))
        state = {"name": out.name, "clips": clips, "log": [], "_numbers": {}, "funnel": [],
                 "source": {"file": "truck.mp4", "fps": 30.0, "size": [1920, 1080]},
                 "engine": {"upscaler": "x4", "labeler": "claude-haiku-5-5"},
                 "stages": [{"key": k, "title": t, "status": "pending", "progress": 0.0} for k, t in pipeline.STAGES]}
        run = pipeline.Run(out, state)
        run.begin("upscale")
        opts = pipeline.Options(multiply=multiply_n, cache=False, label_workers=1)
        pipeline._refine(run, opts, plan_, 30.0, lambda: None)
        return run, calls

    return run_it


def test_the_grab_handle_is_never_swapped_and_the_cup_is(refined):
    run, calls = refined(multiply_n=2)
    clips = {c["id"]: c for c in run.state["clips"]}
    # the planner was asked about the two clips that pass the cheap checks, and only the cup went to the edit model
    assert sorted(calls["planned"]) == ["c000", "c001"]
    assert calls["edited"] == [("c001", [("paper cup", "ceramic mug")], ["swap-ceramic-mug"])]
    assert clips["c000"]["multiply"] == "skipped" and clips["c000"]["variants"] == []
    assert "not a loose item" in clips["c000"]["multiply_note"] and "grab handle" in clips["c000"]["multiply_note"]
    assert clips["c001"]["multiply"] == "done" and clips["c001"]["swap"]["to"] == ["ceramic mug"]
    assert [v["edit"] for v in clips["c001"]["variants"]] == ["paper cup -> ceramic mug"]
    assert "multiply" not in clips["c002"] and "no clearly named object" in clips["c002"]["multiply_note"]
    stage = next(s for s in run.state["stages"] if s["key"] == "multiply")
    assert stage["status"] == "done" and stage["detail"] == "1 of 1 variants passed the check"
    assert any("not swapped" in line["msg"] and "c000" in line["msg"] for line in run.state["log"])


def test_when_no_clip_is_fit_nothing_is_swapped_and_the_stage_says_so(refined, monkeypatch):
    monkeypatch.setattr(multiply, "plan_swap", lambda clip, card, **kw: (plan(object_seen="cannot tell", swappable=False,
                                                                              replacements=[]), {"usd": 0.001}))
    run, calls = refined(multiply_n=2)
    assert calls["edited"] == [] and all(not c["variants"] for c in run.state["clips"])
    stage = next(s for s in run.state["stages"] if s["key"] == "multiply")
    assert stage["status"] == "done" and stage["detail"] == "no clip was fit for an object swap"


def test_a_planner_that_cannot_be_asked_is_a_no(refined, monkeypatch):
    def down(clip, card, **kw):
        raise RuntimeError("overloaded")

    monkeypatch.setattr(multiply, "plan_swap", down)
    run, calls = refined(multiply_n=1)
    assert calls["edited"] == []
    notes = [c.get("multiply_note", "") for c in run.state["clips"]]
    assert any("could not be planned" in note and "overloaded" in note for note in notes)


def test_without_multiply_nobody_is_asked(refined):
    run, calls = refined(multiply_n=0)
    assert calls == {"planned": [], "edited": []}
    assert next(s for s in run.state["stages"] if s["key"] == "multiply")["status"] == "skipped"


def test_the_planner_is_asked_at_most_four_times_per_variant_wanted(refined, monkeypatch):
    """Whoever asks: a label thread as a clip comes in, or the pass over what is left at the end."""
    asked = []

    def never(clip, card, **kw):
        asked.append(clip.parent.name)
        return plan(object_seen="cannot tell", swappable=False, replacements=[]), {"usd": 0.001}

    monkeypatch.setattr(multiply, "plan_swap", never)
    monkeypatch.setattr(pipeline, "swap_candidate", lambda clip, min_quality: (True, ""))      # every clip is a candidate
    monkeypatch.setattr(pipeline, "swap_score", lambda label, seconds: 9.0)                    # ... and a plainly good one
    refined(multiply_n=1)
    assert len(asked) == 3 and len(set(asked)) == 3            # three clips, each asked once, under the bound of four
    asked.clear()
    monkeypatch.setattr(pipeline.Options, "multiply", 0, raising=False)
    refined(multiply_n=0)
    assert asked == []
