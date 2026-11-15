"""The second look: the whole-body model's answers turned into hand tracks, and what the pipeline does with them.

No model runs here. The model's answers are made up (a person with both hands where the test puts them), so
what is tested is everything around the model: the geometry, the confidence scale, what counts as a located
hand, the cleaning over time, and how a clip's busy hand, crop and hands.json follow from a track.
"""

import json

import numpy as np
import pytest

from conftest import FPS, STEP, person, reach
from relentless import hands, interactions, label, live, pipeline, render, wholebody
from relentless.people import L_ELBOW, L_SHOULDER, L_WRIST, R_ELBOW, R_SHOULDER, R_WRIST

GOOD, BAD = 4.5, 1.2        # model scores of a keypoint it is sure of, and of one it is guessing


def answers(t, hand_score=(GOOD, GOOD), body_score=GOOD, size=0.12):
    """What the whole-body model would say about a tracklet: (n, 133, 3). The body joints are the tracklet's;
    each hand is 21 keypoints in a small cluster just past its wrist, along the forearm."""
    n = len(t.frames)
    height = float(t.box[0, 3] - t.box[0, 1])
    kp = np.zeros((n, 133, 3), np.float32)
    kp[:, :17, :2] = t.kp[:, :, :2]
    kp[:, :, 2] = body_score
    rng = np.random.default_rng(0)
    spread = (rng.random((21, 2)) - 0.5) * size * height      # the same hand shape in every sample
    for s, (part, elbow, wrist) in enumerate(((wholebody.LEFT_HAND, L_ELBOW, L_WRIST), (wholebody.RIGHT_HAND, R_ELBOW, R_WRIST))):
        fore = t.kp[:, wrist, :2] - t.kp[:, elbow, :2]
        centre = t.kp[:, wrist, :2] + 0.35 * fore
        kp[:, part, :2] = centre[:, None, :] + spread[None]
        kp[:, part, 2] = hand_score[s]
    return kp


def track_of(t, **kw):
    return hands.assemble(t.frames, t.box, answers(t, **kw), t.step, t.fps)


# ---------- the model's geometry and scale ----------

def test_confidence_is_half_at_the_measured_score_and_keeps_order():
    assert wholebody.confidence(wholebody.HAND_HALF, wholebody.HAND_HALF) == pytest.approx(0.5)
    scores = np.array([0.5, 1.5, 2.3, 2.8, 3.5, 4.5, 6.0])
    conf = wholebody.confidence(scores, wholebody.HAND_HALF)
    assert (np.diff(conf) > 0).all() and conf[0] < 0.05 and conf[-1] > 0.99
    # a hand keypoint the model is guessing at is far below the bar, one it is sure of far above
    assert wholebody.confidence(BAD, wholebody.HAND_HALF) < 0.1 < 0.9 < wholebody.confidence(GOOD, wholebody.HAND_HALF)


def test_a_point_survives_the_trip_into_the_model_and_back():
    """Cut a person out of a frame, find a bright pixel in the cut-out, decode it: it is where it was."""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[498:503, 898:903] = 255                               # a mark at (900, 500)
    size = (288, 384)
    centre, scale = wholebody.crop_box(np.array([800, 300, 1000, 700], np.float32), size)
    assert scale[0] / scale[1] == pytest.approx(288 / 384) and scale[1] == pytest.approx(400 * wholebody.PADDING)
    patch = wholebody.cut(frame, centre, scale, size)
    assert patch.shape == (384, 288, 3)
    ys, xs = np.nonzero(patch[:, :, 0] > 128)
    # the model answers in half pixels: build its answer for a keypoint at the mark
    simcc_x = np.zeros((1, 1, 576), np.float32)
    simcc_y = np.zeros((1, 1, 768), np.float32)
    simcc_x[0, 0, int(round(xs.mean() * 2))], simcc_y[0, 0, int(round(ys.mean() * 2))] = 5.0, 3.0
    out = wholebody.decode(simcc_x, simcc_y, centre[None], scale[None], size)
    assert out.shape == (1, 1, 3)
    assert out[0, 0, 0] == pytest.approx(900, abs=1.5) and out[0, 0, 1] == pytest.approx(500, abs=1.5)
    assert out[0, 0, 2] == pytest.approx(4.0)                    # the score: the mean of the two responses


def test_hand_box_holds_the_hand_and_cannot_grow_past_a_hand(standing):
    kp = answers(standing)[0]
    boxes, conf = wholebody.hand_boxes(kp, 0.34 * 400)
    for s, part in enumerate(wholebody.HANDS):
        cx, cy, w, h = boxes[s]
        pts = kp[part, :2]
        assert (np.abs(pts[:, 0] - cx) <= w / 2 + 1).all() and (np.abs(pts[:, 1] - cy) <= h / 2 + 1).all()
        assert conf[s] > 0.9
    # one keypoint flung across the frame, and unsure: the box does not follow it
    kp[wholebody.LEFT_HAND.start + 3] = (50.0, 50.0, 0.3)
    wild, _ = wholebody.hand_boxes(kp, 0.34 * 400)
    assert np.allclose(wild[0, :2], boxes[0, :2], atol=12) and wild[0, 2] <= 1.1 * 0.34 * 400
    # scale given per sample, for a whole track at once
    many, many_conf = wholebody.hand_boxes(answers(standing), np.full(len(standing.frames), 0.34 * 400, np.float32))
    assert many.shape == (len(standing.frames), 2, 4) and many_conf.shape == (len(standing.frames), 2)


def test_the_model_name_follows_the_machine(monkeypatch):
    monkeypatch.setattr(wholebody, "accelerated", lambda: True)
    monkeypatch.setattr(wholebody, "setting", lambda name, default: default)
    assert wholebody.model_name() == "x" and wholebody.label() == "RTMW-x 384x288"
    monkeypatch.setattr(wholebody, "accelerated", lambda: False)
    assert wholebody.model_name() == "m" and wholebody.label() == "RTMW-m 256x192"
    monkeypatch.setattr(wholebody, "setting", lambda name, default: "x" if name == "WHOLEBODY" else default)
    assert wholebody.model_name() == "x"                         # the setting wins over the machine
    monkeypatch.setattr(wholebody, "setting", lambda name, default: "off" if name == "WHOLEBODY" else default)
    assert wholebody.available() is False


class FakeSession:
    """Stands in for the ONNX session: remembers the batches it was given."""

    def __init__(self):
        self.batches = []

    def run(self, _outputs, feed):
        (tensor,) = feed.values()
        self.batches.append(tensor.shape)
        n = len(tensor)
        return np.ones((n, 133, 576), np.float32), np.ones((n, 133, 768), np.float32)


def model_with(session, provider, batch=4):
    model = wholebody.WholeBody.__new__(wholebody.WholeBody)
    model.name, model.size, model.batch, model.session, model.provider, model._input = "x", (288, 384), batch, session, provider, "input"
    return model


def test_a_gpu_always_gets_full_batches_and_a_cpu_only_what_there_is():
    frame = np.zeros((480, 640, 3), np.uint8)
    items = [(frame, np.array([100, 100, 200, 400], np.float32))] * 6
    gpu, cpu = FakeSession(), FakeSession()
    out = model_with(gpu, "CoreML").many(items)
    assert out.shape == (6, 133, 3)
    assert gpu.batches == [(4, 3, 384, 288), (4, 3, 384, 288)]   # the last two are padded: no new shape to prepare for
    model_with(cpu, "CPU").many(items)
    assert cpu.batches == [(4, 3, 384, 288), (2, 3, 384, 288)]


# ---------- a track ----------

def test_a_clear_person_has_both_hands_located(standing):
    track = track_of(standing)
    assert track.readable.all() and track.seen.all()
    assert track.kp.shape == (len(standing.frames), 17, 3) and (track.kp[:, 5:11, 2] > 0.9).all()
    # the hand is past the wrist, on the forearm's line, not on the wrist itself
    assert (track.hand[:, 0, 1] > standing.kp[:, L_WRIST, 1]).all()


def test_an_unreadable_person_has_no_hands_whatever_the_hand_scores_say(standing):
    """Fog, glare, a person half out of frame: the arms are guesses, so a 'hand' at the end of them is one too."""
    track = track_of(standing, body_score=BAD)
    assert not track.readable.any() and not track.seen.any()
    found = hands.for_clip(track, 0, int(standing.frames[-1]), ["left", "right"], (1920, 1080))
    assert found.hands == [] and found.crop is None and found.readable == 0
    assert "no hand could be located" in found.notes[0]


def test_a_hidden_arm_does_not_cost_the_hand_in_plain_view(standing):
    """Someone at the edge of the frame, or with one arm behind the body: the arm that is seen is read on
    its own. (Judged over both arms, the visible hand of such a person lost its box.)"""
    kp = answers(standing, hand_score=(BAD, GOOD))
    kp[:, [L_ELBOW, L_WRIST], 2] = BAD                            # the left arm is hidden from the elbow down
    track = hands.assemble(standing.frames, standing.box, kp, STEP, FPS)
    assert track.readable.all() and track.seen[:, 1].all() and not track.seen[:, 0].any()
    found = hands.for_clip(track, 0, 60 * STEP, ["right"], (1920, 1080))
    assert found.hands == ["right"] and found.located == {"left": 0.0, "right": 1.0}
    joints, located, clear = hands.glance(kp[0], standing.box[0])
    assert clear and set(located) == {"r"}


def test_a_hand_the_model_guesses_at_is_not_located(standing):
    track = track_of(standing, hand_score=(GOOD, BAD))
    assert track.seen[:, 0].all() and not track.seen[:, 1].any()


def test_a_hand_must_hang_on_its_wrist_and_stay_together(standing):
    kp = answers(standing)
    kp[:, wholebody.RIGHT_HAND, 0] += 300                         # the right "hand" is half a body away from its wrist
    spread = answers(standing)
    spread[:, wholebody.LEFT_HAND.start:wholebody.LEFT_HAND.start + 10, 1] -= 250   # the left hand's keypoints in two places
    off = hands.assemble(standing.frames, standing.box, kp, STEP, FPS)
    apart = hands.assemble(standing.frames, standing.box, spread, STEP, FPS)
    assert off.seen[:, 0].all() and not off.seen[:, 1].any()
    assert not apart.seen[:, 0].any() and apart.seen[:, 1].all()


def test_slips_are_removed_holes_bridged_and_blips_dropped(standing):
    kp = answers(standing)
    kp[30, wholebody.LEFT_HAND, 0] += 160                         # one sample far from its neighbours (but near enough the wrist)
    kp[50:52, wholebody.LEFT_HAND, 2] = BAD                       # a 0.2 s hole
    kp[:, wholebody.RIGHT_HAND, 2] = BAD
    kp[70:72, wholebody.RIGHT_HAND, 2] = GOOD                     # a lone 0.2 s sighting of the right hand
    track = hands.assemble(standing.frames, standing.box, kp, STEP, FPS)
    left = track.hand[:, 0, 0]
    assert abs(left[30] - left[29]) < 10                          # the slip is back in line
    assert track.seen[:, 0].all()                                 # the hole is bridged
    assert not track.seen[:, 1].any()                             # the blip proves nothing
    assert np.isfinite(track.hand).all()


def test_a_long_hole_stays_a_hole(standing):
    kp = answers(standing)
    kp[40:60, wholebody.LEFT_HAND, 2] = BAD                       # 2 s hidden
    track = hands.assemble(standing.frames, standing.box, kp, STEP, FPS)
    assert not track.seen[42:58, 0].any() and track.seen[:38, 0].all() and track.seen[62:, 0].all()


def test_between_samples_a_track_is_interpolated_and_unreadable_people_carry_no_joints(standing):
    track = track_of(reach(person(), 40, 60))
    f = int(track.frames[45])
    kp, hand, seen, readable = track.on(np.array([f, f + 1, f + 2, f + 3]))
    assert kp.shape == (4, 17, 3) and seen.all() and readable.all()
    assert hand[1, 1, 0] == pytest.approx((2 * track.hand[45, 1, 0] + track.hand[46, 1, 0]) / 3, abs=0.5)
    rows = hands.person_rows(track, track.frames[44:47])
    joints, located, clear = rows[f]
    assert clear and set(located) == {"l", "r"} and located["r"][2] > located["r"][0]
    blind = hands.person_rows(track_of(standing, body_score=BAD), standing.frames[:3])
    joints, located, clear = blind[int(standing.frames[0])]
    assert not clear and located == {} and (joints[:, 2] == 0).all()


def test_on_a_samples_own_frame_only_that_sample_counts(standing):
    """A hand located up to and including a sample is located at that sample, whatever the next one says."""
    kp = answers(standing)
    kp[60:, wholebody.RIGHT_HAND, 2] = BAD                        # the right hand is lost from sample 60 on
    track = hands.assemble(standing.frames, standing.box, kp, STEP, FPS)
    assert track.seen[59, 1] and not track.seen[60, 1]
    _kp, _hand, seen, readable = track.on(standing.frames[57:62])
    assert seen[:, 1].tolist() == [True, True, True, False, False] and readable.all()
    between = track.on(np.array([standing.frames[59] + 1]))[2]    # between a located and a lost sample: not located
    assert not between[0, 1]


def test_one_frame_alone(standing):
    kp, located, clear = hands.glance(answers(standing)[0], standing.box[0])
    assert clear and set(located) == {"l", "r"} and kp.shape == (17, 3)
    kp, located, clear = hands.glance(answers(standing, body_score=BAD)[0], standing.box[0])
    assert not clear and located == {} and (kp[:, 2] == 0).all()


# ---------- what a clip takes from a track ----------

def test_the_hand_that_moves_is_the_busy_one_and_the_crop_holds_its_path(reaching):
    track = track_of(reaching)
    start, end = 35 * STEP, 65 * STEP
    found = hands.for_clip(track, start, end, ["left"], (1920, 1080))      # stage 2 had flagged the wrong hand
    assert found.hands == ["right"] and found.located == {"left": 1.0, "right": 1.0} and found.readable == 1.0
    x, y, w, h = found.crop
    assert w % 2 == 0 and h % 2 == 0 and abs(w / h - 16 / 9) < 0.03
    a, b = track.at(start), track.at(end - 1) + 1
    path = track.hand[a:b, 1, :2]
    assert (path[:, 0] >= x).all() and (path[:, 0] <= x + w).all() and (path[:, 1] >= y).all() and (path[:, 1] <= y + h).all()


def test_a_still_but_located_hand_keeps_what_motion_said(standing):
    """Nothing moves on the careful keypoints (stage 2's wrist was jitter). The flagged hand is in view, so the
    clip stays, with that hand: the labeler decides whether anything is handled."""
    found = hands.for_clip(track_of(standing), 0, 60 * STEP, ["left"], (1920, 1080))
    assert found.hands == ["left"] and found.crop is not None
    assert "kept the hand the motion filter flagged" in found.notes[0]


def test_a_person_readable_only_on_and_off_is_not_a_hand_clip(standing):
    """The worker next to the lens on the night footage: the model's arm crosses the bar every other moment.
    A hand at the end of it, even one that 'moves', is a guess."""
    kp = answers(reach(person(), 40, 60))
    kp[np.arange(len(kp)) % 20 < 11, :17, 2] = BAD                 # readable 45% of the time, in stretches
    track = hands.assemble(standing.frames, standing.box, kp, STEP, FPS)
    assert 0.3 < track.readable.mean() < 0.6
    found = hands.for_clip(track, 0, 99 * STEP, ["right"], (1920, 1080))
    assert found.hands == [] and found.crop is None and "cannot be read for most" in found.notes[0]


def test_when_the_flagged_hand_is_hidden_a_hand_in_view_is_taken(standing):
    track = track_of(standing, hand_score=(GOOD, BAD))             # stage 2 flagged the right hand; only the left is seen
    found = hands.for_clip(track, 0, 60 * STEP, ["right"], (1920, 1080))
    assert found.hands == ["left"] and found.located == {"left": 1.0, "right": 0.0}


def test_hands_json_has_a_box_for_every_frame_in_crop_coordinates(reaching):
    track = track_of(reaching)
    start, end = 35 * STEP, 65 * STEP
    found = hands.for_clip(track, start, end, ["right"], (1920, 1080))
    out = hands.tracks_json(track, start, end, found.crop, found.hands, "RTMW-x 384x288")
    assert out["frames"] == end - start and out["fps"] == FPS and out["busy"] == ["right"]
    assert "RTMW-x" in out["source"] and len(out["box_size"]) == 2
    for side in ("left", "right"):
        t = out[side]
        assert {len(t[k]) for k in ("x", "y", "w", "h", "conf", "seen")} == {end - start}
        assert all(t["seen"]) and all(0 < v < 1 for v in t["w"]) and all(c > 0.9 for c in t["conf"])
    assert all(0 <= v <= 1 for v in out["right"]["x"]) and all(0 <= v <= 1 for v in out["right"]["y"])
    x, y, w, h = found.crop      # the middle of the reach, back in source pixels, is where the track has the hand
    mid = (end - start) // 2
    assert out["right"]["x"][mid] * w + x == pytest.approx(np.interp(start + mid, track.frames, track.hand[:, 1, 0]), abs=1)
    json.dumps(out)              # plain numbers and booleans all the way down


def test_a_hand_outside_the_crop_is_not_in_the_clip(standing):
    """The crop is around the busy hand. The other hand can be located and still be nowhere in this clip:
    no box on the page, no mark on the labeler's frames, nothing to compare a variant against."""
    track = track_of(standing)
    left_x = float(track.hand[0, 0, 0])
    crop = (int(left_x + 40), 300, 480, 270)                      # starts to the right of the left hand
    out = hands.tracks_json(track, 0, 90, crop, ["right"], "RTMW-x 384x288")
    assert not any(out["left"]["seen"]) and all(out["right"]["seen"]) and out["left"]["x"][0] < 0
    frame = np.zeros((270, 480, 3), np.uint8)
    assert label.mark_hands(frame, out, 10) == 1                  # the right hand; no letter for the left one
    # a file from elsewhere that says "seen" for a hand outside the frame gets no letter either
    out["left"]["seen"] = [True] * 90
    frame[:] = 0
    assert label.mark_hands(frame, out, 10) == 1


def test_where_a_hand_is_not_located_hands_json_says_so(standing):
    track = track_of(standing, hand_score=(GOOD, BAD))
    out = hands.tracks_json(track, 0, 90, (600, 300, 480, 270), ["left"], "RTMW-x 384x288")
    assert all(out["left"]["seen"]) and not any(out["right"]["seen"])
    assert all(np.isfinite(out["right"]["x"]))                   # numbers stay numbers; `seen` is what a reader goes by


# ---------- running the model over the source ----------

def stamped(f, size=(1080, 1920)):
    """A black frame that carries its own number in its first pixel, so a fake model knows which frame it sees."""
    frame = np.zeros((*size, 3), np.uint8)
    frame[0, 0] = (f % 256, f // 256 % 256, 0)
    return frame


def number(frame):
    return int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])


class FakeReader:
    def __init__(self, path):
        self.reads = []
        FakeReader.last = self

    def read(self, f):
        self.reads.append(f)
        return stamped(f)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class FakeModel:
    """Answers with the made-up poses of the tracklets it is asked about, by the box it is given."""

    batch = 8

    def __init__(self, tracklets, how=None):
        """`how` are the scores of some of the people: {tracklet id: keyword arguments of answers()}."""
        self.by_box = {}
        for t in tracklets:
            for i, kp in enumerate(answers(t, **(how or {}).get(t.id, {}))):
                self.by_box[tuple(np.round(t.box[i], 2).tolist()) + (int(t.frames[i]),)] = kp
        self.asked = 0

    def many(self, items):
        self.asked += len(items)
        return np.stack([self.by_box[tuple(np.round(box, 2).tolist()) + (number(frame),)] for frame, box in items])


def test_a_person_sample_asked_for_twice_goes_through_the_model_once(monkeypatch, reaching):
    monkeypatch.setattr(hands.media, "Reader", FakeReader)
    model = FakeModel([reaching])
    clip = hands.clip_request(("clip", 0), reaching, 40 * STEP, 60 * STEP)
    stretch = hands.stretch_request(("person", 0), reaching, 30 * STEP, 70 * STEP)
    seen = []
    out = hands.look("video.mp4", [clip, stretch], model=model, progress=seen.append)
    wanted = set(clip.samples()) | set(stretch.samples())
    assert model.asked == len(wanted) and seen[-1] == pytest.approx(1.0)
    assert set(out) == {("clip", 0), ("person", 0)}
    assert len(out[("clip", 0)].frames) == len(clip.samples()) and out[("clip", 0)].seen.all()
    assert FakeReader.last.reads == sorted(FakeReader.last.reads)          # one pass over the source, in order


def test_a_thinned_out_stretch_is_filled_in_between(monkeypatch, reaching):
    monkeypatch.setattr(hands.media, "Reader", FakeReader)
    request = hands.stretch_request(0, reaching, 0, 90 * STEP, stride=3)
    track = hands.look("video.mp4", [request], model=FakeModel([reaching]))[0]
    assert track.step == 3 * STEP and len(track.frames) == 30
    rows = hands.person_rows(track, reaching.frames[:60])                  # the page gets every sample of the tracklet
    assert len(rows) == 60 and all(clear for _kp, _hands, clear in rows.values())


def test_stretch_requests_stay_within_a_budget(monkeypatch):
    people_ = []
    for i in range(12):
        t = person(n=160, x=200 + 120 * i)
        t.id = i
        people_.append(t)
    monkeypatch.setattr(wholebody, "accelerated", lambda: True)
    full = live.stretch_requests(people_, (0, 140 * STEP))
    assert len(full) == 12 and sum(len(r.samples()) for r in full) <= 600 and full[0].stride == 3    # 1,716 samples asked for
    monkeypatch.setattr(wholebody, "accelerated", lambda: False)
    lean = live.stretch_requests(people_, (0, 140 * STEP))
    assert sum(len(r.samples()) for r in lean) <= 240 and lean[0].stride > full[0].stride


# ---------- the pipeline's use of it ----------

class Quiet:
    """A run that only remembers what it said."""

    def __init__(self, tmp_path):
        self.dir, self.said = tmp_path, []

    def say(self, msg):
        self.said.append(msg)


@pytest.fixture
def three_people():
    """Three people who each reach once. Stage 2 flagged the left hand of all of them (it was the right)."""
    tracklets, cands = [], []
    for i in range(3):
        t = reach(person(x=400 + 500 * i), 40, 60)
        t.id = i
        tracklets.append(t)
        cands.append(interactions.Candidate(track=i, start=35 * STEP, end=65 * STEP, crop=(0, 0, 640, 360), hands=["left"],
                                            busy_s=1.0, peak_speed=1.0, extent=1.0, scale=136.0, seen=1.0, score=0.9 - 0.1 * i))
    return tracklets, cands


def test_a_moment_with_no_locatable_hand_is_replaced_by_the_next_best(monkeypatch, tmp_path, three_people):
    tracklets, cands = three_people
    monkeypatch.setattr(hands.media, "Reader", FakeReader)
    model = FakeModel(tracklets, {0: dict(body_score=BAD)})                # the best-scoring person cannot be read
    monkeypatch.setattr(wholebody, "available", lambda: True)
    monkeypatch.setattr(wholebody, "shared", lambda: model)
    monkeypatch.setattr(pipeline, "setting", lambda name, default: default)
    run = Quiet(tmp_path)
    chosen, tracks, found, poses = pipeline.locate_hands(run, "video.mp4", cands, {t.id: t for t in tracklets}, 2, None,
                                                         (1920, 1080), None, tracklets)
    assert chosen == [1, 2]                                                # person 0 is out, person 2 took the place
    assert found[0].hands == [] and set(tracks) == {0, 1, 2} and poses == {}
    assert cands[1].hands == ["right"] and cands[2].hands == ["right"]     # decided again on the careful keypoints
    assert cands[1].crop != (0, 0, 640, 360) and cands[0].crop == (0, 0, 640, 360)
    assert run.said == []


def test_without_the_model_the_engine_works_as_before(monkeypatch, tmp_path, three_people):
    tracklets, cands = three_people
    monkeypatch.setattr(wholebody, "available", lambda: False)
    run = Quiet(tmp_path)
    chosen, tracks, found, poses = pipeline.locate_hands(run, "video.mp4", cands, {t.id: t for t in tracklets}, 2, None,
                                                         (1920, 1080), None, tracklets)
    assert chosen == [0, 1] and tracks == {} and found == {} and cands[0].hands == ["left"]
    assert "estimates from the wrist" in run.said[0] and "onnxruntime" in run.said[0]
    monkeypatch.setattr(pipeline, "setting", lambda name, default: "off" if name == "WHOLEBODY" else default)
    pipeline.locate_hands(run, "video.mp4", cands, {t.id: t for t in tracklets}, 2, None, (1920, 1080), None, tracklets)
    assert "switched off" in run.said[1]


def test_a_model_that_fails_does_not_sink_the_run(monkeypatch, tmp_path, three_people):
    tracklets, cands = three_people
    monkeypatch.setattr(wholebody, "available", lambda: True)
    monkeypatch.setattr(pipeline, "setting", lambda name, default: default)

    def broken(*a, **k):
        raise RuntimeError("no such provider")

    monkeypatch.setattr(hands, "look", broken)
    run = Quiet(tmp_path)
    chosen, tracks, found, poses = pipeline.locate_hands(run, "video.mp4", cands, {t.id: t for t in tracklets}, 2, None,
                                                         (1920, 1080), None, tracklets)
    assert chosen == [0, 1] and tracks == {} and "no such provider" in run.said[0]


def test_select_passes_over_excluded_candidates():
    cands = [interactions.Candidate(track=i, start=0, end=90, crop=(0, 0, 320, 180), hands=["right"], busy_s=1.0,
                                    peak_speed=1.0, extent=1.0, scale=100.0, seen=1.0, score=0.9 - 0.1 * i) for i in range(4)]
    assert interactions.select(cands, 2) == [0, 1]
    assert interactions.select(cands, 2, exclude={0}) == [1, 2]
    assert interactions.select(cands, 2, {0: 3, 1: 0, 2: 2, 3: 1}, exclude={0}) == [2, 3]


def test_the_fallback_estimate_names_its_busy_hands(reaching):
    from relentless import activity

    sig = activity.signals(reaching)
    cand = interactions.find(reaching, sig, activity.active_spans(reaching, sig), (1920, 1080))[0]
    out = pipeline.hand_tracks(reaching, sig, cand)
    assert out["busy"] == ["right"] and "estimated" in out["source"] and "w" not in out["right"]


# ---------- the previews of a job page ----------

def test_the_replayed_stretch_is_drawn_from_the_careful_poses(tmp_path, reaching):
    reaching.id = 0
    other = person(n=100, x=1400)
    other.id = 1
    kit = live.Kit(tmp_path / "live", 0, 100 * STEP, FPS, (1920, 1080))
    poses = {0: track_of(reaching), 1: track_of(other, body_score=BAD)}
    out = kit.tracks([reaching, other], {0: [(0, 300)], 1: [(0, 300)]}, [], 30, 240, poses)
    assert out["pose"].startswith("RTMW")
    rows = {p["id"]: p for p in out["samples"][10]["people"]}
    assert rows[0]["busy"] is True and rows[0]["clear"] is True and set(rows[0]["hands"]) == {"l", "r"}
    assert all(c >= 0.9 for _x, _y, c in rows[0]["kp"][5:11])
    # the person the model cannot read: no joints, no hands, and not called busy
    assert rows[1]["clear"] is False and rows[1]["busy"] is False and rows[1]["hands"] == {}
    assert all(c == 0 for _x, _y, c in rows[1]["kp"])
    # a person the model did not look at is drawn from stage 2, as before: no `hands`, no `clear`
    plain = kit.tracks([reaching], {0: []}, [], 30, 240, {})
    assert "hands" not in plain["samples"][0]["people"][0] and plain["pose"] is None


def test_the_stretch_is_decided_once(tmp_path, reaching):
    kit = live.Kit(tmp_path / "live", 0, 3000, FPS, (1920, 1080))
    first = kit.stretch([reaching], [(600, 700), (650, 760)])
    assert first == kit.stretch([reaching], [(2000, 2100)])               # later callers get the same seconds
    assert first[1] - first[0] == 14 * FPS and first[0] <= 600 and first[1] >= 760


def wait_for(path, seconds=5.0):
    """The second look of a live frame is written by a thread: give it a moment."""
    import time

    end = time.time() + seconds
    while time.time() < end:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(0.01)
    raise AssertionError(f"{path.name} was not written")


def test_the_live_frame_shows_the_second_look(tmp_path, standing, monkeypatch):
    standing.id = 3
    model = type("M", (), {"__call__": lambda self, frame, boxes: answers(standing)[:len(boxes)]})()
    monkeypatch.setattr(wholebody, "shared", lambda: model)
    view = live.TrackingView(tmp_path, 0, 3000, FPS, every=0.0, refine=True)
    frame = np.full((1080, 1920, 3), 90, np.uint8)
    for i in range(live.REACH + 2):
        view(i * STEP, frame, [(3, standing.box[i], standing.kp[i])], [], i + 1)
    shown = wait_for(tmp_path / "now.json")["people"][0]
    assert shown["id"] == 3 and shown["clear"] is True and set(shown["hands"]) == {"l", "r"}
    view.close()
    # without the second look the page gets stage 2's keypoints, and no hands to draw as boxes
    plain = live.TrackingView(tmp_path / "plain", 0, 3000, FPS, every=0.0)
    for i in range(live.REACH + 2):
        plain(i * STEP, frame, [(3, standing.box[i], standing.kp[i])], [], i + 1)
    assert "hands" not in json.loads((tmp_path / "plain" / "now.json").read_text())["people"][0]


def test_tracking_never_waits_for_the_second_look(tmp_path, standing, monkeypatch):
    """The model is slow (here: stuck). Tracking goes on at its own pace, and when tracking is over nothing
    more is written, whatever the model was still busy with."""
    import threading
    import time

    release, asked = threading.Event(), []

    class Slow:
        def __call__(self, frame, boxes):
            asked.append(len(boxes))
            release.wait(5)
            return answers(standing)[:len(boxes)]

    monkeypatch.setattr(wholebody, "shared", lambda: Slow())
    kit = live.Kit(tmp_path / "live", 0, 3000, FPS, (1920, 1080))
    monkeypatch.setattr(kit, "spec", lambda: {"folder": str(kit.folder), "first": 0, "last": 3000, "fps": FPS, "refine": True})
    view = kit.tracking()
    view.every = 0.0
    frame = np.full((1080, 1920, 3), 90, np.uint8)
    crowd = [(n, standing.box[0] + [40 * n, 0, 40 * n + n, 0], standing.kp[0]) for n in range(12)]   # 12 people, #11 the widest
    started = time.time()
    for i in range(live.REACH + 6):
        view(i * STEP, frame, crowd, [], i + 1)
    assert time.time() - started < 2.0                         # six frames handed over while the model sat on the first
    assert not (kit.folder / "now.json").exists()
    kit.quiet()                                                # tracking is over
    release.set()
    time.sleep(0.2)
    assert not (kit.folder / "now.json").exists()              # the frame the model was on is not written after that
    assert asked and max(asked) == live.LOOK_AT                # one model call's worth of people, not all twelve


def test_the_overview_draws_no_skeleton_on_someone_it_cannot_read(tmp_path, standing):
    standing.id = 0
    job = render.Overview(tmp_path / "o.mp4", 0, 300, (1920, 1080), FPS, STEP, [standing], {0: [(0, 300)]}, [],
                          poses={0: [track_of(standing, body_score=BAD)]})
    assert (job._joints(standing, 30)[:, 2] == 0).all()
    job.poses = {0: [track_of(standing)]}
    assert (job._joints(standing, 30)[5:11, 2] > 0.9).all()
    job.poses = {}
    assert np.array_equal(job._joints(standing, 30), standing.kp[10])       # nobody looked again: stage 2's joints


# ---------- the variant check's measurement ----------

def test_same_path_finds_the_hands_where_the_track_has_them(monkeypatch, reaching):
    track = track_of(reaching)
    crop = (500, 200, 640, 360)
    scale = 1280 / 640                                                       # the video is the crop, twice as large

    class Clip(FakeReader):
        width, height, fps, count = 1280, 720, FPS, 120

        def read(self, f):
            self.reads.append(f)
            return stamped(f, (720, 1280))

    poses = answers(reaching)

    class Echo:
        """Finds the hands exactly where the track has them, or a few hand sizes away."""

        def __init__(self, shift=0.0):
            self.shift = shift

        def many(self, items):
            out = np.zeros((len(items), 133, 3), np.float32)
            for n, (frame, _box) in enumerate(items):          # frame k of the video is source frame 35 * STEP + k
                kp = poses[track.at(35 * STEP + number(frame))].copy()
                kp[:, 0] = (kp[:, 0] - crop[0]) * scale + self.shift
                kp[:, 1] = (kp[:, 1] - crop[1]) * scale
                out[n] = kp
            return out

    monkeypatch.setattr(hands.media, "Reader", Clip)
    same = hands.same_path("variant.mp4", track, crop, 35 * STEP, 0.0, 3.0, model=Echo())
    assert same["left"]["same"] == 1.0 and same["right"]["same"] == 1.0 and same["right"]["n"] >= 25
    moved = hands.same_path("variant.mp4", track, crop, 35 * STEP, 0.0, 3.0, model=Echo(shift=260))
    assert moved["left"]["found"] == 1.0 and moved["left"]["same"] == 0.0     # hands are there, but not where they were


# ---------- the labeler's marks ----------

def test_the_labelers_frames_are_marked_only_with_located_hands(reaching):
    track = track_of(reaching, hand_score=(GOOD, BAD))
    out = hands.tracks_json(track, 0, 90, (500, 200, 640, 360), ["left"], "RTMW-x 384x288")
    assert label.marked(out)
    frame = np.zeros((360, 640, 3), np.uint8)
    assert label.mark_hands(frame, out, 10) == 1 and frame.any()             # the left hand only
    blue = (frame[:, :, 0] > 200) & (frame[:, :, 2] < 100)
    assert blue.any() and not ((frame[:, :, 2] > 200) & (frame[:, :, 0] < 100)).any()   # no orange R anywhere
    # an estimate from the wrist is not good enough to draw on a labeler's frames
    estimate = {"fps": 30.0, "frames": 90, "box_size": [0.1, 0.1], "left": {"x": [0.5] * 90, "y": [0.5] * 90, "seen": [True] * 90},
                "right": {"x": [0.5] * 90, "y": [0.5] * 90, "seen": [True] * 90}}
    assert not label.marked(estimate) and not label.marked(None)
    nobody = hands.tracks_json(track_of(reaching, body_score=BAD), 0, 90, (500, 200, 640, 360), [], "RTMW-x 384x288")
    assert not label.marked(nobody)                                          # nothing located: nothing to mark
