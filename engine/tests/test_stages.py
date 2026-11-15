import numpy as np

from conftest import FPS, STEP, person, reach
from relentless import activity, interactions, media, motion, people


def test_motion_spans_pad_merge_and_drop():
    frames = np.arange(0, 900, 3)
    active = np.zeros(len(frames), bool)
    active[30:40] = True      # frames 90-120
    active[50:60] = True      # frames 150-180, close enough to merge
    active[250] = True        # a lone blip far away
    spans = motion.spans_from_mask(active, frames, FPS, 0, 900, hold_s=1.0, merge_gap_s=2.0, min_span_s=1.0)
    assert spans[0] == (60, 210)                 # padded by 1 s each side and merged
    assert spans[1][0] == 720 and len(spans) == 2
    assert motion.spans_from_mask(np.zeros(len(frames), bool), frames, FPS, 0, 900, 1.0, 2.0, 1.0) == []


def test_fit_box_is_even_clamped_and_keeps_aspect():
    x, y, w, h = media.fit_box(10, 10, 300, 100, 1920, 1080, aspect=16 / 9)
    assert (x, y) == (0, 0) and w % 2 == 0 and h % 2 == 0
    assert abs(w / h - 16 / 9) < 0.03
    x, y, w, h = media.fit_box(1900, 1070, 4000, 4000, 1920, 1080, aspect=16 / 9)
    assert (w, h) == (1920, 1080) and (x, y) == (0, 0)


def test_standing_person_is_idle(standing):
    sig = activity.signals(standing)
    assert not sig.moving.any() and not sig.walking.any()
    assert activity.active_spans(standing, sig) == []
    assert abs(float(np.median(sig.scale)) - 0.34 * 400) < 2      # torso length in px


def test_reach_is_found_on_the_right_hand_only(reaching):
    sig = activity.signals(reaching)
    assert sig.busy[:, 1].any() and not sig.busy[:, 0].any()      # index 1 = right wrist (keypoint 10)
    spans = activity.active_spans(reaching, sig)
    assert len(spans) == 1
    a, b = spans[0]
    assert a <= 40 * STEP and b >= 60 * STEP                      # covers the reach, with padding


def test_walking_arm_swing_is_not_activity():
    t = person(n=120)
    drift = np.linspace(0, 3000, 120)                             # 0.6 body heights per second: a normal walk
    swing = 30 * np.sin(np.linspace(0, 24 * np.pi, 120))          # arms swinging along the body
    t.box[:, [0, 2]] += drift[:, None]
    t.kp[:, :, 0] += drift[:, None]
    t.kp[:, 9, 1] += swing
    t.kp[:, 10, 1] -= swing
    sig = activity.signals(t)
    assert sig.walking.mean() > 0.9
    assert not sig.busy.any()


def test_hidden_wrist_contributes_no_motion(reaching):
    reaching.kp[:, 10, 2] = 0.1                                   # right wrist never confidently seen
    assert not activity.signals(reaching).moving.any()


def test_candidate_window_and_crop(reaching):
    sig = activity.signals(reaching)
    cands = interactions.find(reaching, sig, activity.active_spans(reaching, sig), (1920, 1080))
    assert len(cands) == 1
    c = cands[0]
    assert c.hands == ["right"]
    assert 2.0 <= (c.end - c.start) / FPS <= 8.0
    x, y, w, h = c.crop
    assert w % 2 == 0 and h % 2 == 0 and abs(w / h - 16 / 9) < 0.03
    seen = sig.hand[reaching.at(c.start):reaching.at(c.end - 1) + 1, 1]
    assert (seen[:, 0] >= x).all() and (seen[:, 0] <= x + w).all()   # the whole hand path is inside the crop
    assert (seen[:, 1] >= y).all() and (seen[:, 1] <= y + h).all()
    assert 0 < c.score <= 1


def test_small_person_gives_no_candidate():
    t = reach(person(height=90), 40, 60)                          # torso about 30 px: hands are a few pixels
    sig = activity.signals(t)
    assert interactions.find(t, sig, activity.active_spans(t, sig), (1920, 1080)) == []


def _cand(track, score, start=0):
    return interactions.Candidate(track=track, start=start, end=start + 90, crop=(0, 0, 320, 180), hands=["right"],
                                  busy_s=1.0, peak_speed=1.0, extent=1.0, scale=100.0, seen=1.0, score=score)


def test_select_by_motion_score_spreads_over_people():
    cands = [_cand(0, 0.9), _cand(0, 0.8), _cand(0, 0.7), _cand(0, 0.6), _cand(1, 0.5), _cand(2, 0.4)]
    picked = interactions.select(cands, 4)
    assert picked[:2] == [0, 1]                 # best two of the busiest person
    assert set(picked) == {0, 1, 4, 5}          # then other people before a third clip of person 0
    assert interactions.select(cands, None) == [0, 1, 2, 3, 4, 5]


def test_select_with_ratings_drops_zero_and_prefers_rated():
    cands = [_cand(0, 0.9), _cand(1, 0.5), _cand(2, 0.4), _cand(3, 0.3)]
    rating = {0: 0, 1: 2, 2: 3}                 # candidate 3 was not screened: counts as 1
    assert interactions.select(cands, 10, rating) == [2, 1, 3]
    assert interactions.select(cands, 1, rating) == [2]


def test_build_tracklets_fills_short_gaps_and_splits_long_ones():
    def rows(frames, tid):
        n = len(frames)
        return dict(frame=np.array(frames), id=np.full(n, tid), box=np.tile([100.0, 100, 250, 500], (n, 1)),
                    conf=np.full(n, 0.9), kp=np.full((n, 17, 3), 0.9))
    a = rows([f for f in range(0, 150, 3) if f not in (30, 33)], 0)       # 0.2 s hole: interpolated
    b = rows(list(range(300, 360, 3)) + list(range(480, 540, 3)), 1)      # 4 s hole: two pieces
    raw = {k: np.concatenate([a[k], b[k]]) for k in a}
    raw.update(step=3, fps=30.0, loose={})
    kept, dropped = people.build_tracklets(raw)
    assert [len(t.frames) for t in kept] == [50, 20, 20]
    assert kept[0].conf[10] == 0 and kept[0].conf[9] > 0                  # the filled samples carry no confidence
    assert [t.id for t in kept] == [0, 1, 2]
    short = rows(list(range(0, 30, 3)), 0)
    short.update(step=3, fps=30.0, loose={})
    assert people.build_tracklets(short) == ([], {"short": 1, "small": 0})


def test_head_circle_covers_the_face(standing):
    cx, cy, r = people.head_circle(standing.box[0], standing.kp[0])
    face = standing.kp[0, :5, :2]
    assert (np.hypot(face[:, 0] - cx, face[:, 1] - cy) < r).all()
    # no keypoints at all: fall back to the top of the box
    cx, cy, r = people.head_circle(standing.box[0], np.zeros((17, 3)))
    assert standing.box[0, 0] < cx < standing.box[0, 2] and standing.box[0, 1] < cy < standing.box[0, 1] + 100 and r > 20


def test_head_circles_interpolate_between_samples(standing):
    raw = dict(frame=standing.frames, id=np.zeros(len(standing.frames), int), box=standing.box, kp=standing.kp,
               step=STEP, loose={})
    heads = people.head_circles(raw)
    assert set(range(0, int(standing.frames[-1]) + STEP)) <= set(heads)   # every source frame has a circle
    assert heads[1].shape == (1, 3)


def test_tracking_in_a_child_process_returns_what_it_saved(tmp_path, monkeypatch):
    """Low-memory mode: the child tracks, saves and exits; the parent reads the file and gets the progress."""
    import numpy as np

    from relentless import people

    raw = {"frame": np.array([0, 3]), "id": np.array([1, 1]), "box": np.zeros((2, 4), np.float32),
           "conf": np.ones(2, np.float32), "kp": np.zeros((2, 17, 3), np.float32), "loose": {}, "step": 3,
           "fps": 30.0, "size": (1920, 1080), "samples": 2, "seconds": 0.5}

    class FakeProcess:
        def __init__(self, target, args, daemon):
            self.target, self.args, self.exitcode, self.alive = target, args, None, True

        def start(self):
            src, spans, out, updates, live = self.args
            updates.put(0.5)
            people.save(out, raw)
            self.exitcode = 0

        def is_alive(self):
            self.alive, was = False, self.alive      # one pass through the wait loop, then done
            return was

        def join(self):
            pass

    class FakeContext:
        Process = FakeProcess

        @staticmethod
        def Queue():
            import queue
            return queue.Queue()

    import multiprocessing
    monkeypatch.setattr(multiprocessing, "get_context", lambda kind: FakeContext)
    seen = []
    back = people.track_apart(tmp_path / "video.mp4", [(0, 6)], tmp_path / "tracks.npz", progress=seen.append)
    assert seen == [0.5] and back["samples"] == 2 and back["step"] == 3 and list(back["frame"]) == [0, 3]
