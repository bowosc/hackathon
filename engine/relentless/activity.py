"""Stage 3 - activity filter: keep only the parts of each person's tracklet where their hands are busy.

Works on the keypoints stage 2 already produced, so it is pure arithmetic. For every sample it
measures how fast each wrist moves relative to the body (in torso lengths per second, so the
number means the same thing for a person near the camera and one far away). Someone standing
still, sitting idle or just walking past scores low and is dropped.

It also marks when a hand is in the "working zone" (raised above the hips or reaching out
from the body). Stage 4 uses that to tell handling something apart from arms swinging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .people import L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, R_ELBOW, R_HIP, R_SHOULDER, R_WRIST, Tracklet

SIDES = ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST))  # index 0 = left, 1 = right
LEGS = (13, 14, 15, 16)  # knees and ankles


@dataclass
class ActivityConfig:
    kp_conf: float = 0.45         # keypoints below this confidence are treated as unseen
    hand_speed: float = 0.4       # torso lengths per second: wrist motion that counts as moving
    hand_speed_px: float = 25.0   # and at least this many source pixels per second (keypoint jitter floor)
    walk_speed: float = 0.3       # body heights per second: above this the person is walking
    leg_speed: float = 1.0        # torso lengths per second of knee/ankle motion: also walking
    walk_factor: float = 2.5      # while walking, wrists must move this many times faster to count
    raise_margin: float = 0.1     # torso lengths: wrist this far above the hip line counts as raised
    reach_out: float = 0.75       # torso lengths sideways from the body axis that counts as reaching out
    window_s: float = 1.0         # a sample is active if enough of the surrounding window is moving
    window_share: float = 0.2
    pad_s: float = 0.7            # keep this long before and after busy moments
    merge_gap_s: float = 1.5
    min_span_s: float = 1.5


@dataclass
class Signals:
    """Per-sample measurements for one tracklet (arrays share the tracklet's sample grid)."""

    scale: np.ndarray        # (n,) torso length in px
    hand: np.ndarray         # (n, 2, 2) estimated hand centre in px, left then right
    seen: np.ndarray         # (n, 2) wrist confidently detected
    speed: np.ndarray        # (n, 2) wrist speed relative to the body, torso lengths/s
    working: np.ndarray      # (n, 2) wrist raised above the hips or reaching out from the body
    walking: np.ndarray      # (n,) the person is moving through the scene
    moving: np.ndarray       # (n, 2) wrist moving fast enough to matter (stricter while walking)
    busy: np.ndarray         # (n, 2) moving and in the working zone: a likely manipulation


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average along axis 0 with edge padding."""
    if k <= 1 or len(x) < 2:
        return x
    pad = k // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)])
    c = np.cumsum(np.concatenate([np.zeros_like(xp[:1]), xp]), axis=0)
    return (c[k:] - c[:-k]) / k


def _median(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x
    pad = k // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)])
    return np.median(np.stack([xp[i:i + len(x)] for i in range(k)]), axis=0)


def _derivative(x: np.ndarray, dt: float) -> np.ndarray:
    """Central difference along axis 0 (one-sided at the ends)."""
    if len(x) < 2:
        return np.zeros_like(x)
    d = np.empty_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    d[0], d[-1] = (x[1] - x[0]) / dt, (x[-1] - x[-2]) / dt
    return d


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    return _smooth(mask.astype(np.float32), k) > 0 if k > 1 else mask


def body_scale(t: Tracklet, conf: float) -> np.ndarray:
    """Torso length in px per sample.

    Shoulder-to-hip distance where the model sees both, taken as a high percentile over a
    long window (a torso looks shorter when someone leans or turns, never longer). When hips
    are hidden, e.g. behind a counter, fall back to a share of the box height.
    """
    kp, box = t.kp, t.box
    box_h = np.maximum(box[:, 3] - box[:, 1], 1.0)
    sh, hip = kp[:, [L_SHOULDER, R_SHOULDER]], kp[:, [L_HIP, R_HIP]]
    ok = (sh[:, :, 2] > conf).all(axis=1) & (hip[:, :, 2] > conf).all(axis=1)
    torso = np.linalg.norm(sh[:, :, :2].mean(axis=1) - hip[:, :, :2].mean(axis=1), axis=1)
    ok &= torso > 0.15 * box_h
    fallback = 0.3 * box_h
    n = len(kp)
    half = max(1, int(round(8.0 * t.fps / t.step)))
    scale = np.empty(n, np.float32)
    whole = float(np.percentile(torso[ok], 85)) if ok.sum() >= 5 else None
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        local = torso[a:b][ok[a:b]]
        if len(local) >= 5:
            scale[i] = np.percentile(local, 85)
        elif whole is not None and abs(np.log(box_h[i] / np.median(box_h))) < 0.25:
            scale[i] = whole
        else:
            scale[i] = fallback[i]
    return np.maximum(scale, 8.0)


def signals(t: Tracklet, cfg: ActivityConfig | None = None) -> Signals:
    cfg = cfg or ActivityConfig()
    dt = t.step / t.fps
    per_s = max(1, int(round(1 / dt)))
    kp, box = t.kp, t.box
    n = len(kp)
    box_h = np.maximum(box[:, 3] - box[:, 1], 1.0)
    scale = body_scale(t, cfg.kp_conf)

    sh, hip = kp[:, [L_SHOULDER, R_SHOULDER]], kp[:, [L_HIP, R_HIP]]
    sh_ok = (sh[:, :, 2] > cfg.kp_conf).all(axis=1)
    hip_ok = (hip[:, :, 2] > cfg.kp_conf).all(axis=1)
    sh_mid = np.where(sh_ok[:, None], sh[:, :, :2].mean(axis=1),
                      np.stack([(box[:, 0] + box[:, 2]) / 2, box[:, 1] + 0.2 * box_h], axis=1))
    sh_mid = _smooth(sh_mid, 3)
    # body axis (unit vector from shoulders towards hips); straight down when the hips are hidden
    down = np.where(hip_ok[:, None], hip[:, :, :2].mean(axis=1) - sh_mid, np.array([0.0, 1.0]))
    down = _smooth(down / np.maximum(np.linalg.norm(down, axis=1, keepdims=True), 1e-6), per_s // 2 * 2 + 1)
    down /= np.maximum(np.linalg.norm(down, axis=1, keepdims=True), 1e-6)

    hand = np.zeros((n, 2, 2), np.float32)
    seen = np.zeros((n, 2), bool)
    speed = np.zeros((n, 2), np.float32)
    working = np.zeros((n, 2), bool)
    for s, (_i_sh, i_el, i_wr) in enumerate(SIDES):
        wr, el = kp[:, i_wr], kp[:, i_el]
        seen[:, s] = wr[:, 2] > cfg.kp_conf
        fore = np.where(el[:, 2:3] > cfg.kp_conf, wr[:, :2] - el[:, :2], 0.0)
        hand[:, s] = wr[:, :2] + 0.35 * fore   # the hand sits a bit past the wrist along the forearm
        rel = _smooth(_median((wr[:, :2] - sh_mid) / scale[:, None], 3), 3)
        v = np.linalg.norm(_derivative(rel, dt), axis=1)
        # a wrist that is not confidently seen contributes no motion, and neither do its neighbours
        ok = seen[:, s] & np.roll(seen[:, s], 1) & np.roll(seen[:, s], -1)
        speed[:, s] = np.where(ok, v, 0.0)
        along = (rel * down).sum(axis=1)                       # how far down the body the wrist is
        across = np.abs(rel[:, 0] * down[:, 1] - rel[:, 1] * down[:, 0])  # how far out to the side
        working[:, s] = seen[:, s] & ((along < 1.0 - cfg.raise_margin) | (across > cfg.reach_out))

    # locomotion: the feet travel across the image, or (walking towards/away from the camera) the legs swing
    feet = _smooth(np.stack([(box[:, 0] + box[:, 2]) / 2, box[:, 3]], axis=1), per_s // 2 * 2 + 1)
    body_speed = _smooth(np.linalg.norm(_derivative(feet, dt), axis=1) / box_h, 3)
    legs = kp[:, LEGS]
    leg_rel = _smooth((legs[:, :, :2] - sh_mid[:, None]) / scale[:, None, None], 3)
    leg_v = np.linalg.norm(_derivative(leg_rel, dt), axis=2)
    leg_ok = legs[:, :, 2] > cfg.kp_conf
    leg_speed = np.where(leg_ok.any(axis=1), (leg_v * leg_ok).sum(axis=1) / np.maximum(leg_ok.sum(axis=1), 1), 0.0)
    leg_speed = _smooth(leg_speed, per_s // 2 * 2 + 1)
    walking = _dilate((body_speed > cfg.walk_speed) | (leg_speed > cfg.leg_speed), per_s // 2 * 2 + 1)

    need = np.maximum(cfg.hand_speed, cfg.hand_speed_px / scale) * np.where(walking, cfg.walk_factor, 1.0)
    moving = speed > need[:, None]
    return Signals(scale, hand, seen, speed, working, walking, moving, moving & working)


def spans(mask: np.ndarray, frames: np.ndarray, step: int, fps: float, pad_s: float, merge_gap_s: float,
          min_span_s: float) -> list[tuple[int, int]]:
    """[start, end) source-frame spans where `mask` is on, padded, merged and length-filtered."""
    out: list[list[int]] = []
    pad = int(round(pad_s * fps))
    lo, hi = int(frames[0]), int(frames[-1]) + step
    for f, on in zip(frames, mask):
        if not on:
            continue
        a, b = max(lo, int(f) - pad), min(hi, int(f) + step + pad)
        if out and a - out[-1][1] <= merge_gap_s * fps:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out if b - a >= min_span_s * fps]


def active_spans(t: Tracklet, sig: Signals, cfg: ActivityConfig | None = None) -> list[tuple[int, int]]:
    """Source-frame spans of the tracklet worth keeping: hands moving for a meaningful share of a window."""
    cfg = cfg or ActivityConfig()
    k = max(1, int(round(cfg.window_s * t.fps / t.step))) // 2 * 2 + 1
    share = _smooth(sig.moving.any(axis=1).astype(np.float32), k)
    return spans(share >= cfg.window_share, t.frames, t.step, t.fps, cfg.pad_s, cfg.merge_gap_s, cfg.min_span_s)
