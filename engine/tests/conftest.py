"""Synthetic people for testing the stages without video or models."""

import numpy as np
import pytest

from relentless.people import Tracklet

FPS, STEP = 30.0, 3


def person(n=100, x=800.0, y=300.0, height=400.0, start=0):
    """A person standing still, facing the camera: keypoints laid out on a simple skeleton."""
    kp = np.zeros((n, 17, 3), np.float32)
    kp[:, :, 2] = 0.9
    sh_y, hip_y = y + 0.18 * height, y + 0.52 * height
    layout = {0: (0, 0.08), 1: (-0.02, 0.07), 2: (0.02, 0.07), 3: (-0.05, 0.08), 4: (0.05, 0.08),
              5: (-0.11, 0.18), 6: (0.11, 0.18), 7: (-0.13, 0.35), 8: (0.13, 0.35), 9: (-0.14, 0.52), 10: (0.14, 0.52),
              11: (-0.07, 0.52), 12: (0.07, 0.52), 13: (-0.07, 0.75), 14: (0.07, 0.75), 15: (-0.07, 0.97), 16: (0.07, 0.97)}
    for k, (dx, dy) in layout.items():
        kp[:, k, 0] = x + dx * height
        kp[:, k, 1] = y + dy * height
    box = np.tile(np.array([x - 0.17 * height, y, x + 0.17 * height, y + height], np.float32), (n, 1))
    frames = start + np.arange(n) * STEP
    assert abs((hip_y - sh_y) - 0.34 * height) < 1e-3
    return Tracklet(0, frames, box, kp, np.full(n, 0.9, np.float32), STEP, FPS)


def reach(t, a, b, side=10, dx=0.45, dy=-0.25):
    """Make one wrist (and its elbow) swing out and back between samples a and b."""
    height = float(t.box[0, 3] - t.box[0, 1])
    phase = np.sin(np.linspace(0, np.pi, b - a))
    t.kp[a:b, side, 0] += dx * height * phase
    t.kp[a:b, side, 1] += dy * height * phase
    t.kp[a:b, side - 2, 0] += 0.5 * dx * height * phase
    t.kp[a:b, side - 2, 1] += 0.5 * dy * height * phase
    return t


@pytest.fixture
def standing():
    return person()


@pytest.fixture
def reaching():
    return reach(person(), 40, 60)
