import math
import random

import pytest

from fencebot.dubins import dubins_paths, sample

R = 590


def close_pose(a, b, pos=1e-6, ang=1e-6):
    dh = (a[2] - b[2] + math.pi) % (2 * math.pi) - math.pi
    return math.dist(a[:2], b[:2]) < pos and abs(dh) < ang


def test_straight_ahead_is_single_straight():
    paths = dubins_paths((0, 0, 0), (1000, 0, 0), R)
    best = paths[0]
    assert best.length == pytest.approx(1000)
    assert close_pose(sample(best, 50)[-1], (1000, 0, 0))


def test_paths_are_sorted_by_length():
    paths = dubins_paths((0, 0, 0), (500, 800, math.pi), R)
    lengths = [p.length for p in paths]
    assert lengths == sorted(lengths) and len(paths) >= 2


def test_every_word_ends_at_goal_random():
    rng = random.Random(1)
    for _ in range(200):
        start = (rng.uniform(-3000, 3000), rng.uniform(-3000, 3000), rng.uniform(-math.pi, math.pi))
        goal = (rng.uniform(-3000, 3000), rng.uniform(-3000, 3000), rng.uniform(-math.pi, math.pi))
        for p in dubins_paths(start, goal, R):
            end = sample(p, 30)[-1]
            assert close_pose(end, goal, pos=1e-3, ang=1e-6), p.word


def test_sample_spacing_and_curvature():
    p = dubins_paths((0, 0, 0), (0, 2 * R, math.pi), R)[0]  # 왼쪽 반원
    assert p.length == pytest.approx(math.pi * R)
    pts = sample(p, 20)
    assert pts[0] == (0, 0, 0)
    steps = [math.dist(a[:2], b[:2]) for a, b in zip(pts, pts[1:])]
    assert max(steps) <= 20 + 1e-9
    assert all(math.hypot(x, y - R) == pytest.approx(R) for x, y, _ in pts)
