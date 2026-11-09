"""전진만 쓰는 Dubins 경로 (배치_시뮬레이션_계획 4-2). 순수 기하, 단위 mm·라디안.

여섯 가지 단어(LSL, RSR, LSR, RSL, RLR, LRL)를 모두 계산해 짧은 순서로 돌려준다.
가장 짧은 경로가 장애물에 막히면 다음 경로를 시험할 수 있게 하기 위해서다.
식은 Shkel & Lumelsky(2001)의 정규화 형식(반지름 1)을 따른다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Pose = tuple[float, float, float]  # (x, y, heading)

_TAU = 2 * math.pi


@dataclass(frozen=True)
class DubinsPath:
    start: Pose
    word: str                          # 예: "LSR"
    lengths: tuple[float, float, float]  # 각 구간 길이 (mm)
    radius: float

    @property
    def length(self) -> float:
        return sum(self.lengths)


def dubins_paths(start: Pose, goal: Pose, radius: float) -> list[DubinsPath]:
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    d = math.hypot(dx, dy) / radius
    theta = math.atan2(dy, dx)
    a, b = _mod(start[2] - theta), _mod(goal[2] - theta)
    out = []
    for word, fn in _WORDS.items():
        tpq = fn(a, b, d)
        if tpq is not None:
            out.append(DubinsPath(start, word, tuple(v * radius for v in tpq), radius))
    return sorted(out, key=lambda p: p.length)


def sample(path: DubinsPath, step: float) -> list[Pose]:
    """시작 자세부터 끝 자세까지 step 이하 간격의 자세 목록."""
    pts = [path.start]
    x, y, h = path.start
    for kind, seg in zip(path.word, path.lengths):
        n = max(1, math.ceil(seg / step))
        x0, y0, h0 = x, y, h
        for k in range(1, n + 1):
            x, y, h = _advance(x0, y0, h0, kind, seg * k / n, path.radius)
            pts.append((x, y, h))
    return pts


def _advance(x, y, h, kind, s, r):
    if kind == "S":
        return x + s * math.cos(h), y + s * math.sin(h), h
    if kind == "L":
        h1 = h + s / r
        return x + r * (math.sin(h1) - math.sin(h)), y + r * (math.cos(h) - math.cos(h1)), h1
    h1 = h - s / r
    return x + r * (math.sin(h) - math.sin(h1)), y + r * (math.cos(h1) - math.cos(h)), h1


def _mod(v: float) -> float:
    return v % _TAU


def _lsl(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    p2 = 2 + d * d - 2 * math.cos(a - b) + 2 * d * (sa - sb)
    if p2 < 0:
        return None
    tmp = math.atan2(cb - ca, d + sa - sb)
    return _mod(-a + tmp), math.sqrt(p2), _mod(b - tmp)


def _rsr(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    p2 = 2 + d * d - 2 * math.cos(a - b) + 2 * d * (sb - sa)
    if p2 < 0:
        return None
    tmp = math.atan2(ca - cb, d - sa + sb)
    return _mod(a - tmp), math.sqrt(p2), _mod(-b + tmp)


def _lsr(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    p2 = -2 + d * d + 2 * math.cos(a - b) + 2 * d * (sa + sb)
    if p2 < 0:
        return None
    p = math.sqrt(p2)
    tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2, p)
    return _mod(-a + tmp), p, _mod(-b + tmp)


def _rsl(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    p2 = -2 + d * d + 2 * math.cos(a - b) - 2 * d * (sa + sb)
    if p2 < 0:
        return None
    p = math.sqrt(p2)
    tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2, p)
    return _mod(a - tmp), p, _mod(b - tmp)


def _rlr(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    tmp = (6 - d * d + 2 * math.cos(a - b) + 2 * d * (sa - sb)) / 8
    if abs(tmp) > 1:
        return None
    p = _mod(_TAU - math.acos(tmp))
    t = _mod(a - math.atan2(ca - cb, d - sa + sb) + p / 2)
    return t, p, _mod(a - b - t + p)


def _lrl(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    tmp = (6 - d * d + 2 * math.cos(a - b) + 2 * d * (sb - sa)) / 8
    if abs(tmp) > 1:
        return None
    p = _mod(_TAU - math.acos(tmp))
    t = _mod(-a + math.atan2(-ca + cb, d + sa - sb) + p / 2)
    return t, p, _mod(b - a - t + p)


_WORDS = {"LSL": _lsl, "RSR": _rsr, "LSR": _lsr, "RSL": _rsl, "RLR": _rlr, "LRL": _lrl}
