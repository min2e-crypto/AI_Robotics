"""그물 모양과 기둥 로봇 걸기 판정 (ADR-011). 시뮬레이터와 경로 계획이 같이 쓴다.

그물은 보관 로봇(릴)과 당김 로봇(클램프)에만 묶여 있다. 기둥 로봇은 처음에는 떨어져 있다가,
기둥(링 클립)이 그물선을 가로지르거나 그물선이 기둥 위를 쓸고 지나가면 걸린다.
걸린 뒤로는 그물이 그 기둥을 거쳐 꺾이고, 기둥 로봇 몸체는 그물보다 낮아 걸리기 전에는 아래로 지나간다.
"""

from __future__ import annotations

import math

Point = tuple[float, float]


def chain(order: list[int], hooked: set[int]) -> list[int]:
    """그물이 지나는 로봇 순서: 보관 → 걸린 기둥들(꿰인 순서) → 당김."""
    return [order[0], *(r for r in order[1:-1] if r in hooked), order[-1]]


def net_length(order: list[int], hooked: set[int], pts: dict[int, Point]) -> float:
    ids = chain(order, hooked)
    return sum(math.dist(pts[a], pts[b]) for a, b in zip(ids, ids[1:]))


def hook_events(order: list[int], hooked: set[int], prev: dict[int, Point],
                now: dict[int, Point], tol: float) -> list[tuple[int, int, int, bool]]:
    """prev → now 사이에 그물에 걸린 기둥: (기둥, 구간 앞 로봇, 구간 뒤 로봇, 순서가 맞는지).

    지금 위치가 구간 안쪽(투영이 양 끝 사이)이고, 선에 tol 이내로 닿았거나 선의 반대편으로 넘어가면 걸린다.
    순서가 맞지 않으면(구간 양 끝 사이에 꿰일 기둥이 아니면) 그물이 꼬인다.
    """
    ids = chain(order, hooked)
    rank = {r: i for i, r in enumerate(order)}
    out = []
    for p in order[1:-1]:
        if p in hooked:
            continue
        for a, b in zip(ids, ids[1:]):
            t, dist = _project(now[p], now[a], now[b])
            if not 0.0 <= t <= 1.0:
                continue
            crossed = _side(prev[p], prev[a], prev[b]) * _side(now[p], now[a], now[b]) < 0
            if dist <= tol or crossed:
                out.append((p, a, b, rank[a] < rank[p] < rank[b]))
                break
    return out


def _side(p: Point, a: Point, b: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _project(p: Point, a: Point, b: Point) -> tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = dx * dx + dy * dy
    t = 0.0 if n == 0 else ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / n
    return t, math.dist(p, (a[0] + t * dx, a[1] + t * dy))
