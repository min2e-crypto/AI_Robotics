"""배치 계산 (구현_계획 7절, ADR-009). Zone + 차선 → 로봇 5대의 Target, 배치 전 검사.

형상: 갓길선(lane2 y_min) 위의 보관 로봇에서 테이퍼로 올라가 1·2차로 경계(lane2 y_max)를 따라 막는다.
    보관 ─ 기둥1 ─ 기둥2 ─── 기둥3 ─── 당김
          (테이퍼)   └ 범위 상류 끝 - 여유      └ 범위 하류 끝 + 여유
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Config
from .types import Role, Target, Zone

SUPPORTED_LANE = "lane2"  # 임시 전제: 2차로 사고만 지원 (구현_계획 3절)


@dataclass(frozen=True)
class Failure:
    code: str    # zone_lane / gap / net_length / out_of_road
    detail: str  # 관제자에게 보여 줄 설명


@dataclass(frozen=True)
class Placement:
    targets: tuple[Target, ...]  # 검사를 통과했을 때만 채운다. 그물 순서
    failures: tuple[Failure, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def plan_placement(zone: Zone, config: Config) -> Placement:
    shoulder_y, boundary_y = config.road.lanes[SUPPORTED_LANE]
    x_min, _, x_max, _ = zone.bbox
    post2 = (x_min - config.placement.upstream_margin_mm, boundary_y)
    puller = (x_max + config.placement.downstream_margin_mm, boundary_y)
    post3 = _mid(post2, puller)
    reserve = (post2[0] - config.net.taper_length_mm, shoulder_y)
    post1 = _mid(reserve, post2)
    taper = math.atan2(post2[1] - reserve[1], post2[0] - reserve[0])
    # 기둥 로봇은 그물 방향(하류 쪽), 기둥2는 꺾이는 곳이라 양쪽 방향의 가운데,
    # 보관 로봇은 그물과 직각인 갓길 쪽 (도로 쪽은 대기 배치에서 갈 경로가 없음, ADR-010)
    poses = {
        Role.RESERVE: (*reserve, taper - math.pi / 2),
        Role.POST1: (*post1, taper),
        Role.POST2: (*post2, taper / 2),
        Role.POST3: (*post3, 0.0),
        Role.PULLER: (*puller, 0.0),
    }
    targets = tuple(Target(r.robot_id, *poses[r.role]) for r in config.robots)
    failures = _check(zone, targets, config)
    return Placement(() if failures else targets, failures)


def _check(zone: Zone, targets: tuple[Target, ...], config: Config) -> tuple[Failure, ...]:
    out = []
    lo, hi = config.road.lanes[SUPPORTED_LANE]
    _, y1, _, y2 = zone.bbox
    if y1 < lo or y2 > hi:
        out.append(Failure("zone_lane", f"범위 y {y1:.0f}~{y2:.0f}가 {SUPPORTED_LANE}({lo:.0f}~{hi:.0f})를 벗어남"))

    roles = {r.robot_id: r.role.value for r in config.robots}
    gaps = [(roles[a.robot_id], roles[b.robot_id], math.dist((a.x, a.y), (b.x, b.y)))
            for a, b in zip(targets, targets[1:])]
    over = [f"{a}-{b} {d:.0f}" for a, b, d in gaps if d > config.net.max_gap_mm]
    if over:
        out.append(Failure("gap", f"구간 간격 최대 {config.net.max_gap_mm:.0f} 초과: {', '.join(over)}"))

    total = sum(d for _, _, d in gaps)
    if total > config.net.length_mm:
        out.append(Failure("net_length", f"전체 길이 {total:.0f} > 그물 {config.net.length_mm:.0f}"))

    # 카메라 시야는 도로 범위와 같다고 본다 (H1에서 실제 시야로 교체)
    r, road = config.robot.radius_mm, config.road
    y_lo = min(a for a, _ in road.lanes.values())
    y_hi = max(b for _, b in road.lanes.values())
    outside = [f"{roles[t.robot_id]}({t.x:.0f},{t.y:.0f})" for t in targets
               if not (road.x_min_mm + r <= t.x <= road.x_max_mm - r and y_lo + r <= t.y <= y_hi - r)]
    if outside:
        out.append(Failure("out_of_road", f"도로 밖 목표: {', '.join(outside)}"))
    return tuple(out)


def _mid(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
