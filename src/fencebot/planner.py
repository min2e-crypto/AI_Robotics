"""전개 경로 계획 (배치_시뮬레이션_계획 4-1·4-2, ADR-010).

한 번에 한 대씩 움직이고, 나머지는 그 자리에 서 있다고 본다. 움직이는 로봇의 경로를 샘플마다 검사한다.
- 장애물 지도: 도로 밖(반지름만큼 안쪽까지), 사고 범위(부풀리지 않음, 중심만 금지),
  차량(반지름 + 여유), 다른 로봇(반지름 2배 + 여유)
- 그물: 전체 길이, 사고 범위·차량 위를 지나는 구간, 몸체가 다른 구간에 닿거나 자기 구간이 다른 로봇에 닿는 경우
경로는 전진 Dubins이며, 가장 짧은 것부터 시험해 처음 통과하는 것을 쓴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Config
from .dubins import DubinsPath, Pose, dubins_paths, sample
from .placement import Failure, plan_placement
from .types import BBox, Role, RobotState, Target, Zone


@dataclass(frozen=True)
class World:
    zone: BBox | None
    vehicles: tuple[BBox, ...]


@dataclass(frozen=True)
class Move:
    robot_id: int
    goal: Pose
    path: DubinsPath


@dataclass(frozen=True)
class Deployment:
    targets: tuple[Target, ...]  # 배치 검사와 경로 계획을 모두 통과했을 때만 채운다
    moves: tuple[Move, ...]      # 실행 순서
    failures: tuple[Failure, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def plan_deployment(zone: Zone, vehicles: tuple[BBox, ...], start: tuple[RobotState, ...],
                    config: Config) -> Deployment:
    placement = plan_placement(zone, config)
    if not placement.ok:
        return Deployment((), (), placement.failures)
    world = World(zone.bbox, tuple(vehicles))
    roles = {r.robot_id: r.role.value for r in config.robots}
    poses = {s.robot_id: (s.x, s.y, s.heading) for s in start}
    moves = []
    for rid, goal in _sequence(placement.targets, config):
        path, reason = plan_move(rid, goal, poses, world, config)
        if path is None:
            detail = f"{roles[rid]} -> ({goal[0]:.0f}, {goal[1]:.0f}): {reason}"
            return Deployment((), (), (Failure("path", detail),))
        moves.append(Move(rid, goal, path))
        poses[rid] = goal
    return Deployment(placement.targets, tuple(moves), ())


def _sequence(targets: tuple[Target, ...], config: Config) -> list[tuple[int, Pose]]:
    """경계선 위 기둥2 목표에서 상류로 당김·기둥3·기둥2를 줄 세운 뒤 앞에서부터 전진한다.

    한 대씩 바로 목표로 보내면 그물이 사고 범위를 가로지르기 때문이다 (ADR-010).
    """
    rid = {r.role: r.robot_id for r in config.robots}
    t = {r.role: (tg.x, tg.y, tg.heading) for r in config.robots for tg in targets
         if tg.robot_id == r.robot_id}
    x0, yb, _ = t[Role.POST2]
    gap = config.planning.stage_gap_mm
    line = [Role.PULLER, Role.POST3, Role.POST2]
    stages = [(rid[role], (x0 - k * gap, yb, 0.0)) for k, role in enumerate(line)]
    advance = [(rid[role], t[role]) for role in line]
    rest = [(rid[role], t[role]) for role in (Role.POST1, Role.RESERVE)]
    return stages + advance + rest


def plan_move(robot_id: int, goal: Pose, poses: dict[int, Pose], world: World,
              config: Config) -> tuple[DubinsPath | None, str]:
    """통과하는 경로와 빈 문자열, 또는 None과 가장 짧은 경로가 처음 걸린 이유."""
    first_reason = "no_path"
    for i, path in enumerate(dubins_paths(poses[robot_id], goal, config.robot.min_turn_radius_mm)):
        reason = next((v for p in sample(path, config.planning.step_mm)
                       if (v := violation(robot_id, p, poses, world, config))), "")
        if not reason:
            return path, ""
        if i == 0:
            first_reason = reason
    return None, first_reason


def violation(robot_id: int, pose: Pose, poses: dict[int, Pose], world: World,
              config: Config) -> str:
    """robot_id가 pose에 있고 나머지는 poses에 있을 때 어긴 조건. 없으면 빈 문자열."""
    x, y = pose[0], pose[1]
    r, c, road = config.robot.radius_mm, config.planning.clearance_mm, config.road
    y_lo = min(a for a, _ in road.lanes.values())
    y_hi = max(b for _, b in road.lanes.values())
    if not (road.x_min_mm + r <= x <= road.x_max_mm - r and y_lo + r <= y <= y_hi - r):
        return "road"
    if world.zone and _inside(x, y, world.zone):
        return "zone"
    for k, box in enumerate(world.vehicles):
        if _dist_rect(x, y, box) < r + c:
            return f"vehicle {k}"
    roles = {s.robot_id: s.role.value for s in config.robots}
    for rid, p in poses.items():
        if rid != robot_id and math.dist((x, y), p[:2]) < 2 * r + c:
            return f"robot {roles[rid]}"

    order = [s.robot_id for s in config.robots]
    pts = {rid: (pose if rid == robot_id else poses[rid])[:2] for rid in order}
    segs = [(a, b) for a, b in zip(order, order[1:])]
    total = sum(math.dist(pts[a], pts[b]) for a, b in segs)
    if total > config.net.length_mm:
        return f"net_length {total:.0f} > {config.net.length_mm:.0f}"
    for a, b in segs:
        if world.zone and _segment_hits_rect(pts[a], pts[b], world.zone):
            return f"net_over zone ({roles[a]}-{roles[b]})"
        for k, box in enumerate(world.vehicles):
            if _segment_hits_rect(pts[a], pts[b], box):
                return f"net_over vehicle {k} ({roles[a]}-{roles[b]})"
    for a, b in segs:
        mine = robot_id in (a, b)
        # 내 구간은 다른 로봇에, 남의 구간은 내 몸체에 닿으면 안 된다
        others = [rid for rid in order if rid not in (a, b)] if mine else [robot_id]
        for rid in others:
            if _dist_segment(pts[rid], pts[a], pts[b]) < r:
                return f"net_cross ({roles[a]}-{roles[b]}, {roles[rid]})"
    return ""


def _inside(x: float, y: float, box: BBox) -> bool:
    return box[0] < x < box[2] and box[1] < y < box[3]


def _dist_rect(x: float, y: float, box: BBox) -> float:
    return math.hypot(max(box[0] - x, 0.0, x - box[2]), max(box[1] - y, 0.0, y - box[3]))


def _dist_segment(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = dx * dx + dy * dy
    s = 0.0 if n == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / n))
    return math.dist(p, (a[0] + s * dx, a[1] + s * dy))


def _segment_hits_rect(a, b, box: BBox, edge: float = 1.0) -> bool:
    """선분이 박스 내부를 지나는지 (Liang-Barsky). 테두리를 따라가는 것은 허용한다."""
    x1, y1, x2, y2 = box[0] + edge, box[1] + edge, box[2] - edge, box[3] - edge
    if x1 >= x2 or y1 >= y2:
        return False
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x1), (dx, x2 - a[0]), (-dy, a[1] - y1), (dy, y2 - a[1])):
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True
