"""전개 경로 계획 (배치_시뮬레이션_계획 4-1·4-2, ADR-010·011).

한 번에 한 대씩 움직이고, 나머지는 그 자리에 서 있다고 본다. 움직이는 로봇의 경로를 샘플마다 검사한다.
- 장애물 지도: 도로 밖(반지름만큼 안쪽까지), 사고 범위(부풀리지 않음, 중심만 금지),
  차량(반지름 + 여유), 다른 로봇(반지름 2배 + 여유)
- 그물(보관 → 걸린 기둥 → 당김, net.py): 전체 길이, 사고 범위·차량 위를 지나는 구간,
  걸린 로봇의 몸체가 다른 구간에 닿거나 자기 구간이 걸린 다른 로봇에 닿는 경우,
  기둥이 순서가 틀린 구간에 걸리는 경우(꼬임). 걸리지 않은 기둥은 그물 아래로 지나간다.
경로는 전진 Dubins이며, 가장 짧은 것부터 시험해 처음 통과하는 것을 쓴다.
한 번에 가는 경로가 모두 막히면 경유점 하나를 거치는 두 구간 경로를 찾는다 (ADR-013).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .config import Config
from .dubins import DubinsPath, Pose, dubins_paths, sample
from .net import chain, hook_events
from .placement import Failure, plan_placement
from .types import BBox, Role, RobotState, Target, Zone

VIA_GRID_MM = 300  # 경유점 후보 격자 간격
VIA_HEADINGS = (0.0, math.pi / 2, math.pi, -math.pi / 2)


@dataclass(frozen=True)
class World:
    zone: BBox | None
    vehicles: tuple[BBox, ...]


@dataclass(frozen=True)
class Move:
    robot_id: int
    goal: Pose
    paths: tuple[DubinsPath, ...]  # 이어 달리는 구간들. 경유점을 거치면 2개 (ADR-013)
    hooks: tuple[int, ...] = ()  # 이 이동 중 그물에 걸리는 기둥 로봇


@dataclass(frozen=True)
class Deployment:
    targets: tuple[Target, ...]  # 배치 검사와 경로 계획을 모두 통과했을 때만 채운다
    moves: tuple[Move, ...]      # 실행 순서
    failures: tuple[Failure, ...]
    warnings: tuple[Failure, ...] = ()  # 완화 모드에서 넘어간 검사 (ADR-014)

    @property
    def ok(self) -> bool:
        return not self.failures


def plan_deployment(zone: Zone, vehicles: tuple[BBox, ...], start: tuple[RobotState, ...],
                    config: Config) -> Deployment:
    strict = config.planning.strict
    placement = plan_placement(zone, config, strict)
    if strict and not placement.ok:
        return Deployment((), (), placement.failures)
    warnings = list(placement.failures)
    if not strict:  # 여유 거리와 그물 길이는 보지 않는다
        config = replace(config, planning=replace(config.planning, clearance_mm=0.0),
                         net=replace(config.net, length_mm=math.inf))
    world = World(zone.bbox, tuple(vehicles))
    roles = {r.robot_id: r.role.value for r in config.robots}
    poses = {s.robot_id: (s.x, s.y, s.heading) for s in start}
    hooked: set[int] = set()  # 기둥 로봇은 그물과 떨어져 출발한다 (ADR-011)
    moves = []
    for rid, goal in _sequence(placement.targets, config):
        paths, reason, hooks = plan_route(rid, goal, poses, hooked, world, config)
        if paths is None:
            failure = Failure("path", f"{roles[rid]} -> ({goal[0]:.0f}, {goal[1]:.0f}): {reason}")
            if strict:
                return Deployment((), (), (failure,))
            warnings.append(failure)  # 막혀도 가장 짧은 경로로 보낸다
            paths = (dubins_paths(poses[rid], goal, config.robot.min_turn_radius_mm)[0],)
        moves.append(Move(rid, goal, paths, hooks))
        poses[rid] = goal
        hooked |= set(hooks)
    loose = [roles[r.robot_id] for r in config.robots[1:-1] if r.robot_id not in hooked]
    if loose:
        failure = Failure("path", f"not hooked: {', '.join(loose)}")
        if strict:
            return Deployment((), (), (failure,))
        warnings.append(failure)
    return Deployment(placement.targets, tuple(moves), (), tuple(warnings))


def _sequence(targets: tuple[Target, ...], config: Config) -> list[tuple[int, Pose]]:
    """당김이 경계선 위 기둥2 목표까지 그물을 펴면, 기둥3·기둥2가 들어오며 그물을 걸고 그 뒤에 줄 선다.
    보관 로봇이 자리를 잡은 뒤 앞에서부터 목표까지 전진하고, 마지막으로 기둥1이 들어와 걸어 테이퍼를 만든다.

    당김이 한 번에 목표까지 가면 보관-당김 그물선이 사고 범위를 지나기 때문이다 (ADR-010·011).
    보관 로봇을 전진 전에 보내야 그물 6m 안에 든다 (ADR-012). 기둥1은 보관 로봇이 자리 잡은 뒤에 와야
    목표에서 보관-기둥2 그물선에 닿아 걸린다.
    """
    rid = {r.role: r.robot_id for r in config.robots}
    t = {r.role: (tg.x, tg.y, tg.heading) for r in config.robots for tg in targets
         if tg.robot_id == r.robot_id}
    x0, yb, _ = t[Role.POST2]
    gap = config.planning.stage_gap_mm
    line = [Role.PULLER, Role.POST3, Role.POST2]
    stages = [(rid[role], (x0 - k * gap, yb, 0.0)) for k, role in enumerate(line)]
    reserve = [(rid[Role.RESERVE], t[Role.RESERVE])]
    advance = [(rid[role], t[role]) for role in line]
    post1 = [(rid[Role.POST1], t[Role.POST1])]
    return stages + reserve + advance + post1


def plan_route(robot_id: int, goal: Pose, poses: dict[int, Pose], hooked: set[int], world: World,
               config: Config) -> tuple[tuple[DubinsPath, ...] | None, str, tuple[int, ...]]:
    """한 번에 가는 경로를 먼저 찾고, 없으면 경유점 하나를 거치는 가장 짧은 두 구간 경로를 찾는다.

    실패하면 (None, 한 번에 가는 경로가 막힌 이유, ()).
    """
    path, reason, hooks = plan_move(robot_id, goal, poses, hooked, world, config)
    if path is not None:
        return (path,), "", hooks
    start = poses[robot_id]
    vias = [(math.dist(start[:2], v[:2]) + math.dist(v[:2], goal[:2]), v)
            for v in _via_candidates(config) if not violation(robot_id, v, poses, hooked, world, config)]
    best, best_len = None, math.inf
    for bound, via in sorted(vias):
        if bound >= best_len:
            break  # 남은 후보는 직선거리만으로도 더 길다
        first, _, hooks1 = plan_move(robot_id, via, poses, hooked, world, config)
        if first is None or first.length >= best_len:
            continue
        second, _, hooks2 = plan_move(robot_id, goal, {**poses, robot_id: via}, hooked | set(hooks1),
                                      world, config)
        if second is not None and first.length + second.length < best_len:
            best, best_len = ((first, second), "", hooks1 + hooks2), first.length + second.length
    return best or (None, reason, ())


def _via_candidates(config: Config) -> list[Pose]:
    r, road = config.robot.radius_mm, config.road
    y_lo = min(a for a, _ in road.lanes.values()) + r
    y_hi = max(b for _, b in road.lanes.values()) - r
    xs = _grid(road.x_min_mm + r, road.x_max_mm - r)
    return [(x, y, h) for x in xs for y in _grid(y_lo, y_hi) for h in VIA_HEADINGS]


def _grid(lo: float, hi: float) -> list[float]:
    n = int((hi - lo) // VIA_GRID_MM)
    return [lo + k * VIA_GRID_MM for k in range(n + 1)]


def plan_move(robot_id: int, goal: Pose, poses: dict[int, Pose], hooked: set[int], world: World,
              config: Config) -> tuple[DubinsPath | None, str, tuple[int, ...]]:
    """(통과하는 경로, "", 이동 중 걸리는 기둥) 또는 (None, 가장 짧은 경로가 처음 걸린 이유, ())."""
    first_reason = "no_path"
    for i, path in enumerate(dubins_paths(poses[robot_id], goal, config.robot.min_turn_radius_mm)):
        reason, hooks = _follow(robot_id, sample(path, config.planning.step_mm), poses, hooked,
                                world, config)
        if not reason:
            return path, "", hooks
        if i == 0:
            first_reason = reason
    return None, first_reason, ()


def _follow(robot_id, pts, poses, hooked, world, config) -> tuple[str, tuple[int, ...]]:
    """경로를 따라가며 걸리는 기둥을 반영해 검사한다. (어긴 조건, 걸린 기둥 순서)."""
    order = [s.robot_id for s in config.robots]
    roles = {s.robot_id: s.role.value for s in config.robots}
    state, hooks = set(hooked), []
    prev = {rid: p[:2] for rid, p in poses.items()}
    for pose in pts:
        now = dict(prev)
        now[robot_id] = pose[:2]
        for post, a, b, ok in hook_events(order, state, prev, now, config.net.hook_tol_mm):
            if not ok:
                return f"net_tangle ({roles[post]} into {roles[a]}-{roles[b]})", ()
            state.add(post)
            hooks.append(post)
        if v := violation(robot_id, pose, poses, state, world, config):
            return v, ()
        prev = now
    return "", tuple(hooks)


def violation(robot_id: int, pose: Pose, poses: dict[int, Pose], hooked: set[int], world: World,
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
    on_net = chain(order, hooked)
    segs = list(zip(on_net, on_net[1:]))
    total = sum(math.dist(pts[a], pts[b]) for a, b in segs)
    if total > config.net.length_mm:
        return f"net_length {total:.0f} > {config.net.length_mm:.0f}"
    for a, b in segs:
        if world.zone and _segment_hits_rect(pts[a], pts[b], world.zone):
            return f"net_over zone ({roles[a]}-{roles[b]})"
        for k, box in enumerate(world.vehicles):
            if _segment_hits_rect(pts[a], pts[b], box):
                return f"net_over vehicle {k} ({roles[a]}-{roles[b]})"
    if robot_id not in on_net:
        return ""  # 걸리지 않은 기둥은 그물 아래로 지나간다 (선에 닿으면 _follow에서 걸림)
    for a, b in segs:
        mine = robot_id in (a, b)
        # 내 구간은 걸린 다른 로봇에, 남의 구간은 내 몸체에 닿으면 안 된다
        others = [rid for rid in on_net if rid not in (a, b)] if mine else [robot_id]
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
