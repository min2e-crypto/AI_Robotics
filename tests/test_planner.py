import math
from dataclasses import replace

import pytest

from fencebot.config import load_config
from fencebot.dubins import sample
from fencebot.planner import World, plan_deployment, plan_move, plan_route
from fencebot.scenario import load_scenario
from fencebot.types import Role, Zone

BOUNDARY = 1167


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def scenario():
    return load_scenario("scenarios/lane2_basic.toml")


def world_of(scenario):
    return World(scenario.zone.bbox, tuple(v.bbox for v in scenario.vehicles))


def vehicles_of(scenario):
    return tuple(v.bbox for v in scenario.vehicles)


def ids(cfg):
    return {r.role: r.robot_id for r in cfg.robots}


def poses_of(states):
    return {s.robot_id: (s.x, s.y, s.heading) for s in states}


def test_basic_scenario_deploys(cfg, scenario):
    d = plan_deployment(scenario.zone, vehicles_of(scenario), cfg.standby, cfg)
    assert d.ok, d.failures
    rid = ids(cfg)
    order = [m.robot_id for m in d.moves]
    # 당김이 먼저 그물을 펴고 → 기둥3·2가 들어오며 걸고 줄 섬 → 보관 → 앞에서부터 전진 → 기둥1이 걸어 테이퍼
    assert order == [rid[Role.PULLER], rid[Role.POST3], rid[Role.POST2], rid[Role.RESERVE],
                     rid[Role.PULLER], rid[Role.POST3], rid[Role.POST2], rid[Role.POST1]]
    assert [m.hooks for m in d.moves] == [(), (rid[Role.POST3],), (rid[Role.POST2],),
                                          (), (), (), (), (rid[Role.POST1],)]
    final = {m.robot_id: m.goal for m in d.moves}
    assert {t.robot_id: (t.x, t.y, t.heading) for t in d.targets} == final
    for m in d.moves:
        assert len(m.paths) == 1  # 기본 시나리오는 경유점 없이 간다
        end = sample(m.paths[-1], 20)[-1]
        assert math.dist(end[:2], m.goal[:2]) < 1e-3


def test_moves_stay_out_of_zone_and_road(cfg, scenario):
    d = plan_deployment(scenario.zone, vehicles_of(scenario), cfg.standby, cfg)
    x1, y1, x2, y2 = scenario.zone.bbox
    r = cfg.robot.radius_mm
    for m in d.moves:
        for path in m.paths:
            for x, y, _ in sample(path, 20):
                assert not (x1 < x < x2 and y1 < y < y2)
                assert -1000 + r <= y <= 2333 - r


def test_direct_puller_move_drags_net_over_zone(cfg, scenario):
    """계획 원안처럼 당김 로봇을 대기 배치에서 바로 목표로 보내면 그물이 사고 범위 위를 지난다."""
    rid = ids(cfg)[Role.PULLER]
    path, reason, _ = plan_move(rid, (6000, BOUNDARY, 0.0), poses_of(cfg.standby), set(),
                                world_of(scenario), cfg)
    assert path is None and "zone" in reason


def test_vehicle_blocks_path(cfg, scenario):
    blocker = ((3000, 900, 3600, 1400),)  # 줄 세우는 자리를 막는 차량
    d = plan_deployment(scenario.zone, blocker, cfg.standby, cfg)
    assert not d.ok and d.moves == () and d.targets == ()
    assert {f.code for f in d.failures} == {"path"}
    assert "vehicle" in d.failures[0].detail


def test_robot_collision_is_detected(cfg, scenario):
    poses = poses_of(cfg.standby)
    rid = ids(cfg)
    poses[rid[Role.PULLER]] = (2700, -760, 0.0)  # 기둥3 바로 앞을 막음
    path, reason, _ = plan_move(rid[Role.POST3], (3500, -760, 0.0), poses, set(), World(None, ()), cfg)
    assert path is None and "robot" in reason


def test_crossing_other_net_segment_is_detected(cfg):
    rid = ids(cfg)
    long_net = replace(cfg, net=replace(cfg.net, length_mm=100000))  # 길이 검사는 빼고 본다
    poses = poses_of(cfg.standby)
    poses[rid[Role.RESERVE]] = (2000, -500, 0.0)
    poses[rid[Role.POST2]] = (500, 500, 0.0)  # 기둥2~3 구간이 도로를 가로로 막음
    poses[rid[Role.POST3]] = (8500, 500, 0.0)
    poses[rid[Role.PULLER]] = (8500, 1500, 0.0)
    hooked = {rid[Role.POST2], rid[Role.POST3]}
    path, reason, _ = plan_move(rid[Role.RESERVE], (4000, 1500, 0.0), poses, hooked, World(None, ()),
                                long_net)
    assert path is None and "net_cross" in reason


def test_hooking_wrong_segment_is_tangle(cfg):
    rid = ids(cfg)
    poses = poses_of(cfg.standby)
    poses[rid[Role.POST3]] = (3000, 800, 0.0)    # 기둥3만 걸림: 보관-기둥3, 기둥3-당김 구간
    poses[rid[Role.PULLER]] = (6000, -250, 0.0)
    poses[rid[Role.POST2]] = (8500, -760, 0.0)   # 길을 비켜 둠
    # 기둥1이 기둥3-당김 구간을 가로지르면 순서가 틀린 걸림
    path, reason, _ = plan_move(rid[Role.POST1], (4500, 600, 0.0), poses, {rid[Role.POST3]},
                                World(None, ()), cfg)
    assert path is None and "net_tangle" in reason


def test_net_length_limit_applies_during_deployment(cfg, scenario):
    short = replace(cfg, net=replace(cfg.net, length_mm=5700))  # 최종 5610은 통과하지만 전진 중(약 5.78m)에 넘음
    d = plan_deployment(scenario.zone, vehicles_of(scenario), short.standby, short)
    assert {f.code for f in d.failures} == {"path"}
    assert "net_length" in d.failures[0].detail


def test_placement_failure_skips_path_planning(cfg, scenario):
    long_zone = Zone((4000, 150, 6500, 1050), ("lane2",), "damaged")
    d = plan_deployment(long_zone, vehicles_of(scenario), cfg.standby, cfg)
    assert {f.code for f in d.failures} == {"gap"} and d.moves == ()


def shifted(x):
    """기본 시나리오와 같은 크기의 사고를 범위 상류 끝 x로 옮긴 것."""
    zone = Zone((x, 134, x + 2160, 1034), ("lane2",), "damaged")
    return zone, ((x + 300, 274, x + 1860, 894),)


def test_waypoint_takes_post1_around_reserve(cfg):
    """사고가 하류에 있으면 기둥1의 Dubins 경로가 모두 보관 로봇에 막힌다. 경유점 하나로 돌아간다."""
    zone, vehicles = shifted(5000)
    d = plan_deployment(zone, vehicles, cfg.standby, cfg)
    assert d.ok, d.failures
    rid = ids(cfg)
    last = d.moves[-1]
    assert last.robot_id == rid[Role.POST1] and last.hooks == (rid[Role.POST1],)
    assert len(last.paths) == 2
    via, start2 = sample(last.paths[0], 20)[-1], last.paths[1].start
    assert math.dist(via[:2], start2[:2]) < 1e-3 and abs(via[2] - start2[2]) < 1e-6
    assert math.dist(sample(last.paths[1], 20)[-1][:2], last.goal[:2]) < 1e-3


def test_route_prefers_direct_path(cfg):
    rid = ids(cfg)[Role.POST3]
    paths, reason, _ = plan_route(rid, (3500, -760, 0.0), poses_of(cfg.standby), set(),
                                  World(None, ()), cfg)
    assert reason == "" and len(paths) == 1


def test_route_fails_with_direct_reason_when_no_waypoint_helps(cfg, scenario):
    blocker = ((3000, 900, 3600, 1400),)  # 줄 세우는 자리(목표) 자체를 막는 차량
    d = plan_deployment(scenario.zone, blocker, cfg.standby, cfg)
    assert "vehicle" in d.failures[0].detail


@pytest.mark.parametrize("x", [3200, 4400, 5600, 6200])
def test_deploys_across_accident_positions(cfg, x):
    """바닥 안에서 배치할 수 있는 사고 위치(범위 상류 끝 약 3200~6300)는 모두 전개된다."""
    d = plan_deployment(*shifted(x), cfg.standby, cfg)
    assert d.ok, d.failures


def relaxed(cfg):
    return replace(cfg, planning=replace(cfg.planning, strict=False))


def test_relaxed_mode_ignores_clearance(cfg):
    """발표용(strict = false): 사고 차량이 경계선 쪽으로 치우쳐도 여유 거리 없이 전개한다 (ADR-014)."""
    zone = Zone((3800, 174, 5960, 1074), ("lane2",), "damaged")
    vehicles = ((4100, 314, 5660, 934),)
    assert not plan_deployment(zone, vehicles, cfg.standby, cfg).ok
    d = plan_deployment(zone, vehicles, cfg.standby, relaxed(cfg))
    assert d.ok and len(d.moves) == 8 and d.warnings == ()


def test_relaxed_mode_turns_failures_into_warnings(cfg, scenario):
    blocker = ((3000, 900, 3600, 1400),)  # 줄 세우는 자리를 막는 차량
    d = plan_deployment(scenario.zone, blocker, cfg.standby, relaxed(cfg))
    assert d.ok and len(d.moves) == 8 and len(d.targets) == 5
    assert "path" in {w.code for w in d.warnings}

    long_zone = Zone((3800, 150, 6300, 1050), ("lane2",), "damaged")
    d = plan_deployment(long_zone, (), cfg.standby, relaxed(cfg))
    assert d.ok and len(d.targets) == 5
    assert "gap" in {w.code for w in d.warnings}
