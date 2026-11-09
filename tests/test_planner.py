import math
from dataclasses import replace

import pytest

from fencebot.config import load_config
from fencebot.dubins import sample
from fencebot.planner import World, plan_deployment, plan_move
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
    # 경계선 위 줄 세우기 → 앞에서부터 전진 → 기둥1 → 보관
    assert order == [rid[Role.PULLER], rid[Role.POST3], rid[Role.POST2],
                     rid[Role.PULLER], rid[Role.POST3], rid[Role.POST2],
                     rid[Role.POST1], rid[Role.RESERVE]]
    final = {m.robot_id: m.goal for m in d.moves}
    assert {t.robot_id: (t.x, t.y, t.heading) for t in d.targets} == final
    for m in d.moves:
        end = sample(m.path, 20)[-1]
        assert math.dist(end[:2], m.goal[:2]) < 1e-3


def test_moves_stay_out_of_zone_and_road(cfg, scenario):
    d = plan_deployment(scenario.zone, vehicles_of(scenario), cfg.standby, cfg)
    x1, y1, x2, y2 = scenario.zone.bbox
    r = cfg.robot.radius_mm
    for m in d.moves:
        for x, y, _ in sample(m.path, 20):
            assert not (x1 < x < x2 and y1 < y < y2)
            assert -1000 + r <= y <= 2333 - r


def test_direct_puller_move_drags_net_over_zone(cfg, scenario):
    """계획 원안처럼 당김 로봇을 대기 배치에서 바로 목표로 보내면 그물이 사고 범위 위를 지난다."""
    rid = ids(cfg)[Role.PULLER]
    path, reason = plan_move(rid, (6000, BOUNDARY, 0.0), poses_of(cfg.standby), world_of(scenario), cfg)
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
    poses[rid[Role.PULLER]] = (3000, -500, 0.0)  # 기둥3 바로 앞을 막음
    path, reason = plan_move(rid[Role.POST3], (3500, -500, 0.0), poses, World(None, ()), cfg)
    assert path is None and "robot" in reason


def test_crossing_other_net_segment_is_detected(cfg):
    rid = ids(cfg)
    long_net = replace(cfg, net=replace(cfg.net, length_mm=100000))  # 길이 검사는 빼고 본다
    poses = poses_of(cfg.standby)
    poses[rid[Role.RESERVE]] = (2000, -500, 0.0)
    poses[rid[Role.POST2]] = (500, 500, 0.0)  # 기둥2~3 구간이 도로를 가로로 막음
    poses[rid[Role.POST3]] = (8500, 500, 0.0)
    poses[rid[Role.PULLER]] = (8500, 1500, 0.0)
    path, reason = plan_move(rid[Role.RESERVE], (4000, 1500, 0.0), poses, World(None, ()), long_net)
    assert path is None and "net_cross" in reason


def test_net_length_limit_applies_during_deployment(cfg, scenario):
    short = replace(cfg, net=replace(cfg.net, length_mm=6000))  # 최종 4650은 통과하지만 중간 자세가 넘음
    d = plan_deployment(scenario.zone, vehicles_of(scenario), short.standby, short)
    assert {f.code for f in d.failures} == {"path"}
    assert "net_length" in d.failures[0].detail


def test_placement_failure_skips_path_planning(cfg, scenario):
    long_zone = Zone((4000, 150, 6500, 1050), ("lane2",), "damaged")
    d = plan_deployment(long_zone, vehicles_of(scenario), cfg.standby, cfg)
    assert {f.code for f in d.failures} == {"gap"} and d.moves == ()
