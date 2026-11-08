import math
from dataclasses import replace

import pytest

from fencebot.config import load_config
from fencebot.placement import plan_placement
from fencebot.types import Role, Zone

BOUNDARY = 1167  # 1·2차로 경계 (lane2 y_max)
TAPER = math.atan2(BOUNDARY, 2600)


@pytest.fixture
def cfg():
    return load_config()


def zone(x1=4500, y1=150, x2=5700, y2=1050):
    return Zone((x1, y1, x2, y2), ("lane2",), "damaged")


def by_role(cfg, placement):
    ids = {r.robot_id: r.role for r in cfg.robots}
    return {ids[t.robot_id]: t for t in placement.targets}


def test_basic_lane2_targets(cfg):
    p = plan_placement(zone(), cfg)
    assert p.ok and p.failures == ()
    t = by_role(cfg, p)
    assert (t[Role.POST2].x, t[Role.POST2].y) == (4200, BOUNDARY)   # 상류 끝 - 300
    assert (t[Role.PULLER].x, t[Role.PULLER].y) == (6000, BOUNDARY)  # 하류 끝 + 300
    assert (t[Role.POST3].x, t[Role.POST3].y) == (5100, BOUNDARY)   # 가운데
    assert (t[Role.RESERVE].x, t[Role.RESERVE].y) == (1600, 0)      # 기둥2 - 테이퍼, 갓길선
    assert (t[Role.POST1].x, t[Role.POST1].y) == (2900, BOUNDARY / 2)
    assert t[Role.PULLER].heading == 0 and t[Role.POST3].heading == 0
    assert t[Role.POST1].heading == pytest.approx(TAPER)
    assert t[Role.POST2].heading == pytest.approx(TAPER / 2)
    assert t[Role.RESERVE].heading == pytest.approx(TAPER + math.pi / 2)


def test_targets_follow_zone_position(cfg):
    a = plan_placement(zone(), cfg)
    b = plan_placement(zone(x1=5500, x2=6700), cfg)
    for ta, tb in zip(a.targets, b.targets):
        assert (tb.x - ta.x, tb.y - ta.y, tb.heading - ta.heading) == pytest.approx((1000, 0, 0))


def failure_codes(p):
    return {f.code for f in p.failures}


def test_long_zone_fails_gap_check(cfg):
    p = plan_placement(zone(x1=4000, x2=6500), cfg)  # 기둥2~3 간격 (2500+600)/2 > 1500
    assert not p.ok and p.targets == ()
    assert failure_codes(p) == {"gap"}


def test_net_length_check(cfg):
    small = replace(cfg, net=replace(cfg.net, length_mm=4000))
    p = plan_placement(zone(), small)  # 전체 길이 약 4650
    assert failure_codes(p) == {"net_length"}


def test_zone_outside_lane2_is_unsupported(cfg):
    assert failure_codes(plan_placement(zone(y2=1300), cfg)) == {"zone_lane"}  # 1차로 침범
    assert failure_codes(plan_placement(zone(y1=-100), cfg)) == {"zone_lane"}  # 갓길 침범


def test_target_outside_road_fails(cfg):
    p = plan_placement(zone(x1=2500, x2=3500), cfg)  # 보관 로봇 x = 2500-300-2600 < 0
    assert failure_codes(p) == {"out_of_road"}
    p = plan_placement(zone(x1=7500, x2=8600), cfg)  # 당김 로봇 x = 8900 > 9000 - 반지름
    assert failure_codes(p) == {"out_of_road"}


def test_all_failures_are_reported_together(cfg):
    p = plan_placement(zone(x1=1000, x2=3500, y2=1300), cfg)  # 전체 길이 5950은 통과
    assert failure_codes(p) == {"zone_lane", "gap", "out_of_road"}
    assert all(f.detail for f in p.failures)
