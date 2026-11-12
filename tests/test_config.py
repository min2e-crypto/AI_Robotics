from fencebot.config import load_config
from fencebot.types import Role


def test_default_config_loads_required_values():
    cfg = load_config()
    assert cfg.net.length_mm == 6000
    assert cfg.net.max_gap_mm == 1500
    assert cfg.robot.min_turn_radius_mm == 590
    assert cfg.robot.wheelbase_mm == 280
    assert set(cfg.road.lanes) == {"shoulder", "lane2", "lane1"}
    assert cfg.arrival.position_mm > 0 and cfg.arrival.settle_s > 0
    assert set(cfg.margins) == {"damaged", "overturned", "fire"}


def test_standby_layout_has_one_robot_per_role_in_net_order():
    cfg = load_config()
    roles = [r.role for r in cfg.robots]
    assert roles == [Role.RESERVE, Role.POST1, Role.POST2, Role.POST3, Role.PULLER]
    assert len({r.robot_id for r in cfg.robots}) == 5
    assert {r.robot_id for r in cfg.robots} == {p.robot_id for p in cfg.standby}


def test_lane_of_y():
    cfg = load_config()
    assert cfg.road.lane_at(-100) == "shoulder"
    assert cfg.road.lane_at(500) == "lane2"
    assert cfg.road.lane_at(1500) == "lane1"
    assert cfg.road.lane_at(99999) is None


def test_config_file_overrides_only_given_values(tmp_path):
    p = tmp_path / "video.toml"
    p.write_text('[camera]\nimage_points_px = [[1, 2], [3, 4], [5, 6], [7, 8]]\n\n[incident]\nconfirm_frames = 2\n',
                 encoding="utf-8")
    base, cfg = load_config(), load_config(p)
    assert cfg.camera.image_points_px == ((1, 2), (3, 4), (5, 6), (7, 8))
    assert cfg.camera.road_points_mm == base.camera.road_points_mm  # 주지 않은 값은 기본 설정
    assert cfg.incident.confirm_frames == 2 and cfg.incident.near_mm == base.incident.near_mm
    assert cfg.net == base.net and cfg.robots == base.robots
