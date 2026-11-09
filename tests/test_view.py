import math
from pathlib import Path

import pytest

from fencebot.config import load_config
from fencebot.io import ActionCommand, DriveCommand
from fencebot.log import LogWriter, read_log
from fencebot.runner import Runner
from fencebot.scenario import load_scenario
from fencebot.sim import SimBackend
from fencebot.view.control import ManualControl, Playback
from fencebot.view.scene import ScreenTransform, build_scene, link_status

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def scenario():
    return load_scenario(ROOT / "scenarios" / "lane2_basic.toml")


def test_transform_flips_y_and_keeps_scale(cfg):
    tf = ScreenTransform.for_road(cfg.road, width_px=1000, margin_px=20)
    left_top = tf.to_px(cfg.road.x_min_mm, 2333)
    assert left_top == pytest.approx((20, 20))
    # 1차로 쪽(+y)이 화면 위, 갓길(-y)이 아래
    assert tf.to_px(0, 0)[1] > tf.to_px(0, 1167)[1]
    assert tf.length(9000) == pytest.approx(1000 - 40)
    assert tf.height_px == pytest.approx(40 + (2333 + 1000) * tf.scale)


def test_scene_from_live_record_equals_scene_from_log(tmp_path, cfg, scenario):
    log_path = tmp_path / "run.jsonl"
    with LogWriter(log_path) as log:
        runner = Runner(scenario, cfg, SimBackend(cfg, seed=1), log)
        live = [build_scene(runner.tick(), log.records, cfg, scenario) for _ in range(25)]
    records = read_log(log_path)
    ticks = [r for r in records if r["type"] == "tick"]
    replayed = [build_scene(tk, records, cfg, scenario) for tk in ticks[:25]]
    assert live == replayed


def test_scene_contents(tmp_path, cfg, scenario):
    with LogWriter(tmp_path / "run.jsonl") as log:
        runner = Runner(scenario, cfg, SimBackend(cfg), log)
        scene = build_scene(runner.tick(), log.records, cfg, scenario)
    assert [r.role for r in scene.robots] == [r.role.value for r in cfg.robots]
    assert scene.net_chain == (0, 4)  # 기둥은 그물과 떨어져 출발 (ADR-011)
    assert len(scene.segments) == 1
    assert sum(scene.segments) == pytest.approx(scene.net_length)
    assert scene.slack == pytest.approx(scene.unwound - scene.net_length)
    assert scene.zone == scenario.zone.bbox
    assert [v[0] for v in scene.vehicles] == ["damaged", "normal"]
    assert len(scene.targets) == 5 and scene.placement_failures == ()
    assert len(scene.paths) == 8  # 줄 세우기 3 + 전진 3 + 보관 + 기둥1


def test_v2_log_without_hooked_draws_all_posts_on_net(tmp_path, cfg, scenario):
    with LogWriter(tmp_path / "run.jsonl") as log:
        tick = Runner(scenario, cfg, SimBackend(cfg), log).tick()
    del tick["net"]["hooked"]
    scene = build_scene(tick, log.records, cfg, scenario)
    assert scene.net_chain == (0, 1, 2, 3, 4) and len(scene.segments) == 4


def test_link_status_follows_events():
    events = [
        {"type": "event", "t": 1.0, "kind": "comm_loss", "data": {"robot_id": 4}},
        {"type": "event", "t": 1.5, "kind": "camera_loss", "data": {}},
        {"type": "event", "t": 2.0, "kind": "net_limit", "data": {"robot_id": 4}},
        {"type": "event", "t": 3.0, "kind": "comm_restore", "data": {"robot_id": 4}},
    ]
    assert link_status(events, 0.5) == (True, frozenset(), ())
    assert link_status(events, 2.5) == (False, frozenset({4}), ("net_limit 4",))
    assert link_status(events, 3.5) == (False, frozenset(), ())


def test_manual_control_keys_to_commands(cfg):
    mc = ManualControl(cfg)
    assert mc.selected == 0
    mc.press("5")
    assert mc.selected == 4
    for _ in range(100):
        mc.press("up")
        mc.press("left")
    drive = {c.robot_id: c for c in mc.drive_commands()}
    assert set(drive) == {r.robot_id for r in cfg.robots}  # 와치독 유지용으로 전부 보냄
    max_steer = math.atan(cfg.robot.wheelbase_mm / cfg.robot.min_turn_radius_mm)
    assert drive[4] == DriveCommand(4, cfg.robot.max_speed_mm_s, pytest.approx(max_steer))
    assert drive[0] == DriveCommand(0, 0.0, 0.0)

    mc.press("space")
    assert mc.drive_commands()[4].speed_mm_s == 0
    assert mc.press("h") == [ActionCommand(0, "reel_hold")]   # 릴은 보관 로봇
    assert mc.press("c") == [ActionCommand(4, "clamp_open")]  # 클램프는 당김 로봇
    assert mc.press("c") == [ActionCommand(4, "clamp_close")]
    assert mc.press("x") == []


def test_playback_seek_speed_and_pause():
    ticks = [{"type": "tick", "t": round(i * 0.1, 9)} for i in range(51)]
    pb = Playback(ticks)
    assert pb.current()["t"] == 0.0
    pb.update(1.0)
    assert pb.current()["t"] == pytest.approx(1.0)
    pb.press("+")
    pb.update(1.0)
    assert pb.current()["t"] == pytest.approx(3.0)
    pb.press("space")
    pb.update(1.0)
    assert pb.current()["t"] == pytest.approx(3.0)
    pb.press("left")
    assert pb.current()["t"] == pytest.approx(2.0)
    pb.press("right")
    pb.press("right")
    pb.press("right")
    pb.press("right")
    assert pb.current()["t"] == pytest.approx(5.0)  # 끝에서 멈춤
    pb.press("-")
    pb.press("-")
    assert pb.speed == 0.5


def test_render_draws_robot_at_its_position(tmp_path, cfg, scenario, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    from fencebot.view.render import ROBOT, render, window_size

    with LogWriter(tmp_path / "run.jsonl") as log:
        runner = Runner(scenario, cfg, SimBackend(cfg), log)
        scene = build_scene(runner.tick(), log.records, cfg, scenario)
    pygame.font.init()
    tf = ScreenTransform.for_road(cfg.road, width_px=1400)
    surface = pygame.Surface(window_size(tf))
    render(surface, scene, tf, cfg, pygame.font.Font(None, 22), selected=None)

    puller = scene.robots[-1]
    x, y = tf.to_px(puller.x, puller.y - cfg.robot.radius_mm / 2)  # 방향 표시선을 피한 몸통
    assert tuple(surface.get_at((int(x), int(y))))[:3] == ROBOT["puller"]
