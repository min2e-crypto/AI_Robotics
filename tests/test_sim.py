import math
from dataclasses import replace

import pytest

from fencebot.config import load_config
from fencebot.io import ActionCommand, DriveCommand
from fencebot.scenario import Event
from fencebot.sim import SimBackend

DT = 0.02
PULLER = 4


def quiet_config(**sim):
    """카메라 잡음·가림이 없는 설정. 필요한 값만 덮어쓴다."""
    cfg = load_config()
    base = dict(pos_noise_mm=0.0, heading_noise_rad=0.0, occlusion_prob=0.0)
    base.update(sim)
    return replace(cfg, sim=replace(cfg.sim, **base))


def drive(sim, seconds, cmds, on_step=None):
    """명령을 매 스텝 다시 보내며 진행한다 (실물 관제처럼 주기 송신)."""
    for _ in range(round(seconds / DT)):
        for c in cmds:
            sim.send(c)
        sim.step(DT)
        if on_step:
            on_step(sim)


def pose(sim, robot_id):
    return next(r for r in sim.truth() if r.robot_id == robot_id)


def test_turn_radius_never_below_minimum():
    cfg = quiet_config()
    sim = SimBackend(cfg)
    cmd = DriveCommand(PULLER, speed_mm_s=10000, steer_rad=1.0)  # 둘 다 상한 초과
    prev = [pose(sim, PULLER)]
    min_radius = []

    def check(s):
        p = pose(s, PULLER)
        dist = math.hypot(p.x - prev[0].x, p.y - prev[0].y)
        dtheta = abs(math.remainder(p.heading - prev[0].heading, math.tau))
        if dtheta > 1e-9:
            min_radius.append(dist / dtheta)
        prev[0] = p

    drive(sim, 15.0, [cmd], check)
    # 스텝 사이 현(chord)으로 재므로 호 반지름보다 아주 조금 작게 나온다
    assert min(min_radius) >= cfg.robot.min_turn_radius_mm * (1 - 1e-4)
    assert min_radius[-1] == pytest.approx(cfg.robot.min_turn_radius_mm, rel=1e-3)


def test_speed_is_clamped_to_max():
    cfg = quiet_config()
    sim = SimBackend(cfg)
    start = pose(sim, PULLER)
    drive(sim, 2.0, [DriveCommand(PULLER, speed_mm_s=10000, steer_rad=0.0)])
    end = pose(sim, PULLER)
    assert end.x - start.x == pytest.approx(cfg.robot.max_speed_mm_s * 2.0, rel=1e-6)
    assert end.y == pytest.approx(start.y)


def test_steering_follows_command_with_delay():
    cfg = quiet_config()
    sim = SimBackend(cfg)
    tau = cfg.robot.steer_delay_s
    drive(sim, tau, [DriveCommand(PULLER, speed_mm_s=0, steer_rad=0.3)])
    assert sim.steer(PULLER) == pytest.approx(0.3 * (1 - math.exp(-1)), rel=0.05)


def test_puller_stops_at_net_length():
    cfg = quiet_config()
    sim = SimBackend(cfg)
    lengths = []
    drive(sim, 20.0, [DriveCommand(PULLER, speed_mm_s=300, steer_rad=0.0)],
          lambda s: lengths.append(s.net_length()))
    assert max(lengths) <= cfg.net.length_mm + 1e-6
    assert lengths[-1] > cfg.net.length_mm - cfg.robot.max_speed_mm_s * DT
    assert [e["kind"] for e in sim.pop_events()] == ["net_limit"]


def test_reel_hold_fixes_unwound_length():
    cfg = quiet_config()
    sim = SimBackend(cfg)
    sim.send(ActionCommand(0, "reel_hold"))
    held = sim.unwound_mm()
    start = pose(sim, PULLER)
    drive(sim, 2.0, [DriveCommand(PULLER, speed_mm_s=300, steer_rad=0.0)])
    assert pose(sim, PULLER).x == pytest.approx(start.x)
    assert sim.unwound_mm() == held

    # 당김 로봇이 후진하면 길이가 줄어 허용되고, 풀린 길이는 그대로 (처짐)
    drive(sim, 1.0, [DriveCommand(PULLER, speed_mm_s=-300, steer_rad=0.0)])
    assert sim.net_length() < held
    assert sim.unwound_mm() == held


def test_reel_wind_takes_up_slack_and_blocks_extension():
    cfg = quiet_config()
    sim = SimBackend(cfg)
    sim.send(ActionCommand(0, "reel_wind"))
    drive(sim, 1.0, [DriveCommand(PULLER, speed_mm_s=-300, steer_rad=0.0)])
    assert sim.unwound_mm() == pytest.approx(sim.net_length())
    before = sim.unwound_mm()
    drive(sim, 1.0, [DriveCommand(PULLER, speed_mm_s=300, steer_rad=0.0)])
    assert sim.unwound_mm() <= before + 1e-6


def test_camera_returns_delayed_pose():
    cfg = quiet_config(camera_latency_s=0.1)
    sim = SimBackend(cfg)
    history = {}

    def record(s):
        history[round(s.time, 6)] = pose(s, PULLER)

    drive(sim, 1.0, [DriveCommand(PULLER, speed_mm_s=300, steer_rad=0.0)], record)
    seen = next(r for r in sim.read(sim.time) if r.robot_id == PULLER)
    expected = history[round(sim.time - 0.1, 6)]
    assert seen.x == pytest.approx(expected.x)
    assert seen.t == pytest.approx(sim.time - 0.1)


def test_camera_noise_is_reproducible_by_seed():
    cfg = load_config()

    def reads(seed):
        sim = SimBackend(cfg, seed=seed)
        out = []
        for _ in range(20):
            sim.step(DT)
            out.append([(r.x, r.y, r.heading) for r in sim.read(sim.time)])
        return out

    assert reads(1) == reads(1)
    assert reads(1) != reads(2)


def test_occluded_robot_keeps_last_seen_pose():
    cfg = quiet_config(occlusion_prob=1.0)
    sim = SimBackend(cfg)
    start = pose(sim, PULLER)
    drive(sim, 1.0, [DriveCommand(PULLER, speed_mm_s=300, steer_rad=0.0)])
    seen = next(r for r in sim.read(sim.time) if r.robot_id == PULLER)
    assert (seen.x, seen.t) == (start.x, 0.0)


def test_camera_loss_freezes_observation():
    cfg = quiet_config()
    sim = SimBackend(cfg)
    cmd = DriveCommand(PULLER, speed_mm_s=300, steer_rad=0.0)
    drive(sim, 0.5, [cmd])
    frozen = sim.read(sim.time)
    sim.handle(Event(sim.time, "camera_loss"))
    drive(sim, 0.5, [cmd])
    assert sim.read(sim.time) == frozen
    sim.handle(Event(sim.time, "camera_restore"))
    drive(sim, 0.5, [cmd])
    assert sim.read(sim.time) != frozen


def test_comm_loss_triggers_watchdog_stop_then_resumes():
    cfg = quiet_config()
    sim = SimBackend(cfg)
    cmd = DriveCommand(PULLER, speed_mm_s=300, steer_rad=0.0)
    drive(sim, 0.5, [cmd])
    sim.handle(Event(sim.time, "comm_loss", {"robot_id": PULLER}))

    drive(sim, cfg.sim.watchdog_hold_s, [cmd])
    assert sim.speed(PULLER) == pytest.approx(300)  # 짧은 끊김은 마지막 명령 유지

    drive(sim, cfg.sim.watchdog_stop_s + 0.1, [cmd])
    assert sim.speed(PULLER) == 0
    stopped = pose(sim, PULLER)
    drive(sim, 0.5, [cmd])
    assert pose(sim, PULLER) == replace(stopped, t=sim.time)

    sim.handle(Event(sim.time, "comm_restore", {"robot_id": PULLER}))
    drive(sim, 0.5, [cmd])
    assert sim.speed(PULLER) == pytest.approx(300)


def test_no_command_means_watchdog_stop_even_without_comm_loss():
    cfg = quiet_config()
    sim = SimBackend(cfg)
    drive(sim, 0.5, [DriveCommand(PULLER, speed_mm_s=300, steer_rad=0.0)])
    drive(sim, cfg.sim.watchdog_hold_s + cfg.sim.watchdog_stop_s + 0.1, [])
    assert sim.speed(PULLER) == 0



def test_posts_start_detached_from_net():
    sim = SimBackend(quiet_config())
    assert sim.hooked() == set()
    p = {r.robot_id: (r.x, r.y) for r in sim.truth()}
    assert sim.net_length() == pytest.approx(math.dist(p[0], p[PULLER]))


def test_post_hooks_when_crossing_net_line():
    cfg = quiet_config()
    sim = SimBackend(cfg)
    post2 = 2
    # 갓길 바깥 줄에서 왼쪽으로 돌며 보관-당김 그물선을 아래에서 위로 가로지른다
    drive(sim, 3.0, [DriveCommand(post2, speed_mm_s=300, steer_rad=1.0)])
    assert sim.hooked() == {post2}
    events = sim.pop_events()
    assert [e["kind"] for e in events] == ["net_hook"]
    assert events[0]["data"] == {"robot_id": post2, "between": [0, PULLER]}
    p = {r.robot_id: (r.x, r.y) for r in sim.truth()}
    assert sim.net_length() == pytest.approx(math.dist(p[0], p[post2]) + math.dist(p[post2], p[PULLER]))
    assert sim.log_fields()["net"]["hooked"] == [post2]

