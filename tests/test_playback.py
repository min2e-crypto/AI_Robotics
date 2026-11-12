import math
from dataclasses import replace
from pathlib import Path

import pytest

from fencebot.config import load_config
from fencebot.log import LogWriter
from fencebot.planner import plan_deployment
from fencebot.playback import PlaybackBackend
from fencebot.runner import Runner
from fencebot.scenario import load_scenario
from fencebot.types import Role

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def scenario():
    return load_scenario(ROOT / "scenarios" / "lane2_basic.toml")


def deploy(cfg, scenario):
    return plan_deployment(scenario.zone, tuple(v.bbox for v in scenario.vehicles), cfg.standby, cfg)


def test_playback_moves_robots_to_targets_and_hooks_posts(cfg, scenario):
    plan = deploy(cfg, scenario)
    backend = PlaybackBackend(cfg, plan, start_delay_s=0.5)
    dt, limit = 0.1, cfg.robot.max_speed_mm_s * 0.1 + 1e-6
    assert backend.phase == "standby"
    prev = {p.robot_id: p for p in backend.truth()}
    moved_at_once = 0
    for _ in range(round(backend.duration_s / dt) + 1):
        backend.step(dt)
        now = {p.robot_id: p for p in backend.truth()}
        steps = [math.dist((now[r].x, now[r].y), (prev[r].x, prev[r].y)) for r in now]
        assert max(steps) <= limit
        moved_at_once = max(moved_at_once, sum(s > 0 for s in steps))
        prev = now
    assert moved_at_once == 1  # 한 번에 한 대씩
    assert backend.done and backend.phase == "hold"
    for t in plan.targets:
        p = prev[t.robot_id]
        assert math.dist((p.x, p.y), (t.x, t.y)) < 1e-6 and abs(p.heading - t.heading) < 1e-6
    rid = {r.role: r.robot_id for r in cfg.robots}
    hooks = [e["data"]["robot_id"] for e in backend.pop_events() if e["kind"] == "net_hook"]
    assert hooks == [rid[Role.POST3], rid[Role.POST2], rid[Role.POST1]]
    assert backend.log_fields()["net"]["hooked"] == [rid[Role.POST1], rid[Role.POST2], rid[Role.POST3]]


def test_playback_log_has_phase_truth_and_net(cfg, scenario, tmp_path):
    plan = deploy(cfg, scenario)
    backend = PlaybackBackend(cfg, plan)
    sc = replace(scenario, duration_s=backend.duration_s, events=())
    with LogWriter(tmp_path / "demo.jsonl") as log:
        runner = Runner(sc, cfg, backend, log)
        while not runner.done:
            runner.tick()
        ticks = [r for r in log.records if r["type"] == "tick"]
    assert ticks[0]["phase"] == "standby" and ticks[-1]["phase"] == "hold"
    assert {"puller_deploy", "posts_enter"} <= {t["phase"] for t in ticks}
    assert ticks[-1]["net"]["length_mm"] == pytest.approx(5610, abs=5)
    assert ticks[-1]["robots"] == [{**p, "t": ticks[-1]["t"]} for p in ticks[-1]["truth"]]


def test_demo_command_writes_log(tmp_path, capsys):
    from fencebot.cli import main

    log_path = tmp_path / "demo.jsonl"
    main(["demo", str(ROOT / "scenarios" / "lane2_basic.toml"), "--log", str(log_path), "--no-window"])
    assert log_path.exists() and "저장" in capsys.readouterr().out
