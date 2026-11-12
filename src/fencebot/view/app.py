"""수동 조작 창(fencebot sim)과 로그 재생 창(fencebot view)."""

from __future__ import annotations

from pathlib import Path

import pygame

from ..config import Config, config_from_raw
from ..log import LogWriter, read_log
from ..runner import Runner
from ..scenario import Scenario, scenario_from_raw
from ..playback import PlaybackBackend
from ..sim import SimBackend
from .control import ManualControl, Playback
from .render import render, window_size
from .scene import ScreenTransform, build_scene, deployment_for

WIDTH_PX = 1400
FPS = 60

MANUAL_HELP = ("1-5 select  up/down speed  left/right steer  space stop  "
               "u/h/w reel unwind/hold/wind  c clamp  esc quit")
DEMO_HELP = "auto deployment (no physics)  esc quit"
VIEW_HELP = "space pause  left/right seek 1s  +/- speed  esc quit"


def run_manual(scenario: Scenario, config: Config, log_path: str | Path) -> None:
    """실시간으로 시뮬레이터를 돌리며 키로 조작한다. 끝나면 log_path를 fencebot view로 다시 볼 수 있다."""
    screen, font, tf = _open(config, f"fencebot sim - {scenario.name}")
    control = ManualControl(config)
    backend = SimBackend(config, seed=scenario.seed)
    clock = pygame.time.Clock()
    plan = deployment_for(config, scenario)
    with LogWriter(log_path) as log:
        runner = Runner(scenario, config, backend, log)
        tick = runner.tick()
        acc = 0.0
        while not runner.done:
            for key in _keys():
                if key == "escape":
                    return
                for cmd in control.press(key):
                    backend.send(cmd)
            acc += clock.tick(FPS) / 1000
            while acc >= scenario.dt_s and not runner.done:
                for cmd in control.drive_commands():  # 주기 송신 (와치독 유지)
                    backend.send(cmd)
                tick = runner.tick()
                acc -= scenario.dt_s
            scene = build_scene(tick, log.records, config, scenario, plan)
            render(screen, scene, tf, config, font, control.selected, MANUAL_HELP)
            pygame.display.flip()


def run_demo(scenario: Scenario, config: Config, backend: PlaybackBackend, log_path: str | Path) -> None:
    """전개 계획을 실시간으로 재생한다 (ADR-014). 끝나면 esc를 누를 때까지 마지막 화면을 유지한다."""
    screen, font, tf = _open(config, f"fencebot demo - {scenario.name}")
    clock = pygame.time.Clock()
    plan = deployment_for(config, scenario)
    with LogWriter(log_path) as log:
        runner = Runner(scenario, config, backend, log)
        tick = runner.tick()
        acc = 0.0
        while "escape" not in _keys():
            acc += clock.tick(FPS) / 1000
            while acc >= scenario.dt_s:
                if not runner.done:
                    tick = runner.tick()
                acc -= scenario.dt_s
            scene = build_scene(tick, log.records, config, scenario, plan)
            render(screen, scene, tf, config, font, None, DEMO_HELP)
            pygame.display.flip()


def run_view(log_path: str | Path) -> None:
    records = read_log(log_path)
    header = records[0]
    config, scenario = config_from_raw(header["config"]), scenario_from_raw(header["scenario"])
    playback = Playback([r for r in records if r["type"] == "tick"])
    screen, font, tf = _open(config, f"fencebot view - {Path(log_path).name}")
    clock = pygame.time.Clock()
    plan = deployment_for(config, scenario)
    while True:
        for key in _keys():
            if key == "escape":
                return
            playback.press(key)
        playback.update(clock.tick(FPS) / 1000)
        scene = build_scene(playback.current(), records, config, scenario, plan)
        state = "paused" if playback.paused else f"x{playback.speed:g}"
        render(screen, scene, tf, config, font, None, f"[{state}]  {VIEW_HELP}")
        pygame.display.flip()


def _open(config: Config, title: str):
    pygame.init()
    pygame.key.set_repeat(250, 60)
    tf = ScreenTransform.for_road(config.road, WIDTH_PX)
    screen = pygame.display.set_mode(window_size(tf))
    pygame.display.set_caption(title)
    return screen, pygame.font.Font(None, 22), tf


def _keys() -> list[str]:
    keys = []
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            keys.append("escape")
        elif ev.type == pygame.KEYDOWN:
            name = pygame.key.name(ev.key)
            keys.append({"[+]": "+", "[-]": "-"}.get(name, name))
    return keys
