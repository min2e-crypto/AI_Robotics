import os
from dataclasses import replace
from pathlib import Path

import pytest

from fencebot.config import config_from_raw, load_config
from fencebot.fake_detection import fake_frames
from fencebot.homography import apply
from fencebot.incident import road_homography
from fencebot.presentation import (DEPLOY, DETECT, DONE, NO_INCIDENT, ZONE, Show, box_outline_px,
                                   frame_views, road_to_image)
from fencebot.scenario import load_scenario
from fencebot.types import Detection, VehicleState

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def scenario():
    return load_scenario(ROOT / "scenarios" / "lane2_basic.toml")


def test_views_count_up_to_confirmation(cfg, scenario):
    views = frame_views(fake_frames(scenario, cfg, count=8), cfg)
    n = cfg.incident.confirm_frames
    assert [v.streak for v in views[:n]] == list(range(1, n + 1)) and views[0].need == n
    assert [v.zone is not None for v in views] == [False] * (n - 1) + [True] * (8 - n + 1)
    assert {(b.track_id, b.kind) for b in views[0].boxes} == {(1, "damaged"), (2, "normal")}
    assert len(views[0].vehicles) == 2 and views[0].checks == ()


def test_show_stops_frames_at_confirmation_then_moves_to_deploy(cfg, scenario):
    show = Show(frame_views(fake_frames(scenario, cfg, count=20), cfg), frame_s=0.5, zone_s=2.0)
    n = cfg.incident.confirm_frames
    assert show.confirmed == n - 1 and show.result.zone is not None
    assert (show.at(0.0).stage, show.at(0.0).index) == (DETECT, 0)
    assert (show.at(0.6).stage, show.at(0.6).index) == (DETECT, 1)
    assert (show.at(n * 0.5 + 0.1).stage, show.at(n * 0.5 + 0.1).index) == (ZONE, n - 1)
    assert (show.at(n * 0.5 + 2.1).stage, show.at(999).index) == (DEPLOY, n - 1)  # 프레임은 더 넘기지 않는다


def test_show_without_incident_plays_all_frames(cfg, scenario):
    calm = replace(scenario, vehicles=tuple(replace(v, label="normal") for v in scenario.vehicles))
    show = Show(frame_views(fake_frames(calm, cfg, count=6), cfg), frame_s=0.5)
    assert show.confirmed is None and show.at(2.6).index == 5
    assert show.at(99).stage == NO_INCIDENT and show.result.zone is None


def test_checks_list_what_is_left_to_the_operator(cfg, scenario):
    f = fake_frames(scenario, cfg, count=1, noise_px=0.0)[0]
    car = next(d for d in f.detections if d.track_id == 2)
    extra = (Detection(0, 0.0, "smoke", 0.6, (100, 100, 200, 200)),
             replace(car, cls="fire", track_id=None),            # 정상 차량 옆의 불
             Detection(0, 0.0, "smoke", 0.1, (0, 0, 10, 10)))    # 확신도 미달은 보이지 않는다
    states = tuple(replace(s, label="uncertain") if s.track_id == 2 else s for s in f.vehicle_states)
    view = frame_views([replace(f, detections=f.detections + extra, vehicle_states=states)], cfg)[0]
    assert set(view.checks) == {"uncertain:2", "smoke", "fire_alone"}
    assert [b.kind for b in view.boxes].count("smoke") == 1


def test_vehicle_without_state_is_shown_as_normal(cfg, scenario):
    f = fake_frames(scenario, cfg, count=1)[0]
    view = frame_views([replace(f, vehicle_states=())], cfg)[0]
    assert {b.kind for b in view.boxes} == {"normal"} and all(b.confidence == 0.9 for b in view.boxes)


def test_zone_outline_maps_back_to_road(cfg):
    box = (4020, 134, 6180, 1034)
    pts = box_outline_px(box, road_to_image(cfg))
    back = [apply(road_homography(cfg), p) for p in pts]
    assert back == [pytest.approx(p) for p in ((4020, 134), (6180, 134), (6180, 1034), (4020, 1034))]


def test_draw_every_stage_without_a_window(cfg, scenario, tmp_path, monkeypatch):
    """창 없이 단계별로 한 장씩 그려 본다 (탐지 → 범위 확정 → 전개 → 완료, 사고 없음, 영상 사진)."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    from fencebot.cli import _playback, _with_incident
    from fencebot.log import LogWriter
    from fencebot.runner import Runner
    from fencebot.view.present import Fonts, Layout, draw, image_files, scene_for

    cfg = config_from_raw({**cfg.raw, "planning": {**cfg.raw["planning"], "strict": False}})
    show = Show(frame_views(fake_frames(scenario, cfg), cfg))
    sc, plan, backend = _playback(_with_incident(scenario, show.result), cfg)
    pygame.init()
    pygame.display.set_mode((10, 10))
    try:
        layout, fonts = Layout.make(cfg, 1000), Fonts.load()
        surface = pygame.Surface(layout.size)
        assert layout.camera.width + layout.panel.width == 1000 and layout.size[1] < 1000
        pygame.image.save(pygame.Surface((640, 360)), str(tmp_path / "a_000000.jpg"))
        image = pygame.image.load(str(image_files(tmp_path)[0]))
        with LogWriter(tmp_path / "log.jsonl") as log:
            runner = Runner(sc, cfg, backend, log)
            draw(surface, layout, fonts, cfg, show.at(0), DETECT,
                 scene_for(show.at(0), DETECT, None, [], cfg, sc, plan), image)
            st = show.at(show.detect_s + 0.1)
            waiting = scene_for(st, ZONE, None, [], cfg, sc, plan)
            assert waiting.zone and waiting.paths and waiting.phase == "standby"
            draw(surface, layout, fonts, cfg, st, ZONE, waiting)
            tick = None
            while not runner.done:
                tick = runner.tick()
            done = scene_for(show.at(99), DONE, tick, log.records, cfg, sc, plan)
            assert done.phase == "hold" and len(done.net_chain) == 5
            draw(surface, layout, fonts, cfg, show.at(99), DONE, done)
        searching = scene_for(show.at(0), DETECT, None, [], cfg, sc, plan)
        assert searching.zone is None and len(searching.vehicles) == 2 and not searching.paths
        none = replace(show.at(0), stage=NO_INCIDENT)
        draw(surface, layout, fonts, cfg, none, NO_INCIDENT, searching)
    finally:
        pygame.quit()
