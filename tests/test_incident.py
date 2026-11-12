from dataclasses import replace
from pathlib import Path

import pytest

from fencebot.config import load_config
from fencebot.detection_io import DetectionFrame, load_frames
from fencebot.fake_detection import fake_frames, write_frames
from fencebot.incident import ZoneTracker, frame_incident, incident_from_frames, zone_from_vehicles
from fencebot.planner import plan_deployment
from fencebot.scenario import Vehicle, load_scenario
from fencebot.types import Detection

ROOT = Path(__file__).resolve().parents[1]
CAR = (4320, 274, 5880, 894)  # 1/3 승용차, 2차로 가운데


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def scenario():
    return load_scenario(ROOT / "scenarios" / "lane2_basic.toml")


def test_zone_adds_margin_and_keeps_width_inside_lane(cfg):
    z = zone_from_vehicles((Vehicle(1, "damaged", CAR), Vehicle(2, "normal", (7000, 1440, 8560, 2060))),
                           (), cfg)
    # 진행 방향은 여유 300 그대로, 폭 방향은 차로 경계까지 남은 거리의 절반까지만 (ADR-015)
    assert z.bbox == pytest.approx((4020, 137, 6180, 1030.5))
    assert z.lanes == ("lane2",) and z.kind == "damaged"


def test_no_accident_vehicle_means_no_zone(cfg):
    assert zone_from_vehicles((Vehicle(1, "normal", CAR),), (), cfg) is None


def test_touching_vehicle_joins_and_far_vehicle_does_not(cfg):
    touching = Vehicle(2, "normal", (5980, 300, 7540, 920))   # 100mm 뒤
    far = Vehicle(3, "normal", (500, 300, 2060, 920))
    z = zone_from_vehicles((Vehicle(1, "damaged", CAR), touching, far), (), cfg)
    assert z.bbox[0] == pytest.approx(4020) and z.bbox[2] == pytest.approx(7840)


def test_kind_priority_fire_over_overturned_over_damaged(cfg):
    cars = (Vehicle(1, "overturned", CAR),)
    assert zone_from_vehicles(cars, (), cfg).kind == "overturned"
    z = zone_from_vehicles(cars, ((5000, 950),), cfg)  # 차량 옆 화염
    assert z.kind == "fire" and z.bbox[0] == pytest.approx(4320 - 800)
    normal_with_smoke = zone_from_vehicles((Vehicle(1, "normal", CAR),), ((4400, 600),), cfg)
    assert normal_with_smoke.kind == "fire"
    assert zone_from_vehicles((Vehicle(1, "normal", CAR),), ((8000, 2000),), cfg) is None  # 먼 연기


def test_separate_accidents_pick_most_severe(cfg):
    a = Vehicle(1, "damaged", (1000, 274, 2560, 894))
    b = Vehicle(2, "overturned", (5000, 274, 6560, 894))
    assert zone_from_vehicles((a, b), (), cfg).kind == "overturned"


def test_fake_frame_recovers_scenario_zone(cfg, scenario):
    """비스듬한 임시 카메라로 만든 가짜 Detection에서 시나리오와 비슷한 범위가 나온다."""
    frame = fake_frames(scenario, cfg, count=1, noise_px=0.0)[0]
    inc = frame_incident(frame, cfg)
    assert {(v.track_id, v.label) for v in inc.vehicles} == {(1, "damaged"), (2, "normal")}
    assert inc.zone.kind == "damaged" and inc.zone.lanes == ("lane2",)
    for got, want in zip(inc.zone.bbox, scenario.zone.bbox):
        assert abs(got - want) < 150


def test_low_confidence_detection_is_ignored(cfg, scenario):
    frame = fake_frames(scenario, cfg, count=1, noise_px=0.0)[0]
    weak = tuple(replace(d, confidence=0.2) if d.track_id == 1 else d for d in frame.detections)
    assert frame_incident(replace(frame, detections=weak), cfg).zone is None


def test_tracker_confirms_after_consecutive_frames_and_only_grows(cfg, scenario):
    frames = fake_frames(scenario, cfg, count=cfg.incident.confirm_frames + 3, noise_px=1.0)
    tracker = ZoneTracker(cfg)
    seen = [tracker.update(frame_incident(f, cfg)) for f in frames]
    n = cfg.incident.confirm_frames
    assert all(z is None for z in seen[:n - 1]) and seen[n - 1] is not None
    for a, b in zip(seen[n - 1:], seen[n:]):
        assert b.bbox[0] <= a.bbox[0] and b.bbox[2] >= a.bbox[2]
    empty = DetectionFrame(999, 33.3, (), (), {})
    assert tracker.update(frame_incident(empty, cfg)) == seen[-1]  # 확정 뒤에는 한 프레임 놓쳐도 유지


def test_tracker_restarts_count_when_zone_disappears_before_confirm(cfg, scenario):
    frames = fake_frames(scenario, cfg, count=cfg.incident.confirm_frames, noise_px=0.0)
    tracker = ZoneTracker(cfg)
    empty = frame_incident(DetectionFrame(0, 0.0, (), (), {}), cfg)
    for f in frames[:-1]:
        tracker.update(frame_incident(f, cfg))
    tracker.update(empty)
    assert tracker.update(frame_incident(frames[-1], cfg)) is None


def test_detections_drive_deployment(cfg, scenario, tmp_path):
    """가짜 Detection 파일 → 범위·장애물 → 발표용 전개 계획까지 이어진다."""
    write_frames(fake_frames(scenario, cfg, count=20), tmp_path)
    result = incident_from_frames(load_frames(tmp_path), cfg)
    assert result.confirmed_frame == cfg.incident.confirm_frames - 1
    relaxed = replace(cfg, planning=replace(cfg.planning, strict=False))
    d = plan_deployment(result.zone, tuple(v.bbox for v in result.vehicles), cfg.standby, relaxed)
    assert d.ok and len(d.moves) == 8 and d.warnings == ()


def test_cli_fake_zone_and_demo(tmp_path, capsys):
    from fencebot.cli import main

    sc = str(ROOT / "scenarios" / "lane2_basic.toml")
    main(["fake", sc, str(tmp_path / "det"), "--frames", "12"])
    main(["zone", str(tmp_path / "det")])
    assert "damaged" in capsys.readouterr().out
    log = tmp_path / "demo.jsonl"
    main(["demo", sc, "--detections", str(tmp_path / "det"), "--log", str(log), "--no-window"])
    assert log.exists()
