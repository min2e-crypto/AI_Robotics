import tomllib

import pytest

from fencebot.calib import camera_toml, default_road_points, guide_lines, parse_rect, rect_points, road_to_image_for
from fencebot.config import load_config
from fencebot.homography import apply
from fencebot.incident import road_homography


def test_rect_gives_points_in_click_order():
    assert rect_points(parse_rect("1000, 0, 7667, 1167")) == ((1000, 0), (7667, 0), (7667, 1167), (1000, 1167))
    for bad in ("1,2,3", "a,b,c,d", "0,0,0,5"):
        with pytest.raises(ValueError):
            parse_rect(bad)


def test_default_road_points_come_from_config():
    cfg = load_config()
    assert default_road_points(cfg, None) == cfg.camera.road_points_mm
    assert default_road_points(cfg, (0, 0, 10, 20)) == ((0, 0), (10, 0), (10, 20), (0, 20))


def test_saved_points_load_as_config_and_map_clicks_to_road(tmp_path):
    clicks = [(302.26, 911.0), (1650.0, 880.5), (1400.0, 300.0), (520.0, 310.0)]
    road = rect_points((1000, 0, 7667, 1167))
    p = tmp_path / "shot.toml"
    p.write_text(camera_toml(clicks, list(road), "shot.jpg"), encoding="utf-8")
    assert set(tomllib.loads(p.read_text(encoding="utf-8"))) == {"camera"}
    cfg = load_config(p)
    assert cfg.camera.road_points_mm == road
    assert cfg.camera.image_points_px[0] == (302.3, 911.0)  # 0.1px로 반올림
    h = road_homography(cfg)
    for px, mm in zip(cfg.camera.image_points_px, road):
        assert apply(h, px) == pytest.approx(mm, abs=1e-6)
    with pytest.raises(ValueError):
        camera_toml(clicks[:3], list(road))


def test_guide_lines_pass_through_clicked_points_and_stay_near_image():
    cfg = load_config()
    road = list(cfg.camera.road_points_mm)
    to_px = road_to_image_for(list(cfg.camera.image_points_px), road)
    lines = guide_lines(cfg.road, to_px, (1920, 1080), road[0])
    assert len(lines) == len(cfg.road.lines) + 10  # 차선 3개 + x 0~9000의 1m 눈금 10개
    pts = [q for line in lines for q in line]
    assert all(-3 * 1920 <= x <= 4 * 1920 and -3 * 1080 <= y <= 4 * 1080 for x, y in pts)
    # 눈금 x = 0은 도로 4점의 (0, -1000)과 (0, 2333)을 잇는다
    first_tick = lines[len(cfg.road.lines)]
    assert first_tick[0] == pytest.approx(cfg.camera.image_points_px[0])
    assert first_tick[-1] == pytest.approx(cfg.camera.image_points_px[3])


def test_guide_lines_drop_points_behind_camera():
    cfg = load_config()
    # 지평선이 영상 안에 있는 낮은 카메라: 도로 x 0~3000만 찍었고 먼 쪽은 지평선 너머로 넘어간다
    road = [(0, 0), (3000, 0), (3000, 1167), (0, 1167)]
    image = [(200, 1000), (700, 420), (1100, 420), (1700, 1000)]
    to_px = road_to_image_for(image, road)
    lines = guide_lines(cfg.road, to_px, (1920, 1080), road[0])
    assert lines
    assert all(-3 * 1920 <= x <= 4 * 1920 and -3 * 1080 <= y <= 4 * 1080 for line in lines for x, y in line)
