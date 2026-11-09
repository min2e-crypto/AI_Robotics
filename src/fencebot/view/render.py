"""Scene을 pygame 화면에 그린다. 글자는 운영체제와 상관없이 보이도록 영문·숫자만 쓴다."""

from __future__ import annotations

import math

import pygame

from ..config import Config
from .scene import Scene, ScreenTransform

HUD_HEIGHT_PX = 110

BG = (52, 56, 60)
SHOULDER = (70, 74, 78)
LINE = (235, 235, 235)
ZONE = (220, 60, 60)
VEHICLE = {"normal": (110, 140, 190), "damaged": (230, 150, 50), "overturned": (170, 90, 200)}
ROBOT = {"reserve": (90, 200, 120), "post1": (80, 170, 230), "post2": (80, 170, 230),
         "post3": (80, 170, 230), "puller": (240, 210, 80)}
OFFLINE = (130, 130, 130)
NET_OK = (120, 230, 160)
NET_BAD = (255, 70, 70)
SELECT = (255, 255, 255)
TEXT = (230, 230, 230)
WARN = (255, 120, 90)

NET_NEAR_LIMIT = 0.95  # 전체 길이가 그물 길이의 이 비율을 넘으면 빨간색


def window_size(tf: ScreenTransform) -> tuple[int, int]:
    return int(tf.width_px), int(tf.height_px + HUD_HEIGHT_PX)


def render(surface: pygame.Surface, scene: Scene, tf: ScreenTransform, config: Config,
           font: pygame.font.Font, selected: int | None = None, extra: str = "") -> None:
    surface.fill(BG)
    _road(surface, scene, tf, config)
    _zone_and_vehicles(surface, scene, tf)
    _paths(surface, scene, tf, config)
    _targets(surface, scene, tf, config)
    _net(surface, scene, tf, config)
    _robots(surface, scene, tf, config, selected)
    _hud(surface, scene, tf, config, font, extra)


def _road(surface, scene, tf, config):
    lo, hi = config.road.lanes["shoulder"]
    x0, y0 = tf.to_px(config.road.x_min_mm, hi)
    pygame.draw.rect(surface, SHOULDER, (x0, y0, tf.length(config.road.x_max_mm - config.road.x_min_mm),
                                         tf.length(hi - lo)))
    for line in scene.lines:
        pts = [tf.to_px(*p) for p in line.points]
        for a, b in zip(pts, pts[1:]):
            if line.kind == "dashed":
                _dashed(surface, LINE, a, b, dash=tf.length(300))
            else:
                pygame.draw.line(surface, LINE, a, b, 2)


def _zone_and_vehicles(surface, scene, tf):
    for label, box in scene.vehicles:
        pygame.draw.rect(surface, VEHICLE.get(label, OFFLINE), _rect(tf, box))
    if scene.zone:
        pygame.draw.rect(surface, ZONE, _rect(tf, scene.zone), 2)


def _paths(surface, scene, tf, config):
    """전개 계획 경로: 역할 색을 어둡게 한 가는 선."""
    roles = {r.robot_id: r.role.value for r in config.robots}
    for rid, pts in scene.paths:
        color = tuple(v // 2 for v in ROBOT[roles[rid]])
        pygame.draw.lines(surface, color, False, [tf.to_px(*p) for p in pts], 1)


def _targets(surface, scene, tf, config):
    """목표 지점: 역할 색의 + 표시와 목표 방향 선."""
    roles = {r.robot_id: r.role.value for r in config.robots}
    arm = tf.length(config.robot.radius_mm) * 0.6
    for t in scene.targets:
        color = ROBOT[roles[t.robot_id]]
        cx, cy = tf.to_px(t.x, t.y)
        pygame.draw.line(surface, color, (cx - arm, cy), (cx + arm, cy), 2)
        pygame.draw.line(surface, color, (cx, cy - arm), (cx, cy + arm), 2)
        tip = tf.to_px(t.x + config.robot.radius_mm * 1.6 * math.cos(t.heading),
                       t.y + config.robot.radius_mm * 1.6 * math.sin(t.heading))
        pygame.draw.line(surface, color, (cx, cy), tip, 1)


def _net(surface, scene, tf, config):
    near_limit = scene.net_length > config.net.length_mm * NET_NEAR_LIMIT
    pts = [tf.to_px(r.x, r.y) for r in scene.robots]
    for a, b, seg in zip(pts, pts[1:], scene.segments):
        bad = near_limit or seg > config.net.max_gap_mm
        pygame.draw.line(surface, NET_BAD if bad else NET_OK, a, b, 2)


def _robots(surface, scene, tf, config, selected):
    radius = tf.length(config.robot.radius_mm)
    for r in scene.robots:
        color = ROBOT[r.role] if r.comm_ok else OFFLINE
        c = tf.to_px(r.x, r.y)
        pygame.draw.circle(surface, color, c, radius)
        tip = tf.to_px(r.x + config.robot.radius_mm * 1.4 * math.cos(r.heading),
                       r.y + config.robot.radius_mm * 1.4 * math.sin(r.heading))
        pygame.draw.line(surface, BG, c, tip, 3)
        pygame.draw.circle(surface, color, tf.to_px(r.seen[0], r.seen[1]), radius, 1)  # 카메라 추정
        if r.robot_id == selected:
            pygame.draw.circle(surface, SELECT, c, radius + 4, 2)


def _hud(surface, scene, tf, config, font, extra):
    top = tf.height_px + 8
    segs = " ".join(f"{s:.0f}" for s in scene.segments)
    rows = [
        f"t {scene.t:7.2f}s   phase {scene.phase}   reel {scene.reel}   clamp {scene.clamp}",
        f"net {scene.net_length:.0f} / unwound {scene.unwound:.0f} / max {config.net.length_mm:.0f} mm"
        f"   slack {scene.slack:.0f} mm   segments [{segs}] (max gap {config.net.max_gap_mm:.0f})",
    ]
    if scene.placement_failures:
        rows[0] += f"   placement FAIL: {', '.join(scene.placement_failures)}"
    elif scene.targets:
        rows[0] += "   placement OK (+ = target, thin line = planned path)"
    alerts = list(scene.flashes)
    if not scene.camera_ok:
        alerts.append("CAMERA LOST")
    alerts += [f"COMM LOST {r.robot_id}" for r in scene.robots if not r.comm_ok]
    rows.append("   ".join(alerts))
    rows.append(extra)
    for i, row in enumerate(rows):
        color = WARN if i == 2 else TEXT
        surface.blit(font.render(row, True, color), (tf.margin, top + i * 24))


def _rect(tf, box):
    x1, y1, x2, y2 = box
    left, top = tf.to_px(x1, y2)
    return pygame.Rect(left, top, tf.length(x2 - x1), tf.length(y2 - y1))


def _dashed(surface, color, a, b, dash):
    length = math.dist(a, b)
    n = max(1, int(length // dash))
    for k in range(0, n, 2):
        s, e = k / n, min(k + 1, n) / n
        pygame.draw.line(surface, color, (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s),
                         (a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e), 2)
