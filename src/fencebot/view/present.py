"""발표 화면 (fencebot present, D2, ADR-017).

위 왼쪽: 카메라 화면(영상 프레임 + 탐지 박스 + 확정한 범위), 위 오른쪽: 정보 패널, 아래: 시뮬레이터.
영상 프레임 사진이 없으면 도로와 차량을 카메라 시점으로 옮겨 그린 그림을 대신 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pygame

from ..config import Config
from ..homography import apply
from ..log import LogWriter
from ..playback import PlaybackBackend
from ..presentation import (DEPLOY, DETECT, DONE, NO_INCIDENT, STAGES, ZONE, Show, ShowState,
                            box_outline_px, road_to_image)
from ..runner import Runner
from ..scenario import Scenario
from .render import BG, LINE, SHOULDER, TEXT, VEHICLE, WARN, render
from .render import ZONE as ZONE_COLOR
from .scene import Scene, ScreenTransform, build_scene

FPS = 60
IMAGE_SIZE_PX = (1920, 1080)  # 영상 프레임 사진이 없을 때 가정하는 영상 크기 (설정 [camera]의 임시 카메라)
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TOP_RATIO = 0.32              # 위쪽 줄 높이 / 창 폭
PANEL_BG = (34, 37, 41)
CAMERA_BG = (24, 26, 29)
DIM = (120, 124, 130)
ACCENT = (120, 230, 160)
BOX = {**VEHICLE, "uncertain": (245, 245, 245), "fire": (255, 110, 40), "smoke": (170, 170, 170)}
KOREAN_FONTS = ("applesdgothicneo", "malgungothic", "nanumgothic", "notosanscjkkr", "applegothic")

TEXTS = {  # 한글 글꼴이 없으면 영문으로 그린다
    "title": ("사고 현장 그물 펜스 전개", "Net fence deployment"),
    "camera": ("카메라", "camera"),
    "sim": ("시뮬레이터 (위에서 본 도로)", "simulator (top view)"),
    DETECT: ("탐지", "detect"), ZONE: ("범위 확정", "zone"), DEPLOY: ("전개", "deploy"), DONE: ("완료", "done"),
    "frame": ("프레임", "frame"), "confirm": ("범위 확정", "zone confirm"),
    "vehicle": ("차량", "vehicle"), "normal": ("정상", "normal"), "damaged": ("파손", "damaged"),
    "overturned": ("전복", "overturned"), "uncertain": ("애매", "uncertain"),
    "fire": ("불", "fire"), "smoke": ("연기", "smoke"),
    "zone_info": ("사고 범위", "zone"), "no_zone": ("사고 범위 없음", "no incident zone"),
    "check": ("확인 필요", "check"), "fire_alone": ("불 (사고 차량 없음)", "fire without accident vehicle"),
    "uncertain_vehicle": ("애매 차량", "uncertain vehicle"),
    "deploy_info": ("전개", "deploy"), "hooked": ("그물 걸림", "hooked"), "warn": ("경고", "warning"),
    "standby": ("대기", "standby"), "puller_deploy": ("당김 로봇이 그물을 폄", "puller deploying"),
    "posts_enter": ("기둥 로봇 진입", "posts entering"), "hold": ("전개 완료", "holding"),
    "lane2": ("2차로", "lane 2"), "lane1": ("1차로", "lane 1"), "shoulder": ("갓길", "shoulder"),
}


@dataclass(frozen=True)
class Fonts:
    big: pygame.font.Font
    body: pygame.font.Font
    small: pygame.font.Font
    korean: bool

    @classmethod
    def load(cls) -> Fonts:
        pygame.font.init()
        path = next((p for p in map(pygame.font.match_font, KOREAN_FONTS) if p), None)
        return cls(pygame.font.Font(path, 26), pygame.font.Font(path, 19), pygame.font.Font(path, 15),
                   path is not None)

    def t(self, key: str) -> str:
        pair = TEXTS.get(key)
        return key if pair is None else pair[0 if self.korean else 1]


@dataclass(frozen=True)
class Layout:
    width: int
    top: int                 # 위쪽 줄 높이
    camera: pygame.Rect
    panel: pygame.Rect
    sim: pygame.Rect
    tf: ScreenTransform

    @classmethod
    def make(cls, config: Config, width: int) -> Layout:
        tf = ScreenTransform.for_road(config.road, width)
        top = round(width * TOP_RATIO)
        cam_w = round(top * IMAGE_SIZE_PX[0] / IMAGE_SIZE_PX[1])
        return cls(width, top, pygame.Rect(0, 0, cam_w, top), pygame.Rect(cam_w, 0, width - cam_w, top),
                   pygame.Rect(0, top, width, int(tf.height_px)), tf)

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.top + self.sim.height


def standby_tick(config: Config) -> dict:
    """전개를 시작하기 전 화면에 쓸 tick (로봇은 대기 배치)."""
    return {"t": 0.0, "phase": "standby",
            "robots": [{"robot_id": s.robot_id, "x": s.x, "y": s.y, "heading": s.heading, "t": 0.0}
                       for s in config.standby]}


def draw(surface: pygame.Surface, layout: Layout, fonts: Fonts, config: Config, state: ShowState,
         stage: str, scene: Scene, image: pygame.Surface | None = None) -> None:
    """발표 화면 한 장. stage는 state.stage에 완료(DONE)까지 반영한 값."""
    surface.fill(PANEL_BG)
    _camera(surface.subsurface(layout.camera), fonts, config, state, image)
    _panel(surface.subsurface(layout.panel), fonts, config, state, stage, scene)
    sim = surface.subsurface(layout.sim)
    render(sim, scene, layout.tf, config, None, hud=False)
    sim.blit(fonts.small.render(fonts.t("sim"), True, TEXT), (layout.tf.margin + 6, layout.tf.margin + 6))
    pygame.draw.line(surface, PANEL_BG, (0, layout.top), (layout.width, layout.top), 2)


def _camera(surface, fonts, config, state, image):
    surface.fill(CAMERA_BG)
    w, h = surface.get_size()
    iw, ih = image.get_size() if image else IMAGE_SIZE_PX
    k = min(w / iw, h / ih)
    ox, oy = (w - iw * k) / 2, (h - ih * k) / 2

    def px(p):
        return ox + p[0] * k, oy + p[1] * k

    to_px = road_to_image(config)
    if image:
        surface.blit(pygame.transform.smoothscale(image, (round(iw * k), round(ih * k))), (ox, oy))
    else:
        _synthetic_road(surface, config, state, to_px, px)
    view = state.view
    if view.zone:
        pygame.draw.polygon(surface, ZONE_COLOR, [px(p) for p in box_outline_px(view.zone.bbox, to_px)], 3)
    for b in view.boxes:
        color = BOX.get(b.kind, BOX["uncertain"])
        x1, y1 = px(b.bbox_px[:2])
        x2, y2 = px(b.bbox_px[2:])
        pygame.draw.rect(surface, color, (x1, y1, x2 - x1, y2 - y1), 1 if b.kind == "normal" else 2)
        name = f"{fonts.t(b.kind)} {b.confidence:.2f}"
        if b.track_id is not None:
            name = f"{b.track_id} {name}"
        label = fonts.small.render(name, True, (20, 20, 20), color)
        surface.blit(label, (x1, max(0, y1 - label.get_height())))
    surface.blit(fonts.small.render(f"{fonts.t('camera')}  ·  {fonts.t('frame')} {view.frame}", True, TEXT),
                 (10, 6))


def _synthetic_road(surface, config, state, to_px, px):
    """영상이 없을 때: 도로·차선·차량 바닥 사각형을 카메라 시점으로 옮겨 그린다."""
    road = config.road
    lo = min(a for a, _ in road.lanes.values())
    hi = max(b for _, b in road.lanes.values())

    def quad(box):
        return [px(p) for p in box_outline_px(box, to_px)]

    pygame.draw.polygon(surface, BG, quad((road.x_min_mm, lo, road.x_max_mm, hi)))
    s_lo, s_hi = road.lanes["shoulder"]
    pygame.draw.polygon(surface, SHOULDER, quad((road.x_min_mm, s_lo, road.x_max_mm, s_hi)))
    for line in road.lines:
        pts = [px(apply(to_px, p)) for p in line.points]
        pygame.draw.lines(surface, LINE, False, pts, 2 if line.kind == "solid" else 1)
    for v in state.view.vehicles:
        color = tuple(c * 2 // 3 for c in VEHICLE.get(v.label, BOX["uncertain"]))
        pygame.draw.polygon(surface, color, quad(v.bbox))


def _panel(surface, fonts, config, state, stage, scene):
    surface.fill(PANEL_BG)
    w, _ = surface.get_size()
    x, y = 22, 16
    view = state.view

    def row(text, color=TEXT, font=None, gap=6):
        nonlocal y
        img = (font or fonts.body).render(text, True, color)
        surface.blit(img, (x, y))
        y += img.get_height() + gap

    row(fonts.t("title"), font=fonts.big, gap=10)
    cx = x
    current = STAGES.index(stage) if stage in STAGES else 0
    for i, name in enumerate(STAGES):  # 단계 표시
        color = ACCENT if i == current else (TEXT if i < current else DIM)
        img = fonts.body.render(fonts.t(name), True, color)
        surface.blit(img, (cx, y))
        if i == current:
            pygame.draw.line(surface, ACCENT, (cx, y + img.get_height() + 2),
                             (cx + img.get_width(), y + img.get_height() + 2), 3)
        cx += img.get_width() + 12
        if i < len(STAGES) - 1:
            arrow = fonts.body.render(">", True, DIM)
            surface.blit(arrow, (cx, y))
            cx += arrow.get_width() + 12
    y += fonts.body.get_height() + 16

    row(f"{fonts.t('confirm')}  {view.streak} / {view.need} {fonts.t('frame')}", gap=4)
    bar = pygame.Rect(x, y, w - 2 * x, 8)
    pygame.draw.rect(surface, DIM, bar, 1)
    pygame.draw.rect(surface, ZONE_COLOR if view.zone else ACCENT,
                     (bar.x, bar.y, bar.width * min(1.0, view.streak / view.need), bar.height))
    y += 20

    for b in view.boxes:
        color = BOX.get(b.kind, BOX["uncertain"])
        pygame.draw.rect(surface, color, (x, y + 5, 12, 12))
        who = f"{fonts.t('vehicle')} {b.track_id}  " if b.track_id is not None else ""
        img = fonts.body.render(f"{who}{fonts.t(b.kind)}  {b.confidence:.2f}", True, TEXT)
        surface.blit(img, (x + 20, y))
        y += img.get_height() + 2
    y += 8

    if view.zone:
        x1, y1, x2, y2 = view.zone.bbox
        lanes = " ".join(fonts.t(n) for n in view.zone.lanes)
        row(f"{fonts.t('zone_info')}: {fonts.t(view.zone.kind)} · {lanes} · {x2 - x1:.0f} × {y2 - y1:.0f} mm",
            ZONE_COLOR)
    elif stage == NO_INCIDENT:
        row(fonts.t("no_zone"), WARN)
    if stage in (DEPLOY, DONE):
        hooked = max(0, len(scene.net_chain) - 2)
        posts = max(0, len(scene.robots) - 2)
        row(f"{fonts.t('deploy_info')}: {fonts.t(scene.phase)} · {fonts.t('hooked')} {hooked} / {posts}"
            f" · {scene.t:.0f} s", ACCENT)
    if stage != DETECT and scene.placement_warnings:
        row(f"{fonts.t('warn')}: {', '.join(scene.placement_warnings)}", WARN, fonts.small)
    for c in view.checks:
        kind, _, who = c.partition(":")
        what = f"{fonts.t('uncertain_vehicle')} {who}" if kind == "uncertain" else fonts.t(kind)
        row(f"{fonts.t('check')}: {what}", WARN)


def image_files(directory: str | Path | None) -> list[Path]:
    """영상 프레임 사진. 이름 순서가 Detection 프레임 순서와 같다고 본다 (vision.py의 runs/<name>/)."""
    if not directory:
        return []
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXT)


def scene_for(state: ShowState, stage: str, tick: dict | None, records: list[dict], config: Config,
              scenario: Scenario, plan) -> Scene:
    """탐지 중에는 그 프레임의 차량만, 범위 확정 뒤에는 범위·계획 경로, 전개 중에는 재생 중인 tick."""
    if stage in (DETECT, NO_INCIDENT):
        seen = replace(scenario, zone=None, vehicles=state.view.vehicles)
        return build_scene(standby_tick(config), [], config, seen)
    if tick is None:
        return build_scene(standby_tick(config), [], config, scenario, plan)
    return build_scene(tick, records, config, scenario, plan)


def run_present(scenario: Scenario, config: Config, backend: PlaybackBackend | None, plan, show: Show,
                log_path: str | Path, images: str | Path | None = None, speed: float = 2.0,
                width: int = 1200) -> None:
    """발표 화면을 실시간으로 재생한다. 끝나도 esc를 누를 때까지 마지막 화면을 유지한다.

    backend와 plan은 범위가 확정됐을 때의 전개 재생(없으면 None). speed는 전개 재생 배속.
    """
    pygame.init()
    layout = Layout.make(config, width)
    screen = pygame.display.set_mode(layout.size)
    pygame.display.set_caption(f"fencebot present - {scenario.name}")
    fonts = Fonts.load()
    files = image_files(images)
    cache: dict[int, pygame.Surface] = {}
    clock = pygame.time.Clock()
    t, acc, tick = 0.0, 0.0, None
    with LogWriter(log_path) as log:
        runner = Runner(scenario, config, backend, log) if backend else None
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    return
            dt = clock.tick(FPS) / 1000
            t += dt
            state = show.at(t)
            stage = state.stage
            if stage == DEPLOY and runner:
                acc += dt * speed
                while acc >= scenario.dt_s:
                    if not runner.done:
                        tick = runner.tick()
                    acc -= scenario.dt_s
                if runner.done:
                    stage = DONE
            if state.index < len(files) and state.index not in cache:
                cache.clear()
                cache[state.index] = pygame.image.load(str(files[state.index])).convert()
            scene = scene_for(state, stage, tick, log.records, config, scenario, plan)
            draw(screen, layout, fonts, config, state, stage, scene, cache.get(state.index))
            pygame.display.flip()
