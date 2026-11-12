"""도로 4점 찍는 창 (fencebot calib).

사진 위에서 바닥의 네 점을 순서대로 누른다. 네 점을 다 찍으면 설정의 차선과 1m 눈금을 사진 위에 겹쳐 보여 준다.
겹친 선이 실제 차선과 맞으면 enter로 저장한다.

    왼쪽 클릭  점 찍기 (네 점을 찍은 뒤에는 가장 가까운 점을 옮김)
    backspace  마지막 점 지우기
    enter      저장하고 닫기 (네 점이 있고 호모그래피를 구할 수 있을 때)
    esc        저장하지 않고 닫기
"""

from __future__ import annotations

from pathlib import Path

import pygame

from ..calib import camera_toml, guide_lines, road_to_image_for
from ..config import Config
from ..homography import Point
from .present import ACCENT, Fonts
from .render import TEXT, WARN

MAX_SIZE = (1280, 800)
POINT = (255, 80, 80)
GUIDE = (80, 220, 255)
CORNERS = (("상류 · 갓길 쪽", "upstream, shoulder side"), ("하류 · 갓길 쪽", "downstream, shoulder side"),
           ("하류 · 1차로 쪽", "downstream, lane-1 side"), ("상류 · 1차로 쪽", "upstream, lane-1 side"))


def run_calib(image_path: str | Path, config: Config, road_points: tuple[Point, ...],
              out_path: str | Path) -> bool:
    """창을 띄워 4점을 받고 out_path에 [camera] 설정을 쓴다. 저장했으면 True."""
    pygame.init()
    image = pygame.image.load(str(image_path))
    iw, ih = image.get_size()
    k = min(MAX_SIZE[0] / iw, MAX_SIZE[1] / ih, 1.0)
    bar = 56
    screen = pygame.display.set_mode((round(iw * k), round(ih * k) + bar))
    pygame.display.set_caption(f"fencebot calib - {Path(image_path).name}")
    shown = pygame.transform.smoothscale(image.convert(), (round(iw * k), round(ih * k)))
    fonts = Fonts.load()
    lang = 0 if fonts.korean else 1
    points: list[Point] = []  # 영상 픽셀 (원본 크기 기준)
    clock = pygame.time.Clock()
    error = None  # 네 점으로 호모그래피를 못 구하면 그 이유. 있으면 저장하지 않는다
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_BACKSPACE and points:
                points.pop()
            saving = ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER)
            if saving and len(points) == 4 and not error:
                out = Path(out_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(camera_toml(points, list(road_points), Path(image_path).name), encoding="utf-8")
                return True
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and ev.pos[1] < ih * k:
                p = (ev.pos[0] / k, ev.pos[1] / k)
                if len(points) < 4:
                    points.append(p)
                else:
                    near = min(range(4), key=lambda i: (points[i][0] - p[0]) ** 2 + (points[i][1] - p[1]) ** 2)
                    points[near] = p
        screen.fill((24, 26, 29))
        screen.blit(shown, (0, 0))
        error = _draw_guides(screen, config, points, road_points, (iw, ih), k)
        for i, (x, y) in enumerate(points):
            pygame.draw.circle(screen, POINT, (x * k, y * k), 6, 2)
            pygame.draw.circle(screen, POINT, (x * k, y * k), 1)
            rx, ry = road_points[i]
            label = fonts.small.render(f"{i + 1} ({rx:g}, {ry:g})", True, (20, 20, 20), POINT)
            screen.blit(label, (x * k + 9, y * k - label.get_height() - 2))
        if len(points) < 4:
            rx, ry = road_points[len(points)]
            line1 = (f"{len(points) + 1}/4  {CORNERS[len(points)][lang]}  "
                     f"→ {'도로 좌표' if lang == 0 else 'road'} ({rx:g}, {ry:g}) mm")
            color = TEXT
        elif error:
            line1, color = error, WARN
        else:
            line1 = ("겹친 선이 차선과 맞으면 enter로 저장. 클릭하면 가까운 점을 옮김" if lang == 0
                     else "enter to save if the overlay matches the lanes. click moves the nearest point")
            color = ACCENT
        line2 = ("backspace 지우기 · enter 저장 · esc 닫기" if lang == 0
                 else "backspace undo · enter save · esc quit")
        screen.blit(fonts.body.render(line1, True, color), (10, ih * k + 4))
        screen.blit(fonts.small.render(line2, True, (150, 154, 160)), (10, ih * k + 32))
        pygame.display.flip()
        clock.tick(30)


def _draw_guides(screen, config, points, road_points, image_size, k) -> str | None:
    """네 점이 있으면 차선·눈금을 겹쳐 그린다. 호모그래피를 못 구하면 그 이유를 돌려준다."""
    if len(points) < 4:
        if len(points) > 1:
            pygame.draw.lines(screen, POINT, False, [(x * k, y * k) for x, y in points], 1)
        return None
    try:
        to_px = road_to_image_for(points, list(road_points))
    except ValueError as e:
        return str(e)
    for line in guide_lines(config.road, to_px, image_size, road_points[0]):
        pygame.draw.lines(screen, GUIDE, False, [(x * k, y * k) for x, y in line], 1)
    pygame.draw.polygon(screen, POINT, [(x * k, y * k) for x, y in points], 1)
    return None
