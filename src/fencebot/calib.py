"""도로 4점 보정 (fencebot calib). 화면 없이 계산하는 부분.

사진에서 바닥의 네 점을 찍어 `[camera]` 설정을 만든다. 만든 파일은 `--config`로 준다.
찍는 순서는 도로 좌표 사각형의 (x0, y0) → (x1, y0) → (x1, y1) → (x0, y1)이다.
x는 진행 방향, y는 갓길선에서 1차로 쪽이므로(ADR-003) 상류 갓길 쪽 → 하류 갓길 쪽 → 하류 1차로 쪽 → 상류 1차로 쪽이다.
"""

from __future__ import annotations

from .config import Config, RoadConfig
from .homography import Matrix, Point, solve

Rect = tuple[float, float, float, float]


def parse_rect(text: str) -> Rect:
    """"x0,y0,x1,y1" (도로 좌표 mm) → 사각형."""
    try:
        x0, y0, x1, y1 = (float(v) for v in text.split(","))
    except ValueError:
        raise ValueError(f"사각형은 x0,y0,x1,y1 형식입니다: {text!r}") from None
    if x0 == x1 or y0 == y1:
        raise ValueError(f"사각형의 폭이나 길이가 0입니다: {text!r}")
    return x0, y0, x1, y1


def rect_points(rect: Rect) -> tuple[Point, ...]:
    x0, y0, x1, y1 = rect
    return (x0, y0), (x1, y0), (x1, y1), (x0, y1)


def camera_toml(image_points: list[Point], road_points: list[Point], source: str = "") -> str:
    """`--config`로 줄 수 있는 설정 조각. 기본 설정 위에 [camera]만 덮어쓴다."""
    if len(image_points) != 4 or len(road_points) != 4:
        raise ValueError("영상 4점과 도로 4점이 필요합니다")

    def pts(points):
        return "[" + ", ".join(f"[{x:g}, {y:g}]" for x, y in points) + "]"

    head = f"# fencebot calib으로 찍은 도로 4점{f' ({source})' if source else ''}. 기본 설정 위에 덮어쓴다\n"
    return (f"{head}[camera]\n"
            f"image_points_px = {pts([(round(x, 1), round(y, 1)) for x, y in image_points])}\n"
            f"road_points_mm = {pts(road_points)}\n")


def guide_lines(road: RoadConfig, to_px: Matrix, image_size: tuple[int, int], ref: Point,
                grid_mm: float = 1000.0, samples: int = 40) -> list[list[Point]]:
    """찍은 4점이 맞는지 보려고 영상 위에 겹쳐 그릴 선: 차선과 진행 방향 눈금(grid_mm 간격).

    ref는 영상에 보이는 도로 위의 한 점(찍은 점 가운데 하나)이다. 호모그래피의 분모 부호가 ref와 다른 점은
    카메라 뒤쪽이므로 뺀다. 영상에서 아주 먼 점도 빼고, 그 자리에서 선을 끊는다.
    """
    front = 1.0 if _w(to_px, ref) > 0 else -1.0
    y_lo = min(lo for lo, _ in road.lanes.values())
    y_hi = max(hi for _, hi in road.lanes.values())
    lines = [list(line.points) for line in road.lines]
    x = road.x_min_mm
    while x <= road.x_max_mm:
        lines.append([(x, y_lo), (x, y_hi)])
        x += grid_mm
    w, h = image_size
    out: list[list[Point]] = []
    for line in lines:
        piece: list[Point] = []
        for a, b in zip(line, line[1:]):
            for i in range(samples + 1):
                p = (a[0] + (b[0] - a[0]) * i / samples, a[1] + (b[1] - a[1]) * i / samples)
                q = _project(to_px, p, front)
                if q and -3 * w <= q[0] <= 4 * w and -3 * h <= q[1] <= 4 * h:
                    piece.append(q)
                else:
                    if len(piece) > 1:
                        out.append(piece)
                    piece = []
        if len(piece) > 1:
            out.append(piece)
    return out


def road_to_image_for(image_points: list[Point], road_points: list[Point]) -> Matrix:
    return solve(list(road_points), list(image_points))


def default_road_points(config: Config, rect: Rect | None) -> tuple[Point, ...]:
    """--rect를 주지 않으면 설정 [camera]의 도로 4점을 그대로 쓴다."""
    return rect_points(rect) if rect else tuple(config.camera.road_points_mm)


def _w(h: Matrix, p: Point) -> float:
    return h[6] * p[0] + h[7] * p[1] + h[8]


def _project(h: Matrix, p: Point, front: float) -> Point | None:
    w = _w(h, p)
    if w * front <= 1e-9:
        return None
    x, y = p
    return (h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w
