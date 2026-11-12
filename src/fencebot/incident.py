"""사고 범위 계산 (구현_계획 6절, 배치_시뮬레이션_계획 6단계, ADR-015).

프레임별 Detection(픽셀) → 차량의 도로 좌표 박스 → 사고 차량 묶기 → 형태 판정 → 범위(Zone) → 시간 안정화.
발표용 최소 규칙이다. 멈춘 차량 후보, 사고 건 선택 화면, 범위 축소 반영은 아직 없다.
묶기와 화재 규칙은 팀원이 시험하며 정한 것(share_model/정해둔_규칙.md)을 따른다 (ADR-016).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Config
from .detection_io import DetectionFrame
from .homography import Matrix, Point, apply, solve
from .scenario import Vehicle
from .types import BBox, Zone

ACCIDENT_LABELS = ("damaged", "overturned")
FIRE_CLASS = "fire"  # 연기(smoke)는 오탐이 많아 범위 계산에 쓰지 않는다 (ADR-016)
SEVERITY = {"damaged": 0, "overturned": 1, "fire": 2}  # 우선순위: 화재 > 전복 > 파손


@dataclass(frozen=True)
class Incident:
    """프레임 하나의 판정. vehicles는 도로 좌표로 옮긴 모든 차량(정상 차량은 장애물로 쓴다)."""

    zone: Zone | None
    vehicles: tuple[Vehicle, ...]


@dataclass(frozen=True)
class IncidentResult:
    zone: Zone | None                # 시간 안정화를 거쳐 확정한 범위
    vehicles: tuple[Vehicle, ...]    # 차량이 보인 마지막 프레임 기준
    confirmed_frame: int | None      # 처음 확정한 프레임의 순번 (0부터)


def road_homography(config: Config) -> Matrix:
    return solve(list(config.camera.image_points_px), list(config.camera.road_points_mm))


def vehicle_box(bbox_px: BBox, h: Matrix, size_mm: tuple[float, float]) -> BBox:
    """픽셀 박스 → 도로 위 차량 박스.

    비스듬한 뷰에서 박스 윗부분은 지붕·옆면이라 바닥에 옮기면 멀리 밀린다. 그래서 박스 아랫변만
    바닥 접점(카메라에 가까운 쪽 변)으로 보고, 카메라에서 먼 쪽으로 설정의 차량 크기만큼 채운다.
    """
    x1, y1, x2, y2 = bbox_px
    a, b = apply(h, (x1, y2)), apply(h, (x2, y2))
    far = apply(h, ((x1 + x2) / 2, y1))  # 먼 쪽 방향을 알아내는 데만 쓴다
    length, width = size_mm
    if abs(b[0] - a[0]) >= abs(b[1] - a[1]):  # 아랫변이 진행 방향: 폭을 채운다
        near = (a[1] + b[1]) / 2
        other = near + math.copysign(width, far[1] - near)
        return (min(a[0], b[0]), min(near, other), max(a[0], b[0]), max(near, other))
    near = (a[0] + b[0]) / 2  # 아랫변이 폭 방향: 길이를 채운다
    other = near + math.copysign(length, far[0] - near)
    return (min(near, other), min(a[1], b[1]), max(near, other), max(a[1], b[1]))


def frame_incident(frame: DetectionFrame, config: Config, h: Matrix | None = None) -> Incident:
    h = h or road_homography(config)
    inc = config.incident
    seen = [d for d in frame.detections if d.confidence >= inc.min_confidence]
    labels = {v.track_id: v.label for v in frame.vehicle_states}
    vehicles = tuple(
        Vehicle(d.track_id, labels.get(d.track_id, "normal"),
                vehicle_box(d.bbox_px, h, tuple(inc.vehicle_size_mm)))
        for d in seen if d.cls == "car" and d.track_id is not None
    )
    hazards = tuple(apply(h, ((d.bbox_px[0] + d.bbox_px[2]) / 2, d.bbox_px[3]))
                    for d in seen if d.cls == FIRE_CLASS)
    return Incident(zone_from_vehicles(vehicles, hazards, config), vehicles)


def zone_from_vehicles(vehicles: tuple[Vehicle, ...], hazards: tuple[Point, ...],
                       config: Config) -> Zone | None:
    """도로 좌표의 차량과 화염 위치에서 사고 범위를 구한다. 사고가 없으면 None.

    hazards는 화염 박스의 바닥 접점이다. 사고가 여러 건이면 가장 심한 것(같으면 상류 쪽)을 고른다.
    - 사고 후보는 파손·전복 차량뿐이다. 정상·애매 차량 옆의 화염만으로는 범위를 만들지 않는다.
    - 후보끼리 차량 길이 안에 있으면 같은 사고 건이다.
    - 후보와 맞닿은 차량은 상태와 상관없이 묶되 한 단계만 묶는다 (정체 때 줄줄이 묶이지 않게).
    """
    inc = config.incident
    same_mm = inc.vehicle_size_mm[0]
    candidates = [v for v in vehicles if v.label in ACCIDENT_LABELS]
    others = [v for v in vehicles if v.label not in ACCIDENT_LABELS]
    groups = []
    while candidates:
        core = [candidates.pop(0)]
        grew = True
        while grew:
            grew = False
            for v in candidates[:]:
                if any(_gap(v.bbox, g.bbox) < same_mm for g in core):
                    core.append(v)
                    candidates.remove(v)
                    grew = True
        group = core + [v for v in others if any(_gap(v.bbox, g.bbox) < inc.touch_mm for g in core)]
        kinds = [v.label for v in core]
        kinds += ["fire"] * any(_dist(p, v.bbox) < inc.near_mm for p in hazards for v in group)
        groups.append((max(kinds, key=SEVERITY.get), group))
    if not groups:
        return None
    kind, group = max(groups, key=lambda g: (SEVERITY[g[0]], -min(v.bbox[0] for v in g[1])))

    boxes = [v.bbox for v in group]
    near = [p for p in hazards if any(_dist(p, b) < inc.near_mm for b in boxes)]
    x1 = min([b[0] for b in boxes] + [p[0] for p in near])
    y1 = min([b[1] for b in boxes] + [p[1] for p in near])
    x2 = max([b[2] for b in boxes] + [p[0] for p in near])
    y2 = max([b[3] for b in boxes] + [p[1] for p in near])
    lanes = tuple(name for name, (lo, hi) in config.road.lanes.items() if y1 < hi and y2 > lo)
    lo = min(config.road.lanes[n][0] for n in lanes)
    hi = max(config.road.lanes[n][1] for n in lanes)
    m = config.margins[kind]
    # 폭 방향 여유는 차로 경계까지 남은 거리의 절반까지만: 경계선 위에 로봇이 설 자리를 남긴다
    return Zone((x1 - m, y1 - min(m, max(0.0, y1 - lo) / 2),
                 x2 + m, y2 + min(m, max(0.0, hi - y2) / 2)), lanes, kind)


class ZoneTracker:
    """시간 안정화. confirm_frames 프레임 연속으로 범위가 나오면 확정한다.

    확정 뒤에는 커지는 쪽만 바로 반영하고 줄이지 않는다. 형태도 더 심한 쪽으로만 바뀐다.
    """

    def __init__(self, config: Config):
        self._need = config.incident.confirm_frames
        self._streak = 0
        self._pending: Zone | None = None
        self.zone: Zone | None = None

    @property
    def progress(self) -> tuple[int, int]:
        """(범위가 연속으로 나온 프레임 수, 확정에 필요한 수). 확정 뒤에는 둘이 같다."""
        return (self._need if self.zone is not None else self._streak), self._need

    def update(self, incident: Incident) -> Zone | None:
        z = incident.zone
        if self.zone is not None:
            if z is not None:
                self.zone = _union(self.zone, z)
            return self.zone
        if z is None:
            self._streak, self._pending = 0, None
            return None
        self._streak += 1
        self._pending = z if self._pending is None else _union(self._pending, z)
        if self._streak >= self._need:
            self.zone = self._pending
        return self.zone


def incident_from_frames(frames: list[DetectionFrame], config: Config) -> IncidentResult:
    h = road_homography(config)
    tracker = ZoneTracker(config)
    vehicles: tuple[Vehicle, ...] = ()
    confirmed = None
    for i, frame in enumerate(frames):
        inc = frame_incident(frame, config, h)
        vehicles = inc.vehicles or vehicles
        if tracker.update(inc) is not None and confirmed is None:
            confirmed = i
    return IncidentResult(tracker.zone, vehicles, confirmed)


def _union(a: Zone, b: Zone) -> Zone:
    box = (min(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1]),
           max(a.bbox[2], b.bbox[2]), max(a.bbox[3], b.bbox[3]))
    lanes = a.lanes + tuple(n for n in b.lanes if n not in a.lanes)
    return Zone(box, lanes, max(a.kind, b.kind, key=SEVERITY.get))


def _dist(p: Point, box: BBox) -> float:
    return math.hypot(max(box[0] - p[0], 0.0, p[0] - box[2]), max(box[1] - p[1], 0.0, p[1] - box[3]))


def _gap(a: BBox, b: BBox) -> float:
    return math.hypot(max(a[0] - b[2], 0.0, b[0] - a[2]), max(a[1] - b[3], 0.0, b[1] - a[3]))
