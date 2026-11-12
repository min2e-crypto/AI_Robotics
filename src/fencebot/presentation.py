"""발표 화면의 진행 (D2, ADR-017). 화면 없이 계산만 한다.

Detection 프레임을 차례로 보여 주다가 사고 범위가 확정되면 멈추고, 범위를 잠깐 보여 준 뒤 전개로 넘어간다.
단계: detect(탐지) → zone(범위 확정) → deploy(전개) → done(완료). done은 전개 재생이 끝났을 때 화면 쪽에서 정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .detection_io import DetectionFrame
from .homography import Matrix, Point, apply, solve
from .incident import ACCIDENT_LABELS, FIRE_CLASS, ZoneTracker, frame_incident, road_homography
from .scenario import Vehicle
from .types import BBox, Zone

DETECT, ZONE, DEPLOY, DONE, NO_INCIDENT = "detect", "zone", "deploy", "done", "none"
STAGES = (DETECT, ZONE, DEPLOY, DONE)


@dataclass(frozen=True)
class Box:
    """영상 위에 그릴 탐지 1건. kind는 normal / damaged / overturned / uncertain / fire / smoke."""

    kind: str
    bbox_px: BBox
    confidence: float           # 차량은 상태 확률(없으면 탐지 확신도), 화염·연기는 탐지 확신도
    track_id: int | None = None


@dataclass(frozen=True)
class FrameView:
    """프레임 하나에서 화면에 보일 것."""

    frame: int
    boxes: tuple[Box, ...]
    vehicles: tuple[Vehicle, ...]   # 도로 좌표로 옮긴 차량 (시뮬레이터 쪽에 그림)
    checks: tuple[str, ...]         # 범위 계산에 쓰지 않고 관제자 확인으로 넘기는 것 (ADR-016)
    streak: int                     # 범위가 연속으로 나온 프레임 수
    need: int                       # 확정에 필요한 프레임 수
    zone: Zone | None               # 이 프레임까지 확정한 범위


def frame_views(frames: list[DetectionFrame], config: Config) -> list[FrameView]:
    h = road_homography(config)
    tracker = ZoneTracker(config)
    out = []
    for f in frames:
        inc = frame_incident(f, config, h)
        zone = tracker.update(inc)
        seen = [d for d in f.detections if d.confidence >= config.incident.min_confidence]
        states = {v.track_id: v for v in f.vehicle_states}
        boxes, checks = [], []
        for d in seen:
            if d.cls != "car":
                boxes.append(Box(d.cls, d.bbox_px, d.confidence))
                continue
            st = states.get(d.track_id)
            label = st.label if st else "normal"
            score = st.scores.get(label, d.confidence) if st else d.confidence
            boxes.append(Box(label, d.bbox_px, score, d.track_id))
            if label not in ACCIDENT_LABELS and label != "normal":
                checks.append(f"uncertain:{d.track_id}")
        if any(b.kind == "smoke" for b in boxes):
            checks.append("smoke")
        if any(b.kind == FIRE_CLASS for b in boxes) and (inc.zone is None or inc.zone.kind != "fire"):
            checks.append("fire_alone")
        streak, need = tracker.progress
        out.append(FrameView(f.frame, tuple(boxes), inc.vehicles, tuple(checks), streak, need, zone))
    return out


@dataclass(frozen=True)
class ShowState:
    stage: str
    index: int          # 보여 줄 프레임 순번
    view: FrameView


class Show:
    """시각 t(초)에 무엇을 보여 줄지. 프레임은 범위가 확정되는 프레임까지만 넘긴다."""

    def __init__(self, views: list[FrameView], frame_s: float = 0.4, zone_s: float = 2.5):
        if not views:
            raise ValueError("Detection 프레임이 없습니다")
        self.views = views
        self.frame_s, self.zone_s = frame_s, zone_s
        self.confirmed = next((i for i, v in enumerate(views) if v.zone is not None), None)
        self.last = len(views) - 1 if self.confirmed is None else self.confirmed
        self.detect_s = (self.last + 1) * frame_s

    @property
    def result(self) -> FrameView:
        """프레임을 멈추는 지점. 전개 계획은 이 프레임의 범위·차량으로 세운다."""
        return self.views[self.last]

    def at(self, t: float) -> ShowState:
        if t < self.detect_s:
            i = min(int(t / self.frame_s), self.last)
            return ShowState(DETECT, i, self.views[i])
        if self.confirmed is None:
            return ShowState(NO_INCIDENT, self.last, self.result)
        stage = ZONE if t < self.detect_s + self.zone_s else DEPLOY
        return ShowState(stage, self.last, self.result)


def road_to_image(config: Config) -> Matrix:
    return solve(list(config.camera.road_points_mm), list(config.camera.image_points_px))


def box_outline_px(box: BBox, to_px: Matrix) -> tuple[Point, ...]:
    """도로 좌표 박스의 네 꼭짓점을 영상 픽셀로 (영상 위에 범위 테두리를 그릴 때)."""
    x1, y1, x2, y2 = box
    return tuple(apply(to_px, p) for p in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))
