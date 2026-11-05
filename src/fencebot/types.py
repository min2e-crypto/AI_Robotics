"""모듈 사이 데이터 형식 (구현_계획 4절, 배치_시뮬레이션_계획 2절).

좌표·각도 규약은 ADR-003: 도로 좌표 mm, 갓길선 y=0, 1차로 쪽 +y, heading 라디안(+x=0, 반시계 +).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    """그물에 꿰인 순서 그대로 (보관 → 기둥 1·2·3 → 당김)."""

    RESERVE = "reserve"
    POST1 = "post1"
    POST2 = "post2"
    POST3 = "post3"
    PULLER = "puller"


class Phase(str, Enum):
    """전체 상태 흐름 (구현_계획 4절). 전이 조건은 S6에서 구현."""

    STANDBY = "standby"
    ACCIDENT_DETECTED = "accident_detected"
    APPROVED = "approved"
    PULLER_DEPLOY = "puller_deploy"
    POSTS_ENTER = "posts_enter"
    HOLD = "hold"
    GATE_OPEN = "gate_open"
    RECOVER = "recover"


BBox = tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class Detection:
    """팀원 모델 출력 1건. 좌표는 이미지 픽셀."""

    frame: int
    timestamp: float
    cls: str
    confidence: float
    bbox_px: BBox
    track_id: int | None = None


@dataclass(frozen=True)
class VehicleState:
    """차량 박스별 상태. label은 normal / damaged / overturned. 애매 판정은 scores로 마스터 쪽에서."""

    track_id: int
    label: str
    scores: dict[str, float]
    stopped: bool | None = None


@dataclass(frozen=True)
class Zone:
    """사고 범위. bbox는 도로 좌표 (x_min, y_min, x_max, y_max), kind는 damaged / overturned / fire."""

    bbox: BBox
    lanes: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class RobotState:
    robot_id: int
    x: float
    y: float
    heading: float
    t: float = 0.0


@dataclass(frozen=True)
class Target:
    robot_id: int
    x: float
    y: float
    heading: float


def to_dict(obj) -> dict:
    return dataclasses.asdict(obj)


def from_dict(cls, data: dict):
    """to_dict의 역. JSON을 거치며 list가 된 tuple 필드를 되돌린다."""
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name])
    return cls(**kwargs)


def _coerce(tp, value):
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return None if value is None else _coerce(args[0], value)
    if origin is tuple and isinstance(value, list):
        return tuple(value)
    if isinstance(tp, type) and issubclass(tp, Enum):
        return tp(value)
    return value
