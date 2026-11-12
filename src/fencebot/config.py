"""설정 파일 읽기. 값의 의미와 임시 여부는 config/default.toml 주석 참고."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .types import Role, RobotState

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.toml"


@dataclass(frozen=True)
class NetConfig:
    length_mm: float
    max_gap_mm: float
    taper_length_mm: float
    hook_tol_mm: float


@dataclass(frozen=True)
class RobotConfig:
    wheelbase_mm: float
    min_turn_radius_mm: float
    radius_mm: float
    max_speed_mm_s: float
    steer_delay_s: float


@dataclass(frozen=True)
class ArrivalConfig:
    position_mm: float
    heading_rad: float
    settle_s: float


@dataclass(frozen=True)
class SimConfig:
    camera_latency_s: float
    pos_noise_mm: float
    heading_noise_rad: float
    occlusion_prob: float
    watchdog_hold_s: float
    watchdog_stop_s: float


@dataclass(frozen=True)
class PlacementConfig:
    upstream_margin_mm: float
    downstream_margin_mm: float


@dataclass(frozen=True)
class PlanningConfig:
    clearance_mm: float
    step_mm: float
    stage_gap_mm: float
    strict: bool = True  # False면 발표용: 검사 실패를 경고로 돌리고 전개를 계속한다 (ADR-014)


@dataclass(frozen=True)
class CameraConfig:
    """영상 픽셀 ↔ 도로 좌표 4점 대응 (ADR-015). 같은 순서의 점끼리 짝이다."""

    image_points_px: tuple[tuple[float, float], ...]
    road_points_mm: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class IncidentConfig:
    min_confidence: float
    near_mm: float
    touch_mm: float
    confirm_frames: int
    vehicle_size_mm: tuple[float, float]


@dataclass(frozen=True)
class LaneLine:
    name: str
    kind: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RoadConfig:
    x_min_mm: float
    x_max_mm: float
    lanes: dict[str, tuple[float, float]]  # 이름 → (y_min, y_max)
    lines: tuple[LaneLine, ...]

    def lane_at(self, y: float) -> str | None:
        for name, (y_min, y_max) in self.lanes.items():
            if y_min <= y < y_max:
                return name
        return None


@dataclass(frozen=True)
class RobotSpec:
    robot_id: int
    role: Role


@dataclass(frozen=True)
class Config:
    net: NetConfig
    robot: RobotConfig
    arrival: ArrivalConfig
    sim: SimConfig
    margins: dict[str, float]
    placement: PlacementConfig
    planning: PlanningConfig
    camera: CameraConfig
    incident: IncidentConfig
    road: RoadConfig
    robots: tuple[RobotSpec, ...]
    standby: tuple[RobotState, ...]
    raw: dict  # 로그 헤더에 그대로 남기는 원본


def load_config(path: str | Path | None = None) -> Config:
    """기본 설정 위에 path의 값을 덮어쓴다. path에는 바꿀 값만 있으면 된다 (예: 영상별 [camera]).

    표([section])는 키 단위로 합치고, 그 밖의 값(lanes·robots 같은 목록 포함)은 통째로 바꾼다.
    """
    raw = tomllib.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    if path:
        for key, value in tomllib.loads(Path(path).read_text(encoding="utf-8")).items():
            both_tables = isinstance(value, dict) and isinstance(raw.get(key), dict)
            raw[key] = {**raw[key], **value} if both_tables else value
    return config_from_raw(raw)


def config_from_raw(raw: dict) -> Config:
    """로그 헤더의 config로도 같은 Config를 만든다."""
    road = raw["road"]
    return Config(
        net=NetConfig(**raw["net"]),
        robot=RobotConfig(**raw["robot"]),
        arrival=ArrivalConfig(**raw["arrival"]),
        sim=SimConfig(**raw["sim"]),
        margins=dict(raw["margins"]),
        placement=PlacementConfig(**raw["placement"]),
        planning=PlanningConfig(**raw["planning"]),
        camera=CameraConfig(*(tuple(tuple(p) for p in raw["camera"][k])
                              for k in ("image_points_px", "road_points_mm"))),
        incident=IncidentConfig(**{**raw["incident"],
                                   "vehicle_size_mm": tuple(raw["incident"]["vehicle_size_mm"])}),
        road=RoadConfig(
            x_min_mm=road["x_min_mm"],
            x_max_mm=road["x_max_mm"],
            lanes={l["name"]: (l["y_min_mm"], l["y_max_mm"]) for l in road["lanes"]},
            lines=tuple(
                LaneLine(l["name"], l["kind"], tuple(tuple(p) for p in l["points"]))
                for l in road["lines"]
            ),
        ),
        robots=tuple(RobotSpec(r["robot_id"], Role(r["role"])) for r in raw["robots"]),
        standby=tuple(RobotState(**s) for s in raw["standby"]),
        raw=raw,
    )
