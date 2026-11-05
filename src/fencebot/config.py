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
    road: RoadConfig
    robots: tuple[RobotSpec, ...]
    standby: tuple[RobotState, ...]
    raw: dict  # 로그 헤더에 그대로 남기는 원본


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_PATH
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    road = raw["road"]
    return Config(
        net=NetConfig(**raw["net"]),
        robot=RobotConfig(**raw["robot"]),
        arrival=ArrivalConfig(**raw["arrival"]),
        sim=SimConfig(**raw["sim"]),
        margins=dict(raw["margins"]),
        placement=PlacementConfig(**raw["placement"]),
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
