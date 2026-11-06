"""그릴 내용 정리. 실행 중인 tick 레코드와 로그에서 읽은 tick 레코드에서 같은 Scene을 만든다."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import Config, LaneLine, RoadConfig
from ..scenario import Scenario
from ..types import BBox

FLASH_S = 1.0  # 백엔드 이벤트(net_limit 등)를 상태 표시줄에 강조하는 시간


@dataclass(frozen=True)
class RobotView:
    robot_id: int
    role: str
    x: float
    y: float
    heading: float
    seen: tuple[float, float, float, float]  # 카메라 추정 (x, y, heading, t)
    comm_ok: bool


@dataclass(frozen=True)
class Scene:
    t: float
    phase: str
    robots: tuple[RobotView, ...]  # 그물에 꿰인 순서
    segments: tuple[float, ...]    # 보관→기둥1, 기둥1→2, 2→3, 3→당김
    net_length: float
    unwound: float
    reel: str
    clamp: str
    camera_ok: bool
    flashes: tuple[str, ...]
    zone: BBox | None
    vehicles: tuple[tuple[str, BBox], ...]
    lines: tuple[LaneLine, ...]

    @property
    def slack(self) -> float:
        """처짐 = 풀린 길이 - 꺾은선 길이 (릴 장력 추정값 대신, ADR-008)."""
        return self.unwound - self.net_length


def link_status(records: list[dict], t: float) -> tuple[bool, frozenset[int], tuple[str, ...]]:
    """시각 t까지의 이벤트로 (카메라 정상, 통신 끊긴 로봇, 최근 백엔드 이벤트)를 구한다."""
    camera_ok, comm_down, flashes = True, set(), []
    for e in records:
        if e["type"] != "event" or e["t"] > t + 1e-9:
            continue
        kind, data = e["kind"], e["data"]
        if kind in ("camera_loss", "camera_restore"):
            camera_ok = kind == "camera_restore"
        elif kind in ("comm_loss", "comm_restore"):
            ids = {data["robot_id"]} if "robot_id" in data else {-1}
            comm_down = comm_down | ids if kind == "comm_loss" else comm_down - ids
        elif kind == "net_limit" and t - e["t"] < FLASH_S:
            flashes.append(f"net_limit {data['robot_id']}")
    return camera_ok, frozenset(comm_down), tuple(flashes)


def build_scene(tick: dict, records: list[dict], config: Config, scenario: Scenario) -> Scene:
    t = tick["t"]
    camera_ok, comm_down, flashes = link_status(records, t)
    roles = {r.robot_id: r.role.value for r in config.robots}
    seen = {r["robot_id"]: r for r in tick["robots"]}
    truth = tick.get("truth") or tick["robots"]  # truth가 없는 로그(StaticBackend)는 추정값으로
    robots = tuple(
        RobotView(p["robot_id"], roles[p["robot_id"]], p["x"], p["y"], p["heading"],
                  tuple(seen[p["robot_id"]][k] for k in ("x", "y", "heading", "t")),
                  p["robot_id"] not in comm_down and -1 not in comm_down)
        for p in truth
    )
    pts = [(r.x, r.y) for r in robots]
    segments = tuple(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    net = tick.get("net", {})
    return Scene(
        t=t, phase=tick["phase"], robots=robots, segments=segments,
        net_length=sum(segments), unwound=net.get("unwound_mm", sum(segments)),
        reel=net.get("reel", "-"), clamp=net.get("clamp", "-"),
        camera_ok=camera_ok, flashes=flashes,
        zone=scenario.zone.bbox if scenario.zone else None,
        vehicles=tuple((v.label, v.bbox) for v in scenario.vehicles),
        lines=config.road.lines,
    )


@dataclass(frozen=True)
class ScreenTransform:
    """도로 좌표(mm) → 화면 픽셀. 화면은 y가 아래로 커지므로 뒤집는다 (+y 1차로 쪽이 위)."""

    x_min: float
    y_max: float
    scale: float  # px / mm
    margin: float
    width_px: float
    height_px: float

    @classmethod
    def for_road(cls, road: RoadConfig, width_px: int, margin_px: int = 20) -> ScreenTransform:
        y_min = min(lo for lo, _ in road.lanes.values())
        y_max = max(hi for _, hi in road.lanes.values())
        scale = (width_px - 2 * margin_px) / (road.x_max_mm - road.x_min_mm)
        return cls(road.x_min_mm, y_max, scale, margin_px, width_px,
                   2 * margin_px + (y_max - y_min) * scale)

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        return (self.margin + (x - self.x_min) * self.scale,
                self.margin + (self.y_max - y) * self.scale)

    def length(self, mm: float) -> float:
        return mm * self.scale
