"""시나리오 파일 읽기. 예: scenarios/lane2_basic.toml"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .types import BBox, Zone

# S2 이후 시뮬레이터가 처리하는 이벤트 종류
EVENT_KINDS = {"comm_loss", "comm_restore", "camera_loss", "camera_restore",
               "obstacle", "emergency_vehicle"}


@dataclass(frozen=True)
class Vehicle:
    """시나리오의 차량 모형. bbox는 도로 좌표, label은 normal / damaged / overturned."""

    track_id: int
    label: str
    bbox: BBox


@dataclass(frozen=True)
class Event:
    t: float
    kind: str
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    duration_s: float
    dt_s: float
    zone: Zone | None
    vehicles: tuple[Vehicle, ...]
    events: tuple[Event, ...]
    raw: dict


def load_scenario(path: str | Path) -> Scenario:
    return scenario_from_raw(tomllib.loads(Path(path).read_text(encoding="utf-8")))


def scenario_from_raw(raw: dict) -> Scenario:
    """로그 헤더의 scenario로도 같은 Scenario를 만든다."""
    z = raw.get("zone")
    events = []
    for e in raw.get("events", []):
        if e["kind"] not in EVENT_KINDS:
            raise ValueError(f"알 수 없는 이벤트 종류: {e['kind']}")
        data = {k: v for k, v in e.items() if k not in ("t", "kind")}
        events.append(Event(e["t"], e["kind"], data))
    return Scenario(
        name=raw["name"],
        seed=raw.get("seed", 0),
        duration_s=raw["duration_s"],
        dt_s=raw["dt_s"],
        zone=Zone(tuple(z["bbox"]), tuple(z["lanes"]), z["kind"]) if z else None,
        vehicles=tuple(Vehicle(v["track_id"], v["label"], tuple(v["bbox"]))
                       for v in raw.get("vehicles", [])),
        events=tuple(sorted(events, key=lambda e: e.t)),
        raw=raw,
    )
