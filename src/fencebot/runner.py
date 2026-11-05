"""시나리오 실행 루프. 시간 진행·이벤트·위치 기록만 하고, 제어는 S6에서 붙인다."""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .io import PoseSource, RobotLink
from .log import LOG_VERSION, LogWriter
from .scenario import Event, Scenario
from .types import Phase, to_dict


class Backend(RobotLink, PoseSource):
    def handle(self, event: Event) -> None: ...
    def step(self, dt: float) -> None: ...
    def pop_events(self) -> list[dict]:
        """step 중 백엔드에서 생긴 이벤트 ({"t", "kind", "data"})."""
        ...
    def log_fields(self) -> dict:
        """tick 레코드에 더할 값 (시뮬레이터의 실제 위치·그물 상태)."""
        ...


def run(scenario: Scenario, config: Config, backend: Backend, log_path: str | Path) -> list[dict]:
    steps = round(scenario.duration_s / scenario.dt_s)
    pending = list(scenario.events)
    phase = Phase.STANDBY
    with LogWriter(log_path) as log:
        log.write({"type": "header", "version": LOG_VERSION,
                   "scenario": scenario.raw, "config": config.raw})
        for i in range(steps + 1):
            t = round(i * scenario.dt_s, 9)
            while pending and pending[0].t <= t + 1e-9:
                e = pending.pop(0)
                log.write({"type": "event", "t": e.t, "kind": e.kind, "data": e.data})
                backend.handle(e)
            for e in backend.pop_events():
                log.write({"type": "event", **e})
            robots = backend.read(t)
            log.write({"type": "tick", "t": t, "phase": phase.value,
                       "robots": [to_dict(r) for r in robots], **backend.log_fields()})
            if i < steps:
                backend.step(scenario.dt_s)
        return log.records
