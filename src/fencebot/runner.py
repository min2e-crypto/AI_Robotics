"""시나리오 실행 루프. S1은 시간 진행·이벤트·위치 기록만 하고, 제어는 S6에서 붙인다."""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .io import PoseSource, RobotLink
from .log import LOG_VERSION, LogWriter
from .scenario import Scenario
from .types import Phase, to_dict


class Backend(RobotLink, PoseSource):
    pass


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
            robots = backend.read(t)
            log.write({"type": "tick", "t": t, "phase": phase.value,
                       "robots": [to_dict(r) for r in robots]})
        return log.records
