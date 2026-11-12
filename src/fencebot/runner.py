"""시나리오 실행 루프. 시간 진행·이벤트·위치 기록만 하고, 제어는 S6에서 붙인다.

Runner는 한 tick씩 진행하므로 일괄 실행(run)과 수동 조작 화면(view.app)이 같이 쓴다.
"""

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


class Runner:
    def __init__(self, scenario: Scenario, config: Config, backend: Backend, log: LogWriter):
        self.scenario, self.backend, self.log = scenario, backend, log
        self.steps = round(scenario.duration_s / scenario.dt_s)
        self.i = 0
        self.phase = Phase.STANDBY
        self._pending = list(scenario.events)
        log.write({"type": "header", "version": LOG_VERSION,
                   "scenario": scenario.raw, "config": config.raw})

    @property
    def t(self) -> float:
        return round(self.i * self.scenario.dt_s, 9)

    @property
    def done(self) -> bool:
        return self.i > self.steps

    def tick(self) -> dict:
        """시각 t의 이벤트 처리 → 위치 기록 → dt 진행. 기록한 tick 레코드를 돌려준다."""
        t, log, backend = self.t, self.log, self.backend
        self.phase = Phase(getattr(backend, "phase", self.phase))  # 재생 백엔드는 단계를 스스로 안다
        while self._pending and self._pending[0].t <= t + 1e-9:
            e = self._pending.pop(0)
            log.write({"type": "event", "t": e.t, "kind": e.kind, "data": e.data})
            backend.handle(e)
        for e in backend.pop_events():
            log.write({"type": "event", **e})
        robots = backend.read(t)
        log.write({"type": "tick", "t": t, "phase": self.phase.value,
                   "robots": [to_dict(r) for r in robots], **backend.log_fields()})
        if self.i < self.steps:
            backend.step(self.scenario.dt_s)
        self.i += 1
        return log.records[-1]


def run(scenario: Scenario, config: Config, backend: Backend, log_path: str | Path) -> list[dict]:
    with LogWriter(log_path) as log:
        runner = Runner(scenario, config, backend, log)
        while not runner.done:
            runner.tick()
        return log.records
