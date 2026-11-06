"""2D 시뮬레이터 코어 (화면 없음). RobotLink·PoseSource를 구현해 StaticBackend 자리에 들어간다.

모델 (배치_시뮬레이션_계획 5절, ADR-007)
- 차체: 자전거 모델. 위치 기준점은 뒷바퀴 축 가운데. 조향은 1차 지연, 속도는 명령 즉시 반영.
- 그물: 보관 → 기둥1·2·3 → 당김 꺾은선 길이. 릴이 장력을 유지한다고 보고, 길이 제약을 어기는
  로봇 이동은 그 스텝만 막는다(다른 로봇을 끌지 않음).
- 카메라: 지연·잡음·가림. 가려진 로봇과 카메라 끊김 중에는 마지막으로 본 위치(옛 t)를 돌려준다.
- 와치독: 마지막 명령 수신 후 watchdog_hold_s까지 유지, 이후 감속 정지.
"""

from __future__ import annotations

import math
import random

from .config import Config
from .io import ActionCommand, Command, DriveCommand
from .scenario import Event
from .types import RobotState

_EPS = 1e-9


class _Robot:
    def __init__(self, s: RobotState):
        self.x, self.y, self.heading = s.x, s.y, s.heading
        self.speed = 0.0
        self.steer = 0.0
        self.cmd_speed = 0.0
        self.cmd_steer = 0.0
        self.last_rx = 0.0
        self.comm_ok = True


class SimBackend:
    def __init__(self, config: Config, seed: int = 0):
        self.cfg = config
        self.time = 0.0
        self._order = [r.robot_id for r in config.robots]  # 그물에 꿰인 순서
        start = {s.robot_id: s for s in config.standby}
        self._robots = {rid: _Robot(start[rid]) for rid in self._order}
        self._max_steer = math.atan(config.robot.wheelbase_mm / config.robot.min_turn_radius_mm)
        self._decel = config.robot.max_speed_mm_s / config.sim.watchdog_stop_s
        self._reel = "unwind"
        self._clamp = "closed"
        self._unwound = self.net_length()
        self._blocked: set[int] = set()
        self._events: list[dict] = []
        self._rng = random.Random(seed)
        self._camera_ok = True
        self._history = [(0.0, self.truth())]
        self._last_seen = {r.robot_id: r for r in self.truth()}

    # --- RobotLink ---
    def send(self, cmd: Command) -> None:
        r = self._robots[cmd.robot_id]
        if not r.comm_ok:
            return
        r.last_rx = self.time
        if isinstance(cmd, DriveCommand):
            r.cmd_speed, r.cmd_steer = cmd.speed_mm_s, cmd.steer_rad
        elif cmd.action == "stop":
            r.cmd_speed = 0.0
        elif cmd.action.startswith("reel_"):
            self._reel = cmd.action.removeprefix("reel_")
        else:
            self._clamp = cmd.action.removeprefix("clamp_")

    # --- PoseSource ---
    def read(self, t: float) -> list[RobotState]:
        if self._camera_ok:
            ts, poses = self._snapshot(t - self.cfg.sim.camera_latency_s)
            c = self.cfg.sim
            for p in poses:
                if self._rng.random() < c.occlusion_prob:
                    continue
                self._last_seen[p.robot_id] = RobotState(
                    p.robot_id,
                    p.x + self._rng.gauss(0, c.pos_noise_mm),
                    p.y + self._rng.gauss(0, c.pos_noise_mm),
                    _wrap(p.heading + self._rng.gauss(0, c.heading_noise_rad)),
                    ts,
                )
        return [self._last_seen[rid] for rid in self._order]

    # --- 시뮬레이션 진행 ---
    def handle(self, event: Event) -> None:
        if event.kind in ("comm_loss", "comm_restore"):
            ids = [event.data["robot_id"]] if "robot_id" in event.data else self._order
            for rid in ids:
                self._robots[rid].comm_ok = event.kind == "comm_restore"
        elif event.kind in ("camera_loss", "camera_restore"):
            self._camera_ok = event.kind == "camera_restore"

    def step(self, dt: float) -> None:
        rc = self.cfg.robot
        lag = 1 - math.exp(-dt / rc.steer_delay_s)
        t_next = round(self.time + dt, 9)  # 이벤트는 막힌 결과가 보이는 시각으로 기록
        for rid in self._order:
            r = self._robots[rid]
            if self.time - r.last_rx > self.cfg.sim.watchdog_hold_s + _EPS:
                r.speed = math.copysign(max(0.0, abs(r.speed) - self._decel * dt), r.speed)
            else:
                r.speed = _clamp(r.cmd_speed, rc.max_speed_mm_s)
            r.steer += (_clamp(r.cmd_steer, self._max_steer) - r.steer) * lag
            old = (r.x, r.y, r.heading)
            r.x, r.y, r.heading = _advance(*old, r.speed, r.steer, rc.wheelbase_mm, dt)
            if self.net_length() > self._net_limit() + _EPS:
                r.x, r.y, r.heading = old
                r.speed = 0.0
                if rid not in self._blocked:
                    self._blocked.add(rid)
                    self._events.append({"t": t_next, "kind": "net_limit",
                                         "data": {"robot_id": rid, "reel": self._reel}})
            else:
                self._blocked.discard(rid)
        if self._reel != "hold":
            self._unwound = self.net_length()  # 릴 장력으로 처짐 없이 팽팽
        self.time = t_next
        self._history.append((self.time, self.truth()))
        keep = self.time - self.cfg.sim.camera_latency_s - 1.0
        while len(self._history) > 1 and self._history[1][0] <= keep:
            self._history.pop(0)

    def pop_events(self) -> list[dict]:
        out, self._events = self._events, []
        return out

    # --- 상태 조회 ---
    def truth(self) -> list[RobotState]:
        return [RobotState(rid, self._robots[rid].x, self._robots[rid].y,
                           self._robots[rid].heading, self.time) for rid in self._order]

    def net_length(self) -> float:
        pts = [(self._robots[rid].x, self._robots[rid].y) for rid in self._order]
        return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))

    def unwound_mm(self) -> float:
        return self._unwound

    def speed(self, robot_id: int) -> float:
        return self._robots[robot_id].speed

    def steer(self, robot_id: int) -> float:
        return self._robots[robot_id].steer

    def log_fields(self) -> dict:
        return {
            "truth": [{"robot_id": p.robot_id, "x": p.x, "y": p.y, "heading": p.heading}
                      for p in self.truth()],
            "net": {"length_mm": self.net_length(), "unwound_mm": self._unwound,
                    "reel": self._reel, "clamp": self._clamp},
        }

    def _net_limit(self) -> float:
        # 풀기: 그물 전체 길이까지. 유지·감기: 지금 풀린 길이 이상으로 늘어나지 않음
        return self.cfg.net.length_mm if self._reel == "unwind" else self._unwound

    def _snapshot(self, t: float) -> tuple[float, list[RobotState]]:
        chosen = self._history[0]
        for item in self._history:
            if item[0] <= t + _EPS:
                chosen = item
            else:
                break
        return chosen


def _advance(x, y, h, v, steer, wheelbase, dt):
    d = v * dt
    if abs(steer) < 1e-12:
        return x + d * math.cos(h), y + d * math.sin(h), h
    radius = wheelbase / math.tan(steer)
    nh = h + d / radius
    return x + radius * (math.sin(nh) - math.sin(h)), y - radius * (math.cos(nh) - math.cos(h)), _wrap(nh)


def _wrap(a: float) -> float:
    return math.remainder(a, math.tau)


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))
