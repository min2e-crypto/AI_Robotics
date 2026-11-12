"""전개 계획을 그대로 재생하는 백엔드 (발표용, ADR-014).

추종 제어·조향 지연·카메라 잡음·와치독 없이, 로봇을 계획 경로 위에서 일정 속도로 옮긴다.
한 번에 한 대씩 계획 순서대로 움직이고, 그물 걸기는 시뮬레이터와 같은 판정(net.py)을 쓴다.
명령(send)과 시나리오 이벤트(handle)는 받지 않는다.
"""

from __future__ import annotations

import math

from .config import Config
from .dubins import sample
from .io import Command
from .net import hook_events, net_length
from .planner import Deployment
from .scenario import Event
from .types import Phase, RobotState

HOLD_S = 3.0  # 전개가 끝난 뒤 완성된 모습을 보여 주는 시간


class PlaybackBackend:
    def __init__(self, config: Config, deployment: Deployment, start_delay_s: float = 1.0):
        self.cfg = config
        self.time = 0.0
        self._delay = start_delay_s
        self._speed = config.robot.max_speed_mm_s
        self._order = [r.robot_id for r in config.robots]
        self._poses = {s.robot_id: (s.x, s.y, s.heading) for s in config.standby}
        self._moves = deployment.moves
        self._tracks = [_track(m, config.planning.step_mm) for m in self._moves]
        self._i = 0       # 지금 실행 중인 이동
        self._s = 0.0     # 그 이동에서 간 거리
        self._hooked: set[int] = set()
        self._events: list[dict] = []
        self.duration_s = start_delay_s + sum(tr[-1][0] for tr in self._tracks) / self._speed + HOLD_S

    @property
    def done(self) -> bool:
        return self._i >= len(self._moves)

    @property
    def phase(self) -> str:
        if self.done:
            return Phase.HOLD.value if self._moves else Phase.STANDBY.value
        if self.time < self._delay:
            return Phase.STANDBY.value
        return (Phase.PULLER_DEPLOY if self._i == 0 else Phase.POSTS_ENTER).value

    # --- RobotLink / PoseSource ---
    def send(self, cmd: Command) -> None:
        pass

    def read(self, t: float) -> list[RobotState]:
        return [RobotState(rid, *self._poses[rid], t) for rid in self._order]

    # --- 진행 ---
    def handle(self, event: Event) -> None:
        pass

    def step(self, dt: float) -> None:
        t_next = round(self.time + dt, 9)
        if not self.done and self.time >= self._delay - 1e-9:
            move, track = self._moves[self._i], self._tracks[self._i]
            before = self._points()
            self._s = min(self._s + self._speed * dt, track[-1][0])
            arrived = self._s >= track[-1][0] - 1e-9
            self._poses[move.robot_id] = move.goal if arrived else _at(track, self._s)
            posts = [p for p, _, _, ok in hook_events(self._order, self._hooked, before, self._points(),
                                                      self.cfg.net.hook_tol_mm) if ok]
            if arrived:
                posts += move.hooks  # 계획에서 걸린다고 본 기둥은 도착 때까지는 걸린다
                self._i, self._s = self._i + 1, 0.0
                if self.done:
                    posts += self._order[1:-1]  # 완화 모드에서 걸리지 못한 기둥도 펜스에 넣는다
            for post in posts:
                if post not in self._hooked:
                    self._hooked.add(post)
                    self._events.append({"t": t_next, "kind": "net_hook", "data": {"robot_id": post}})
        self.time = t_next

    def pop_events(self) -> list[dict]:
        out, self._events = self._events, []
        return out

    def truth(self) -> list[RobotState]:
        return self.read(self.time)

    def log_fields(self) -> dict:
        length = net_length(self._order, self._hooked, self._points())
        return {
            "truth": [{"robot_id": rid, "x": p[0], "y": p[1], "heading": p[2]}
                      for rid, p in ((r, self._poses[r]) for r in self._order)],
            "net": {"length_mm": length, "unwound_mm": length, "reel": "unwind", "clamp": "closed",
                    "hooked": [rid for rid in self._order if rid in self._hooked]},
        }

    def _points(self) -> dict[int, tuple[float, float]]:
        return {rid: p[:2] for rid, p in self._poses.items()}


def _track(move, step: float) -> list[tuple[float, tuple[float, float, float]]]:
    """(출발부터 간 거리, 자세) 목록."""
    pts = [p for path in move.paths for p in sample(path, step)]
    out, s = [(0.0, pts[0])], 0.0
    for a, b in zip(pts, pts[1:]):
        s += math.dist(a[:2], b[:2])
        out.append((s, b))
    return out


def _at(track, s: float) -> tuple[float, float, float]:
    for (s0, a), (s1, b) in zip(track, track[1:]):
        if s <= s1:
            k = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)
            dh = math.remainder(b[2] - a[2], math.tau)
            return a[0] + k * (b[0] - a[0]), a[1] + k * (b[1] - a[1]), a[2] + k * dh
    return track[-1][1]
