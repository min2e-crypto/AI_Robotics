"""키 입력 처리 (pygame 없음). 키는 이름 문자열: "1"~"5", "up", "down", "left", "right", "space", 글자."""

from __future__ import annotations

import bisect
import math

from ..config import Config
from ..io import ActionCommand, DriveCommand
from ..types import Role

SPEED_STEP_MM_S = 50.0
STEER_STEP_RAD = 0.05
REEL_KEYS = {"u": "reel_unwind", "h": "reel_hold", "w": "reel_wind"}


class ManualControl:
    """선택한 로봇의 목표 속도·조향을 키로 바꾸고, 매 tick 모든 로봇에 주행 명령을 보낸다."""

    def __init__(self, config: Config):
        self._ids = [r.robot_id for r in config.robots]
        role_id = {r.role: r.robot_id for r in config.robots}
        self._reserve, self._puller = role_id[Role.RESERVE], role_id[Role.PULLER]
        self._max_speed = config.robot.max_speed_mm_s
        self._max_steer = math.atan(config.robot.wheelbase_mm / config.robot.min_turn_radius_mm)
        self._target = {rid: [0.0, 0.0] for rid in self._ids}  # [속도, 조향]
        self._clamp_open = False
        self.selected = self._ids[0]

    def press(self, key: str) -> list[ActionCommand]:
        """키 하나를 처리하고, 바로 보낼 일회성 명령을 돌려준다."""
        tgt = self._target[self.selected]
        if key.isdigit() and 1 <= int(key) <= len(self._ids):
            self.selected = self._ids[int(key) - 1]
        elif key in ("up", "down"):
            step = SPEED_STEP_MM_S if key == "up" else -SPEED_STEP_MM_S
            tgt[0] = _clamp(tgt[0] + step, self._max_speed)
        elif key in ("left", "right"):  # 왼쪽 = 반시계 = +조향
            step = STEER_STEP_RAD if key == "left" else -STEER_STEP_RAD
            tgt[1] = _clamp(tgt[1] + step, self._max_steer)
        elif key == "space":
            tgt[0] = 0.0
        elif key in REEL_KEYS:
            return [ActionCommand(self._reserve, REEL_KEYS[key])]
        elif key == "c":
            self._clamp_open = not self._clamp_open
            return [ActionCommand(self._puller, "clamp_open" if self._clamp_open else "clamp_close")]
        return []

    def drive_commands(self) -> list[DriveCommand]:
        return [DriveCommand(rid, *self._target[rid]) for rid in self._ids]


SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SEEK_S = 1.0


class Playback:
    """로그 재생 위치. update(실제 경과 초)로 진행하고 current()로 그 시각의 tick을 얻는다."""

    def __init__(self, ticks: list[dict]):
        self._ticks = ticks
        self._times = [tk["t"] for tk in ticks]
        self.pos = self._times[0]
        self.speed = 1.0
        self.paused = False

    def update(self, real_dt: float) -> None:
        if not self.paused:
            self._seek(real_dt * self.speed)

    def press(self, key: str) -> None:
        i = SPEEDS.index(self.speed)
        if key == "space":
            self.paused = not self.paused
        elif key in ("left", "right"):
            self._seek(SEEK_S if key == "right" else -SEEK_S)
        elif key in ("+", "="):
            self.speed = SPEEDS[min(i + 1, len(SPEEDS) - 1)]
        elif key == "-":
            self.speed = SPEEDS[max(i - 1, 0)]

    def current(self) -> dict:
        return self._ticks[max(0, bisect.bisect_right(self._times, self.pos + 1e-9) - 1)]

    def _seek(self, delta: float) -> None:
        self.pos = min(max(self.pos + delta, self._times[0]), self._times[-1])


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))
