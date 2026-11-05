"""로봇 명령을 보내는 곳과 위치를 받는 곳 (시뮬·실물 교체 지점).

상태 흐름·배치·경로 코드는 RobotLink와 PoseSource만 보고, 구현은 시뮬레이터(S2)나 실물(H2)로 바꾼다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Union

from .types import RobotState


@dataclass(frozen=True)
class DriveCommand:
    """주행 명령. steer_rad는 평균 앞바퀴각, 좌우 속도 분배는 로봇 쪽에서."""

    robot_id: int
    speed_mm_s: float
    steer_rad: float


@dataclass(frozen=True)
class ActionCommand:
    """일회성 명령 (실물에서는 확인 응답까지 재전송, 구현_계획 9절)."""

    robot_id: int
    action: str  # stop / reel_unwind / reel_wind / reel_hold / clamp_close / clamp_open


Command = Union[DriveCommand, ActionCommand]


class RobotLink(Protocol):
    def send(self, cmd: Command) -> None: ...


class PoseSource(Protocol):
    def read(self, t: float) -> list[RobotState]:
        """시각 t에 관제 쪽이 알고 있는 로봇 위치 (실물은 카메라 추정값)."""
        ...


class StaticBackend:
    """움직이지 않는 자리 표시용 구현. S2 시뮬레이터가 대체한다."""

    def __init__(self, poses: tuple[RobotState, ...] | list[RobotState]):
        self._poses = list(poses)
        self.sent: list[Command] = []

    def send(self, cmd: Command) -> None:
        self.sent.append(cmd)

    def read(self, t: float) -> list[RobotState]:
        return [replace(p, t=t) for p in self._poses]
