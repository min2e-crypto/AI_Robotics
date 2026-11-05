"""상태 로그 (JSON Lines).

레코드 종류
- header: {"type", "version", "scenario", "config"}  — 첫 줄 1개
- event:  {"type", "t", "kind", "data"}
- tick:   {"type", "t", "phase", "robots": [RobotState dict, ...]}
"""

from __future__ import annotations

import json
from pathlib import Path

LOG_VERSION = 1


class LogWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        self._f.write(line + "\n")
        self.records.append(json.loads(line))  # 읽어 온 값과 같은 모양으로 보관

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_log(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
