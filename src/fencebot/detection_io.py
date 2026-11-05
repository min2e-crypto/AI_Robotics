"""팀원 모델 출력(프레임별 JSON) 읽기. 형식은 examples/detections/README.md."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .types import Detection, VehicleState


@dataclass(frozen=True)
class DetectionFrame:
    frame: int
    timestamp: float
    detections: tuple[Detection, ...]
    vehicle_states: tuple[VehicleState, ...]
    lanes_px: dict[str, tuple[tuple[float, float], ...]]  # 선 이름 → 픽셀 꺾은선 (없으면 빈 dict)


def load_frame(path: str | Path) -> DetectionFrame:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    frame, ts = raw["frame"], raw["timestamp"]
    return DetectionFrame(
        frame=frame,
        timestamp=ts,
        detections=tuple(
            Detection(frame=frame, timestamp=ts, cls=d["cls"], confidence=d["confidence"],
                      bbox_px=tuple(d["bbox_px"]), track_id=d.get("track_id"))
            for d in raw["detections"]
        ),
        vehicle_states=tuple(
            VehicleState(track_id=v["track_id"], label=v["label"], scores=v["scores"])
            for v in raw.get("vehicle_states", [])
        ),
        lanes_px={name: tuple(tuple(p) for p in pts)
                  for name, pts in raw.get("lanes_px", {}).items()},
    )
