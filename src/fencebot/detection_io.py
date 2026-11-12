"""팀원 모델 출력 읽기.

두 가지 형식을 읽는다.
- 프레임별 JSON 폴더 (`frame_*.json`). 형식은 examples/detections/README.md.
- `share_model/vision.py`가 쓰는 `results.jsonl` (한 줄이 한 프레임). 형식은 share_model/README.md.
"""

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


VISION_VEHICLE_TYPES = ("vehicle", "car", "van", "truck", "bus")  # vision.py의 vtype → 모두 "car"


def load_frames(source: str | Path) -> list[DetectionFrame]:
    """폴더면 frame_*.json을 프레임 번호 순서로, .jsonl 파일이면 vision.py 결과를 읽는다."""
    source = Path(source)
    if source.is_file():
        return load_vision_results(source)
    return sorted((load_frame(p) for p in source.glob("frame_*.json")), key=lambda f: f.frame)


def load_vision_results(path: str | Path, fps: float = 30.0) -> list[DetectionFrame]:
    """vision.py의 results.jsonl → 프레임 목록.

    영상이면 줄마다 `frame`이 커지므로 그 값을 쓰고, 사진 폴더(모두 0)면 줄 순번을 쓴다.
    결과에 시각이 없어서 timestamp는 frame / fps로 채운다.
    """
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    numbers = [r.get("frame", 0) for r in rows]
    if any(b <= a for a, b in zip(numbers, numbers[1:])):
        numbers = list(range(len(rows)))
    return [vision_frame(r, n, n / fps) for r, n in zip(rows, numbers)]


def vision_frame(result: dict, frame: int = 0, timestamp: float = 0.0) -> DetectionFrame:
    """`VisionModels.run()` 결과 하나 → DetectionFrame.

    차량 id는 사진 안에서만 뜻이 있다(추적 아님). 상태 `uncertain`은 label로 그대로 넘긴다.
    """
    vehicles = [v for v in result.get("vehicles", []) if v.get("vtype", "vehicle") in VISION_VEHICLE_TYPES]
    detections = [
        Detection(frame=frame, timestamp=timestamp, cls="car", confidence=v["det_conf"],
                  bbox_px=tuple(v["box"]), track_id=v["id"])
        for v in vehicles
    ] + [
        Detection(frame=frame, timestamp=timestamp, cls=f["cls"], confidence=f["conf"],
                  bbox_px=tuple(f["box"]))
        for f in result.get("fires", [])
    ]
    states = tuple(VehicleState(track_id=v["id"], label=v["state"], scores=v.get("probs") or {})
                   for v in vehicles)
    return DetectionFrame(frame, timestamp, tuple(detections), states, {})


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
