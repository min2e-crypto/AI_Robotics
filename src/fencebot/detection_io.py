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


def load_frames(source: str | Path, photo: str | None = None, hold: int = 1) -> list[DetectionFrame]:
    """폴더면 frame_*.json을 프레임 번호 순서로, .jsonl 파일이면 vision.py 결과를 읽는다.

    photo를 주면 results.jsonl에서 그 사진 한 줄만 읽는다 (`select_photo`).
    프레임이 하나뿐이면 멈춘 사진으로 보고 hold번 되풀이한다. 시간 안정화(confirm_frames)를 넘기기 위해서다.
    """
    source = Path(source)
    if source.is_file():
        frames = load_vision_results(source, photo=photo)
    elif photo is not None:
        raise ValueError("photo는 vision.py의 results.jsonl에서만 고를 수 있습니다")
    else:
        frames = sorted((load_frame(p) for p in source.glob("frame_*.json")), key=lambda f: f.frame)
    return frames * hold if len(frames) == 1 else frames


def read_vision_rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_photo(rows: list[dict], photo: str) -> dict:
    """사진 이름(`file`, 확장자는 빼도 됨)이나 줄 순번(0부터)으로 결과 한 줄을 고른다."""
    for r in rows:
        name = str(r.get("file", ""))
        if photo in (name, Path(name).stem):
            return r
    if photo.isdigit() and int(photo) < len(rows):
        return rows[int(photo)]
    names = ", ".join(str(r.get("file", i)) for i, r in enumerate(rows))
    raise ValueError(f"결과에 사진 {photo!r}가 없습니다. 있는 것: {names}")


def load_vision_results(path: str | Path, fps: float = 30.0, photo: str | None = None) -> list[DetectionFrame]:
    """vision.py의 results.jsonl → 프레임 목록. photo를 주면 그 사진 한 줄만.

    영상이면 줄마다 `frame`이 커지므로 그 값을 쓰고, 사진 폴더(모두 0)면 줄 순번을 쓴다.
    결과에 시각이 없어서 timestamp는 frame / fps로 채운다.
    """
    rows = read_vision_rows(path)
    if photo is not None:
        return [vision_frame(select_photo(rows, photo))]
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
