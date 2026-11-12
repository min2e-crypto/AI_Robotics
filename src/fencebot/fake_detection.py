"""시나리오에서 가짜 Detection 프레임을 만든다 (모델 없이 사고 범위 계산을 개발·시연하기 위함).

차량의 바닥 사각형을 설정의 임시 카메라로 영상에 옮기고, 차량 높이만큼 박스 위쪽을 키운다.
형식은 examples/detections/README.md와 같다.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .config import Config
from .detection_io import DetectionFrame
from .homography import apply, solve
from .scenario import Scenario
from .types import Detection, VehicleState

FPS = 30.0
HEIGHT_RATIO = 0.5  # 박스 위쪽을 바닥 사각형 높이의 이 비율만큼 키운다 (차체 높이 흉내)


def fake_frames(scenario: Scenario, config: Config, count: int = 30,
                noise_px: float = 1.5) -> list[DetectionFrame]:
    to_px = solve(list(config.camera.road_points_mm), list(config.camera.image_points_px))
    rng = random.Random(scenario.seed)
    frames = []
    for i in range(count):
        ts = i / FPS
        dets = []
        for v in scenario.vehicles:
            x1, y1, x2, y2 = v.bbox
            pts = [apply(to_px, p) for p in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            top = min(ys) - HEIGHT_RATIO * (max(ys) - min(ys))
            box = tuple(round(c + rng.gauss(0, noise_px), 1) if noise_px else c
                        for c in (min(xs), top, max(xs), max(ys)))
            dets.append(Detection(i, ts, "car", 0.9, box, v.track_id))
        states = tuple(VehicleState(v.track_id, v.label,
                                    {k: 0.9 if k == v.label else 0.05
                                     for k in ("normal", "damaged", "overturned")})
                       for v in scenario.vehicles)
        frames.append(DetectionFrame(i, ts, tuple(dets), states, {}))
    return frames


def write_frames(frames: list[DetectionFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in frames:
        raw = {
            "frame": f.frame,
            "timestamp": f.timestamp,
            "detections": [{"cls": d.cls, "confidence": d.confidence, "bbox_px": list(d.bbox_px),
                            **({"track_id": d.track_id} if d.track_id is not None else {})}
                           for d in f.detections],
            "vehicle_states": [{"track_id": v.track_id, "label": v.label, "scores": v.scores}
                               for v in f.vehicle_states],
        }
        (out / f"frame_{f.frame:06d}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
