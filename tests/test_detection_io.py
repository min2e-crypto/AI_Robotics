import json
from pathlib import Path

from fencebot.config import load_config
from fencebot.detection_io import load_frame, load_frames, load_vision_results, vision_frame
from fencebot.incident import incident_from_frames

ROOT = Path(__file__).resolve().parents[1]


def test_fake_detection_example_loads():
    fr = load_frame(ROOT / "examples" / "detections" / "frame_000120.json")
    assert fr.frame == 120
    assert all(d.frame == fr.frame for d in fr.detections)
    cars = [d for d in fr.detections if d.cls == "car"]
    assert cars and all(d.track_id is not None for d in cars)
    tracked = {d.track_id for d in cars}
    assert {v.track_id for v in fr.vehicle_states} <= tracked
    assert fr.lanes_px and all(len(line) >= 2 for line in fr.lanes_px.values())


# ───── share_model/vision.py 결과 (results.jsonl) ─────
VISION_RESULT = {  # share_model/README.md의 출력 예시
    "image_size": [1920, 1080],
    "vehicles": [
        {"id": 0, "box": [414.3, 867.3, 784.2, 1019.0], "det_conf": 0.892, "vtype": "car",
         "probs": {"damaged": 0.012, "normal": 0.337, "overturned": 0.65},
         "state": "overturned", "rule": "top1"},
        {"id": 1, "box": [100.0, 100.0, 110.0, 108.0], "det_conf": 0.3, "vtype": "vehicle",
         "probs": None, "state": "uncertain", "rule": "too_small"},
    ],
    "fires": [{"id": 0, "cls": "fire", "conf": 0.48, "box": [1045.7, 603.7, 1555.7, 893.9]}],
}


def test_vision_result_becomes_frame():
    f = vision_frame(VISION_RESULT, frame=30, timestamp=1.0)
    assert [(d.cls, d.track_id, d.confidence) for d in f.detections] == [
        ("car", 0, 0.892), ("car", 1, 0.3), ("fire", None, 0.48)]
    assert f.detections[0].bbox_px == (414.3, 867.3, 784.2, 1019.0)
    assert all(d.frame == 30 and d.timestamp == 1.0 for d in f.detections)
    assert [(s.track_id, s.label) for s in f.vehicle_states] == [(0, "overturned"), (1, "uncertain")]
    assert f.vehicle_states[0].scores["overturned"] == 0.65 and f.vehicle_states[1].scores == {}


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_video_results_keep_frame_numbers(tmp_path):
    p = tmp_path / "results.jsonl"
    write_jsonl(p, [{**VISION_RESULT, "file": "a.mp4", "frame": n} for n in (0, 10, 20)])
    frames = load_vision_results(p, fps=10.0)
    assert [(f.frame, f.timestamp) for f in frames] == [(0, 0.0), (10, 1.0), (20, 2.0)]
    assert [f.frame for f in load_frames(p)] == [0, 10, 20]  # 파일을 주면 vision 결과로 읽는다


def test_photo_folder_results_use_line_order(tmp_path):
    p = tmp_path / "results.jsonl"
    write_jsonl(p, [{**VISION_RESULT, "file": f"{n}.jpg", "frame": 0} for n in range(3)])
    assert [f.frame for f in load_vision_results(p)] == [0, 1, 2]


def test_vision_results_give_same_zone_as_frame_folder(tmp_path):
    """가짜 Detection 예제를 vision.py 형식으로 옮겨 적어도 같은 사고 범위가 나온다."""
    cfg = load_config()
    folder = ROOT / "examples" / "detections" / "lane2_basic"
    rows = []
    for f in load_frames(folder):
        labels = {s.track_id: s.label for s in f.vehicle_states}
        rows.append({
            "frame": f.frame,
            "vehicles": [{"id": d.track_id, "box": list(d.bbox_px), "det_conf": d.confidence,
                          "vtype": "vehicle", "probs": None, "state": labels[d.track_id]}
                         for d in f.detections if d.cls == "car"],
            "fires": [{"id": 0, "cls": d.cls, "conf": d.confidence, "box": list(d.bbox_px)}
                      for d in f.detections if d.cls != "car"],
        })
    p = tmp_path / "results.jsonl"
    write_jsonl(p, rows)
    a, b = incident_from_frames(load_frames(folder), cfg), incident_from_frames(load_frames(p), cfg)
    assert a.zone is not None and a.zone == b.zone and a.vehicles == b.vehicles
