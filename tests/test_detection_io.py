from pathlib import Path

from fencebot.detection_io import load_frame

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
