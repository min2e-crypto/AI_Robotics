import json

from fencebot.types import (
    Detection,
    RobotState,
    Target,
    VehicleState,
    Zone,
    from_dict,
    to_dict,
)


def roundtrip(obj):
    return from_dict(type(obj), json.loads(json.dumps(to_dict(obj))))


def test_five_types_roundtrip_through_json():
    objs = [
        Detection(frame=3, timestamp=0.1, cls="car", confidence=0.9,
                  bbox_px=(10.0, 20.0, 110.0, 80.0), track_id=7),
        VehicleState(track_id=7, label="damaged",
                     scores={"normal": 0.1, "damaged": 0.8, "overturned": 0.1}),
        Zone(bbox=(3000.0, 200.0, 4200.0, 1100.0), lanes=("lane2",), kind="fire"),
        RobotState(robot_id=2, x=100.0, y=-300.0, heading=1.57, t=0.5),
        Target(robot_id=2, x=2000.0, y=1167.0, heading=0.0),
    ]
    for obj in objs:
        assert roundtrip(obj) == obj


def test_optional_fields_default():
    d = Detection(frame=0, timestamp=0.0, cls="smoke", confidence=0.5,
                  bbox_px=(0.0, 0.0, 1.0, 1.0))
    assert d.track_id is None
    assert roundtrip(d) == d
    assert VehicleState(track_id=1, label="normal", scores={}).stopped is None
