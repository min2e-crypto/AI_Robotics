from pathlib import Path

from fencebot.scenario import load_scenario

ROOT = Path(__file__).resolve().parents[1]


def test_example_scenario_loads():
    sc = load_scenario(ROOT / "scenarios" / "lane2_basic.toml")
    assert sc.duration_s > 0 and sc.dt_s > 0
    assert sc.zone is not None and sc.zone.lanes == ("lane2",)
    assert any(v.label != "normal" for v in sc.vehicles)
    assert [e.t for e in sc.events] == sorted(e.t for e in sc.events)
