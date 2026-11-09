from pathlib import Path

from fencebot.config import load_config
from fencebot.io import StaticBackend
from fencebot.log import read_log
from fencebot.runner import run
from fencebot.scenario import load_scenario
from fencebot.types import RobotState, from_dict

ROOT = Path(__file__).resolve().parents[1]


def test_run_writes_log_that_replays(tmp_path):
    cfg = load_config()
    sc = load_scenario(ROOT / "scenarios" / "lane2_basic.toml")
    log_path = tmp_path / "run.jsonl"

    written = run(sc, cfg, StaticBackend(cfg.standby), log_path)
    records = read_log(log_path)

    assert records == written
    assert records[0]["type"] == "header"
    assert records[0]["scenario"]["name"] == sc.name

    ticks = [r for r in records if r["type"] == "tick"]
    assert len(ticks) == round(sc.duration_s / sc.dt_s) + 1
    robots = [from_dict(RobotState, d) for d in ticks[0]["robots"]]
    assert {r.robot_id for r in robots} == {p.robot_id for p in cfg.standby}

    events = [r for r in records if r["type"] == "event"]
    assert [(e["t"], e["kind"]) for e in events] == [(e.t, e.kind) for e in sc.events]


def test_sim_run_logs_truth_net_and_handles_events(tmp_path, capsys):
    from fencebot.cli import main
    from fencebot.sim import SimBackend

    cfg = load_config()
    sc = load_scenario(ROOT / "scenarios" / "lane2_basic.toml")
    log_path = tmp_path / "sim.jsonl"
    sim = SimBackend(cfg, seed=sc.seed)

    records = run(sc, cfg, sim, log_path)

    assert records[0]["version"] == 3
    ticks = [r for r in records if r["type"] == "tick"]
    assert ticks[-1]["t"] == sc.duration_s == sim.time
    assert {"truth", "net"} <= ticks[0].keys()
    assert ticks[0]["net"]["unwound_mm"] == sim.net_length()
    assert sim._robots[4].comm_ok  # comm_loss → comm_restore 처리됨

    main(["replay", str(log_path)])
    assert "그물" in capsys.readouterr().out
