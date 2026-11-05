"""명령줄 진입점.

    fencebot run scenarios/lane2_basic.toml [--config config/default.toml] [--log logs/run.jsonl]
    fencebot replay logs/run.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .io import StaticBackend
from .log import read_log
from .runner import run
from .scenario import load_scenario


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fencebot")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="시나리오 실행 후 로그 저장")
    r.add_argument("scenario")
    r.add_argument("--config")
    r.add_argument("--log")
    rp = sub.add_parser("replay", help="로그를 시간 순서대로 출력")
    rp.add_argument("log")
    args = p.parse_args(argv)

    if args.cmd == "run":
        sc = load_scenario(args.scenario)
        cfg = load_config(args.config)
        log_path = Path(args.log or f"logs/{sc.name}.jsonl")
        records = run(sc, cfg, StaticBackend(cfg.standby), log_path)
        print(f"{log_path} 저장 ({len(records)}줄)")
    else:
        replay(args.log)


def replay(path: str | Path) -> None:
    for rec in read_log(path):
        if rec["type"] == "header":
            print(f"# 시나리오 {rec['scenario']['name']} (로그 v{rec['version']})")
        elif rec["type"] == "event":
            print(f"{rec['t']:7.2f}s  이벤트 {rec['kind']} {rec['data']}")
        else:
            poses = " ".join(f"{r['robot_id']}:({r['x']:.0f},{r['y']:.0f})" for r in rec["robots"])
            print(f"{rec['t']:7.2f}s  {rec['phase']:<10} {poses}")


if __name__ == "__main__":
    main()
