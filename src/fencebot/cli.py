"""명령줄 진입점.

    fencebot run scenarios/lane2_basic.toml [--config config/default.toml] [--log logs/run.jsonl]
    fencebot replay logs/run.jsonl
    fencebot sim scenarios/lane2_basic.toml [--duration 600]   # 수동 조작 창 (pygame)
    fencebot view logs/run.jsonl                               # 로그 재생 창 (pygame)
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .log import read_log
from .runner import run
from .scenario import load_scenario
from .sim import SimBackend


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fencebot")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="시나리오 실행 후 로그 저장")
    r.add_argument("scenario")
    r.add_argument("--config")
    r.add_argument("--log")
    rp = sub.add_parser("replay", help="로그를 시간 순서대로 출력")
    rp.add_argument("log")
    s = sub.add_parser("sim", help="수동 조작 창 (실행 내용을 로그로 저장)")
    s.add_argument("scenario")
    s.add_argument("--config")
    s.add_argument("--log")
    s.add_argument("--duration", type=float, default=600.0, help="최대 실행 시간(초)")
    v = sub.add_parser("view", help="로그 재생 창")
    v.add_argument("log")
    args = p.parse_args(argv)

    if args.cmd == "run":
        sc = load_scenario(args.scenario)
        cfg = load_config(args.config)
        log_path = Path(args.log or f"logs/{sc.name}.jsonl")
        records = run(sc, cfg, SimBackend(cfg, seed=sc.seed), log_path)
        print(f"{log_path} 저장 ({len(records)}줄)")
    elif args.cmd == "sim":
        from .view.app import run_manual  # pygame은 화면 명령에서만 불러온다

        sc = replace(load_scenario(args.scenario), duration_s=args.duration)
        log_path = Path(args.log or f"logs/{sc.name}_manual.jsonl")
        run_manual(sc, load_config(args.config), log_path)
        print(f"{log_path} 저장")
    elif args.cmd == "view":
        from .view.app import run_view

        run_view(args.log)
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
            net = rec.get("net")
            net_txt = f"  그물 {net['unwound_mm']:.0f}mm {net['reel']}" if net else ""
            print(f"{rec['t']:7.2f}s  {rec['phase']:<10} {poses}{net_txt}")


if __name__ == "__main__":
    main()
