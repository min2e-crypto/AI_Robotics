"""명령줄 진입점.

    fencebot run scenarios/lane2_basic.toml [--config config/default.toml] [--log logs/run.jsonl]
    fencebot replay logs/run.jsonl
    fencebot sim scenarios/lane2_basic.toml [--duration 600]   # 수동 조작 창 (pygame)
    fencebot view logs/run.jsonl                               # 로그 재생 창 (pygame)
    fencebot demo scenarios/lane2_basic.toml [--strict] [--no-window]  # 전개 자동 재생 (발표용)
    fencebot present scenarios/lane2_basic.toml                # 발표 화면: 탐지 → 범위 확정 → 전개
    fencebot demo scenarios/lane2_basic.toml --detections DIR  # 범위·차량을 Detection에서 계산해 재생
                                                               # DIR 대신 vision.py의 results.jsonl도 됨
    fencebot zone DIR                                          # Detection 폴더 → 사고 범위 출력
    fencebot fake scenarios/lane2_basic.toml DIR [--frames 30]  # 시나리오로 가짜 Detection 생성
    fencebot calib 사진.jpg [--rect x0,y0,x1,y1] [--out config/사진.toml]  # 도로 4점 찍기 → [camera] 설정
    fencebot present scenarios/lane2_basic.toml --config config/사진.toml \
        --detections results.jsonl --photo 사진.jpg --images 사진폴더   # 사진 한 장으로 발표 화면
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import config_from_raw, load_config
from .detection_io import load_frames, read_vision_rows, select_photo
from .fake_detection import fake_frames, write_frames
from .incident import incident_from_frames
from .log import read_log
from .planner import plan_deployment
from .playback import PlaybackBackend
from .runner import run
from .scenario import load_scenario, scenario_from_raw
from .sim import SimBackend

PHOTO_HELP = ("results.jsonl에서 사진 한 장만 씀 (사진 이름 또는 줄 순번). "
              "프레임이 하나면 범위 확정에 필요한 만큼 되풀이함")


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
    d = sub.add_parser("demo", help="전개 계획 자동 재생 (물리 모델 없음, 발표용)")
    d.add_argument("scenario")
    d.add_argument("--config")
    d.add_argument("--log")
    d.add_argument("--detections", help="Detection 폴더 또는 vision.py의 results.jsonl. 주면 시나리오의 범위·차량 대신 씀")
    d.add_argument("--photo", help=PHOTO_HELP)
    d.add_argument("--strict", action="store_true", help="검사를 완화하지 않음")
    d.add_argument("--no-window", action="store_true", help="창 없이 로그만 저장")
    pr = sub.add_parser("present", help="발표 화면: 탐지 → 범위 확정 → 전개가 한 창에서 이어짐")
    pr.add_argument("scenario")
    pr.add_argument("--config")
    pr.add_argument("--log")
    pr.add_argument("--detections", help="Detection 폴더 또는 vision.py의 results.jsonl. 없으면 시나리오로 가짜 Detection을 만듦")
    pr.add_argument("--photo", help=PHOTO_HELP)
    pr.add_argument("--images", help="영상 프레임 사진 폴더 (이름 순서 = 프레임 순서) 또는 사진 파일 하나. "
                                     "없으면 도로를 그려 대신함")
    pr.add_argument("--speed", type=float, default=2.0, help="전개 재생 배속")
    pr.add_argument("--frame-time", type=float, default=0.4, help="프레임 하나를 보여 주는 시간(초)")
    pr.add_argument("--width", type=int, default=1200, help="창 폭(px)")
    pr.add_argument("--strict", action="store_true", help="검사를 완화하지 않음")
    z = sub.add_parser("zone", help="Detection 폴더 또는 vision.py의 results.jsonl에서 사고 범위 계산")
    z.add_argument("detections")
    z.add_argument("--config")
    z.add_argument("--photo", help=PHOTO_HELP)
    c = sub.add_parser("calib", help="사진에서 도로 4점을 찍어 [camera] 설정 파일을 만듦")
    c.add_argument("image")
    c.add_argument("--rect", help="네 점이 가리키는 도로 좌표 사각형 x0,y0,x1,y1 (mm). "
                                  "없으면 설정 [camera]의 road_points_mm. 찍는 순서: (x0,y0) (x1,y0) (x1,y1) (x0,y1)")
    c.add_argument("--out", help="저장할 설정 파일 (기본 config/<사진 이름>.toml)")
    c.add_argument("--config")
    f = sub.add_parser("fake", help="시나리오로 가짜 Detection 생성")
    f.add_argument("scenario")
    f.add_argument("out")
    f.add_argument("--config")
    f.add_argument("--frames", type=int, default=30)
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
    elif args.cmd == "demo":
        cfg = _demo_config(args)
        sc = load_scenario(args.scenario)
        if args.detections:
            sc = _with_incident(sc, incident_from_frames(_frames(args, cfg), cfg))
        if not sc.zone:
            raise SystemExit("시나리오에 사고 범위(zone)가 없습니다")
        sc, plan, backend = _playback(sc, cfg)
        log_path = Path(args.log or f"logs/{sc.name}_demo.jsonl")
        if args.no_window:
            run(sc, cfg, backend, log_path)
        else:
            from .view.app import run_demo

            run_demo(sc, cfg, backend, log_path)
        print(f"{log_path} 저장")
    elif args.cmd == "present":
        from .presentation import Show, frame_views
        from .view.present import run_present

        cfg = _demo_config(args)
        sc = load_scenario(args.scenario)
        frames = _frames(args, cfg) if args.detections else fake_frames(sc, cfg)
        show = Show(frame_views(frames, cfg), frame_s=args.frame_time)
        sc = _with_incident(sc, show.result)  # 범위·차량은 프레임을 멈추는 지점의 것
        plan = backend = None
        if sc.zone:
            sc, plan, backend = _playback(sc, cfg)
        else:
            print("사고 범위가 확정되지 않아 전개 없이 탐지 화면만 보여 줍니다")
        log_path = Path(args.log or f"logs/{sc.name}_present.jsonl")
        run_present(sc, cfg, backend, plan, show, log_path, _images(args), args.speed, args.width)
        print(f"{log_path} 저장")
    elif args.cmd == "zone":
        cfg = load_config(args.config)
        result = incident_from_frames(_frames(args, cfg), cfg)
        if result.zone is None:
            print("확정한 사고 범위 없음")
        else:
            box = ", ".join(f"{v:.0f}" for v in result.zone.bbox)
            print(f"프레임 순번 {result.confirmed_frame}에서 확정: {result.zone.kind} "
                  f"{'/'.join(result.zone.lanes)} [{box}]")
        for v in result.vehicles:
            print(f"차량 {v.track_id} {v.label} [{', '.join(f'{c:.0f}' for c in v.bbox)}]")
    elif args.cmd == "fake":
        write_frames(fake_frames(load_scenario(args.scenario), load_config(args.config), args.frames),
                     args.out)
        print(f"{args.out} 저장 ({args.frames}프레임)")
    elif args.cmd == "calib":
        from .calib import default_road_points, parse_rect
        from .view.calib import run_calib

        cfg = load_config(args.config)
        out = Path(args.out or f"config/{Path(args.image).stem}.toml")
        try:
            road_points = default_road_points(cfg, parse_rect(args.rect) if args.rect else None)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        if run_calib(args.image, cfg, road_points, out):
            print(f"{out} 저장. 쓸 때는 --config {out}")
        else:
            print("저장하지 않고 닫았습니다")
    elif args.cmd == "view":
        from .view.app import run_view

        run_view(args.log)
    else:
        replay(args.log)


def _frames(args, cfg):
    """--detections를 읽는다. 사진 한 장이면 범위 확정에 필요한 만큼 되풀이한다."""
    try:
        return load_frames(args.detections, photo=args.photo, hold=cfg.incident.confirm_frames)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def _images(args) -> list[Path]:
    """발표 화면에 쓸 사진. --photo로 고른 사진이 --images 폴더에 있으면 그것 하나만 쓴다."""
    from .view.present import image_files

    files = image_files(args.images)
    if args.photo and args.detections and Path(args.detections).is_file():
        name = str(select_photo(read_vision_rows(args.detections), args.photo).get("file", ""))
        same = [f for f in files if f.name == name] or [f for f in files if f.stem == Path(name).stem]
        if same:
            return same[:1]
        if len(files) > 1:
            print(f"--images에 {name}이 없어 사진 없이 도로를 그려 보여 줍니다")
            return []
    return files


def _demo_config(args):
    """발표용 설정: --strict가 아니면 검사를 완화한다. 로그 헤더에도 남겨 재생 창이 같은 계획을 그리게 한다."""
    cfg = load_config(args.config)
    if not args.strict:
        cfg = config_from_raw({**cfg.raw, "planning": {**cfg.raw["planning"], "strict": False}})
    return cfg


def _with_incident(sc, result):
    """시나리오의 범위·차량을 Detection에서 구한 것으로 바꾼다. 범위가 안 나오면 시나리오 값도 쓰지 않는다."""
    base = {k: v for k, v in sc.raw.items() if k != "zone"}
    return scenario_from_raw({**base, **_incident_raw(result)})


def _playback(sc, cfg):
    """전개 계획과 그것을 재생할 백엔드. 통신 끊김 같은 시나리오 이벤트는 재생하지 않는다."""
    plan = plan_deployment(sc.zone, tuple(v.bbox for v in sc.vehicles), cfg.standby, cfg)
    for f in plan.failures + plan.warnings:
        print(f"{'실패' if f in plan.failures else '경고'} {f.code}: {f.detail}")
    backend = PlaybackBackend(cfg, plan)
    return replace(sc, duration_s=backend.duration_s, events=()), plan, backend


def _incident_raw(result) -> dict:
    """Detection에서 구한 범위·차량을 시나리오 형식으로 (로그 헤더에 남아 재생 창도 같은 것을 그린다)."""
    out = {"vehicles": [{"track_id": v.track_id, "label": v.label, "bbox": list(v.bbox)}
                        for v in result.vehicles]}
    if result.zone:
        out["zone"] = {"bbox": list(result.zone.bbox), "lanes": list(result.zone.lanes),
                       "kind": result.zone.kind}
    return out


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
