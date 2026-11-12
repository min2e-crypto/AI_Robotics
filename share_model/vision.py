"""펜스로봇 비전 모델 묶음 — 사진(또는 영상 프레임) 한 장 → 차량·상태·불/연기 결과 (픽셀 좌표).

    1번 차량 탐지 (VisDrone 학습 + COCO yolo26s 보조, 두 결과를 합침)
    → 차량마다 2번 상태 분류 (정상 / 파손 / 전복, 애매하면 'uncertain')
    → 3번 불·연기 탐지

사고 범위 계산, 픽셀 → 도로 mm 좌표 변환은 여기 들어 있지 않다 (README 참고).

코드에서 쓰기:
    from vision import VisionModels, imread
    vm = VisionModels("vision_config.yaml")          # 장치 자동 선택: CUDA → mps → cpu
    result = vm.run(imread("사진.jpg"))               # dict (README "출력 형식")

명령으로 확인하기:
    python vision.py --source 사진.jpg
    python vision.py --source 사진폴더 --name test1
    python vision.py --source 영상.mp4 --every 10     # 10프레임마다 한 번
    → runs/<name>/ 에 결과 그림과 results.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}
VEH_NAMES = {"car", "van", "truck", "bus", "vehicle", "cars"}  # COCO 모델에서 차량으로 쓸 클래스

NORMAL, DAMAGED, OVERTURNED, UNCERTAIN = "normal", "damaged", "overturned", "uncertain"
DANGER = (DAMAGED, OVERTURNED)


# ───────────── 입출력 (한글·공백 경로에서도 동작) ─────────────
def imread(p) -> np.ndarray | None:
    data = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(p, im) -> None:
    p = Path(p)
    ok, buf = cv2.imencode(p.suffix or ".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(str(p))


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ───────────── 2번 모델: 학습 때와 반드시 같아야 하는 부분 ─────────────
def crop_square(img: np.ndarray, box_xyxy, pad: float = 0.15, fill: int = 114) -> np.ndarray:
    """박스 중심 기준 정사각형(긴 변 × (1 + 2·pad))으로 자른다. 화면 밖은 회색(114).
    2번 학습 데이터를 만들 때 쓴 함수(vehicle_state_classifier/common.py)와 똑같다. 바꾸면 성능이 떨어진다."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(int(round(max(x2 - x1, y2 - y1) * (1 + 2 * pad))), 2)
    sx1, sy1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    sx2, sy2 = sx1 + side, sy1 + side
    out = np.full((side, side, img.shape[2]) if img.ndim == 3 else (side, side), fill, dtype=img.dtype)
    ix1, iy1, ix2, iy2 = max(sx1, 0), max(sy1, 0), min(sx2, w), min(sy2, h)
    if ix2 > ix1 and iy2 > iy1:
        out[iy1 - sy1:iy2 - sy1, ix1 - sx1:ix2 - sx1] = img[iy1:iy2, ix1:ix2]
    return out


def decide_state(p: dict | None, s: dict) -> tuple[str, str]:
    """확률 → 최종 상태와 그 이유(rule).
      too_small    : 박스가 작아 분류하지 않음 → uncertain
      low_conf     : 최고 확률 < conf → uncertain
      small_margin : 1등 - 2등 < margin → uncertain (3클래스에서 conf 0.6이면 사실상 작동 안 함)
      risky_normal : 정상이 1등이어도 파손+전복 확률 합 ≥ risky_sum → uncertain
      top1         : 1등 클래스 그대로
    """
    if p is None:
        return UNCERTAIN, "too_small"
    ranked = sorted(p.items(), key=lambda kv: kv[1], reverse=True)
    (top, p1), (_, p2) = ranked[0], ranked[1]
    if p1 < s["conf"]:
        return UNCERTAIN, "low_conf"
    if p1 - p2 < s["margin"]:
        return UNCERTAIN, "small_margin"
    if top == NORMAL and sum(p.get(c, 0.0) for c in DANGER) >= s["risky_sum"]:
        return UNCERTAIN, "risky_normal"
    return top, "top1"


# ───────────── 박스 계산 ─────────────
def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def inside_ratio(a, b) -> float:
    """겹친 넓이 / 작은 박스 넓이. 1이면 작은 박스가 큰 박스 안에 완전히 들어감."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def train_imgsz(model, default=640) -> int:
    return int((getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", default))


# ───────────── 모델 묶음 ─────────────
class VisionModels:
    def __init__(self, config=HERE / "vision_config.yaml", device: str | None = None, verbose: bool = True):
        from ultralytics import YOLO
        config = Path(config)
        self.cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        m = {k: (str((config.parent / v).resolve()) if v and str(v).lower() != "none" else None)
             for k, v in self.cfg["models"].items()}
        self.device = device or pick_device()
        for k in ("vehicle", "state", "fire"):
            if not m[k] or not Path(m[k]).exists():
                raise SystemExit(f"모델 파일이 없음: {m[k]}  (vision_config.yaml의 models.{k} 확인)")
        self.vehicle = []
        for k in ("vehicle", "coco"):
            if not m.get(k):
                continue
            mdl = YOLO(m[k])
            ids = [i for i, n in mdl.names.items() if str(n).lower() in VEH_NAMES] or list(mdl.names)
            self.vehicle.append((mdl, ids, train_imgsz(mdl)))
        self.state = YOLO(m["state"])
        self.fire = YOLO(m["fire"])
        self.fire_imgsz = train_imgsz(self.fire)
        if verbose:
            print(f"장치: {self.device}")
            for mdl, ids, sz in self.vehicle:
                print(f"  차량 탐지: {Path(mdl.ckpt_path).name if getattr(mdl, 'ckpt_path', None) else ''} "
                      f"클래스 {[mdl.names[i] for i in ids]} imgsz {sz}")
            print(f"  상태 분류: {m['state']}\n  화재 탐지: {m['fire']} imgsz {self.fire_imgsz}")

    # 1번
    def detect_vehicles(self, im) -> list[dict]:
        d = self.cfg["detect"]
        found = []
        for mdl, ids, imgsz in self.vehicle:
            r = mdl.predict(im, imgsz=imgsz, conf=d["vehicle_conf"], classes=ids, device=self.device, verbose=False)[0]
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                found.append((tuple(float(t) for t in b), float(c), str(mdl.names[int(k)]).lower()))
        found.sort(key=lambda t: -t[1])
        keep = []
        for b, c, name in found:  # 두 모델이 같은 차를 잡으면 확신도 높은 것 하나만
            if all(iou(b, k["box"]) <= d["merge_iou"] and inside_ratio(b, k["box"]) < d["merge_inside"] for k in keep):
                keep.append(dict(id=len(keep), box=[round(t, 1) for t in b], det_conf=round(c, 3), vtype=name))
        return keep

    # 2번
    def classify(self, im, vehicles: list[dict]) -> list[dict]:
        s = self.cfg["state"]
        crops, idx = [], []
        for v in vehicles:
            x1, y1, x2, y2 = v["box"]
            if max(x2 - x1, y2 - y1) < self.cfg["detect"]["min_size"]:
                v["probs"], (v["state"], v["rule"]) = None, decide_state(None, s)
            else:
                crops.append(crop_square(im, v["box"], s["pad"]))
                idx.append(v)
        if crops:
            res = self.state.predict(crops, imgsz=s["imgsz"], device=self.device, verbose=False)
            for v, rr in zip(idx, res):
                p = rr.probs.data.cpu().numpy()
                v["probs"] = {rr.names[i]: round(float(p[i]), 3) for i in range(len(p))}
                v["state"], v["rule"] = decide_state(v["probs"], s)
        return vehicles

    # 3번
    def detect_fire(self, im) -> list[dict]:
        r = self.fire.predict(im, imgsz=self.fire_imgsz, conf=self.cfg["detect"]["fire_conf"],
                              device=self.device, verbose=False)[0]
        return [dict(id=i, cls=str(r.names[int(k)]), conf=round(float(c), 3), box=[round(float(t), 1) for t in b])
                for i, (b, c, k) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                                  r.boxes.cls.cpu().numpy()))]

    def run(self, im) -> dict:
        """사진 한 장 → {"image_size", "vehicles", "fires"} (픽셀 좌표)."""
        h, w = im.shape[:2]
        vehicles = self.classify(im, self.detect_vehicles(im))
        return {"image_size": [w, h], "vehicles": vehicles, "fires": self.detect_fire(im)}


# ───────────── 확인용 그림 ─────────────
COL = {NORMAL: (80, 200, 80), DAMAGED: (0, 160, 255), OVERTURNED: (60, 60, 255), UNCERTAIN: (255, 255, 255)}


def draw(im, result: dict):
    out = im.copy()
    t = max(1, int(max(im.shape[:2]) / 1000))
    for f in result["fires"]:
        x1, y1, x2, y2 = map(int, f["box"])
        col = (0, 80, 255) if f["cls"] == "fire" else (160, 160, 160)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, t)
        cv2.putText(out, f"{f['cls']} {f['conf']:.2f}", (x1, y2 + 14 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, col, t)
    for v in result["vehicles"]:
        x1, y1, x2, y2 = map(int, v["box"])
        col = COL[v["state"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, (2 if v["state"] != NORMAL else 1) * t)
        p = max(v["probs"].values()) if v["probs"] else 0
        txt = f"{v['id']} ?{v['rule']}" if v["state"] == UNCERTAIN else f"{v['id']} {v['state']} {p:.2f}"
        cv2.putText(out, txt, (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * t, col, t, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="사진, 사진 폴더, 또는 영상")
    ap.add_argument("--config", default=str(HERE / "vision_config.yaml"))
    ap.add_argument("--name", default="run")
    ap.add_argument("--every", type=int, default=1, help="영상: N프레임마다 한 번")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    vm = VisionModels(a.config, a.device)
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "results.jsonl", "w", encoding="utf-8")
    src = Path(a.source)

    def one(name, i, im):
        r = vm.run(im)
        r.update(file=name, frame=i)
        log.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_d = sum(v["state"] in DANGER for v in r["vehicles"])
        n_u = sum(v["state"] == UNCERTAIN for v in r["vehicles"])
        print(f"  {name} #{i}: 차량 {len(r['vehicles'])} (위험 {n_d}, 애매 {n_u}), 불·연기 {len(r['fires'])}")
        return r

    if src.is_dir() or src.suffix.lower() in IMG_EXT:
        files = sorted(p for p in src.glob("*") if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
        for p in files:
            im = imread(p)
            if im is None:
                print(f"  읽기 실패: {p}")
                continue
            imwrite(out / p.name, draw(im, one(p.name, 0, im)))
    elif src.suffix.lower() in VID_EXT:
        cap = cv2.VideoCapture(str(src))
        i = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if i % a.every == 0:
                r = one(src.name, i, im)
                imwrite(out / f"{src.stem}_{i:06d}.jpg", draw(im, r))
            i += 1
        cap.release()
    else:
        raise SystemExit(f"지원하지 않는 입력: {src}")
    log.close()
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
