# 펜스로봇 비전 모델 묶음 (share_vision)

- 만든 날: 2026-11-12 / 담당: 재강
- 사진(또는 영상 프레임) 한 장을 넣으면 **차량 박스 + 차량 상태(정상·파손·전복·애매) + 불·연기 박스**를 돌려준다.
- 좌표는 전부 **픽셀**이다. 도로 mm 좌표 변환과 사고 범위 계산은 들어 있지 않다 (아래 "여기서 하지 않는 것").
- **2번(상태 분류)과 3번(화재) 모델은 임시 버전이다.** 나중에 `best.pt`만 바꿔 넣으면 되도록 만들어 두었다.

## 폴더 구성

```
share_vision/
├─ README.md               ← 지금 이 파일
├─ vision.py               ← 세 모델을 한 번에 돌리는 코드
├─ vision_config.yaml      ← 모델 경로·기준값
├─ weights.py              ← 가중치를 GitHub 릴리스에서 받고 올리는 명령
├─ 정해둔_규칙.md           ← (참고) 사고 범위 계산에 쓰려고 정해 둔 규칙과 근거
└─ models/
   ├─ 1_vehicle/  best.pt, yolo26s_coco.pt, README.md
   ├─ 2_state/    best.pt, README.md
   └─ 3_fire/     best.pt, README.md
```

## 설치

```bash
pip install ultralytics pyyaml opencv-python
```
- 재강 PC에서 쓴 버전: Python 3.14.3, PyTorch 2.15.0 (CUDA), **Ultralytics 8.4.163**. 버전이 크게 다르면 결과가 조금 달라질 수 있다.
- 동작 확인 (11-12): Linux + CPU + Ultralytics 8.4.211에서 시험 사진 2장 결과가 재강 PC 결과와 같게 나옴 (확률 차이 0.002 이하).
- 장치는 자동 선택: CUDA → Apple `mps` → `cpu`. 맥북에서는 `mps`로 돈다 (**맥에서 실제로 돌려 본 적은 아직 없음**. 안 되면 `--device cpu`).

## 사용법

### 명령으로 확인
```bash
python vision.py --source 사진.jpg
python vision.py --source 사진폴더 --name test1
python vision.py --source 영상.mp4 --every 10      # 10프레임마다 한 번
```
→ `runs/<name>/`에 결과 그림과 `results.jsonl`이 생긴다.
그림 색: 초록 = 정상, 주황 = 파손, 빨강 = 전복, 흰색 = 애매(`?이유`), 주황 박스 = 불, 회색 박스 = 연기.

### 코드에서
```python
from vision import VisionModels, imread
vm = VisionModels("vision_config.yaml")   # 처음 한 번만 (모델 4개를 읽음)
r = vm.run(imread("사진.jpg"))            # OpenCV BGR 이미지(numpy)를 넣어도 됨: vm.run(frame)
for v in r["vehicles"]:
    print(v["id"], v["state"], v["box"])
```

## 출력 형식 (`vm.run()`이 돌려주는 dict)

```json
{
  "image_size": [1920, 1080],
  "vehicles": [
    {"id": 0, "box": [414.3, 867.3, 784.2, 1019.0], "det_conf": 0.892, "vtype": "car",
     "probs": {"damaged": 0.012, "normal": 0.337, "overturned": 0.65},
     "state": "overturned", "rule": "top1"}
  ],
  "fires": [
    {"id": 0, "cls": "fire", "conf": 0.48, "box": [1045.7, 603.7, 1555.7, 893.9]}
  ]
}
```

| 필드 | 뜻 |
|---|---|
| `box` | `[x1, y1, x2, y2]` 픽셀. 왼쪽 위가 (0, 0) |
| `det_conf` | 1번 탐지 확신도 |
| `vtype` | 박스를 낸 모델의 클래스 이름. 1번 모델이면 `vehicle`, COCO 보조 모델이면 `car`/`bus`/`truck` |
| `probs` | 2번 확률. 박스가 너무 작으면(긴 변 24px 미만) `null` |
| `state` | `normal` / `damaged` / `overturned` / `uncertain` |
| `rule` | `state`를 정한 이유. (위 예시는 실제 시험 사진 결과인데, 사실은 **파손된 차를 '전복'으로 판정한 오답**이다 — 2번 README 참고) `top1`(1등 그대로), `low_conf`(최고 확률 0.6 미만), `risky_normal`(정상 1등이지만 파손+전복 ≥ 0.3), `too_small`, `small_margin` |
| `fires[].cls` | `fire`(불꽃) 또는 `smoke`(연기) |

- **`uncertain`은 학습한 클래스가 아니라 규칙**이다. 모델이 자신 없으면 관제자에게 넘기라는 뜻.
- 차량 id는 사진 안에서만 의미가 있다 (영상 프레임 사이 추적은 하지 않음).

## 여기서 하지 않는 것

1. **픽셀 → 도로 mm 좌표 변환.** 공통 규격(구현 계획 4절: x = 진행 방향, y = 가로, mm)으로 바꾸는 부분은 따로 만들어야 한다.
   - 참고: 박스의 **아랫변 가운데 점**이 차가 땅에 닿은 점에 가장 가깝다. 박스 네 모서리를 바닥으로 보면 비스듬한 카메라에서 차 길이가 3~6배로 부풀었다 (재강 시험, 11-12).
2. **사고 범위(Zone) 계산.** 어떤 차를 묶고 여유를 얼마나 둘지는 `정해둔_규칙.md`에 정리해 두었다.
3. **차선 인식(4번).** 보류. 지금은 클릭으로 차선을 찍어 설정 파일로 쓰는 방식.

## 모델 상태 요약 (자세한 내용은 각 폴더 README)

| # | 모델 | 상태 | 한 줄 요약 |
|---|---|---|---|
| 1 | 차량 탐지 | 학습 완료 | 승용차는 잘 잡음. 화면을 채우는 큰 트럭은 박스가 쪼개지거나 부정확, 직사각형 물체(블록 더미 등) 오탐 |
| 2 | 상태 분류 | **임시 (재학습 예정)** | 위에서 본 승용차 위주로 배움. **비스듬한 CCTV 시점의 파손·옆으로 누운 트럭을 '정상'으로 확신하는 경우가 많음** |
| 3 | 화재 탐지 | **임시 (v1/v2 선택 전)** | 지상 시점 데이터로 배움. 밤 조명·전조등·후미등을 불로 잡는 경우 있음 |

실제 CCTV 사고 사진 30장(TAD, 학습에 안 쓴 사진) 시험 결과 (11-12): 사고 차를 위험으로 잡은 사진 6장, 애매로 넘긴 사진 5장, **놓친 사진 19장**. 놓친 원인 대부분이 2번 모델.

## 가중치 받기·올리기 (GitHub Releases)

가중치(`.pt`)는 용량이 커서 저장소에 없다. GitHub 릴리스에 첨부해 두고, 어느 릴리스의 어느 파일을 쓰는지만 `models/weights.json`에 적는다.
저장소가 공개라서 받는 쪽은 로그인도 초대도 필요 없다.

```bash
python weights.py fetch     # git pull 한 뒤: weights.json과 다른 가중치만 받아 models/ 아래에 채움
python weights.py check     # 지금 가진 가중치가 weights.json과 같은지 확인
```

모델을 새로 학습했을 때 (모델 담당). 저장소 쓰기 권한(협업자 초대)과 GitHub CLI가 필요하다: [cli.github.com](https://cli.github.com)에서 설치하고 `gh auth login`.

```bash
python weights.py publish models-v2 --dry-run   # 무엇이 올라갈지 먼저 확인
python weights.py publish models-v2             # 릴리스를 만들어 올리고 weights.json을 고침
```
그다음 고쳐진 `models/weights.json`을 커밋해 올린다. `vision.py`나 `vision_config.yaml`도 바뀌었으면 같은 커밋에 넣는다.

- 태그는 `models-v1`, `models-v2`처럼 매번 새로 붙인다. 이미 있는 태그로는 올라가지 않는다.
- `models/` 아래의 `.pt`가 전부 올라간다. 쓰지 않는 시험용 `.pt`는 올리기 전에 다른 곳으로 옮긴다.
- 릴리스의 파일 이름은 폴더 이름을 앞에 붙인 것이다 (`2_state/best.pt` → `2_state-best.pt`).

## 모델 바꾸기

- 새 `best.pt`를 해당 폴더에 덮어쓰거나, `vision_config.yaml`의 경로만 바꾸면 된다. 코드는 그대로. 다른 사람도 쓰게 하려면 위의 `publish`로 올린다.
- 2번 모델을 바꿀 때는 **자르기 방식(`pad` 0.15, 정사각형)과 입력 크기(224)가 학습 때와 같은지** 확인한다.
