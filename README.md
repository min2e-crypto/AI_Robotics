# 펜스로봇 배치·시뮬레이션

고속도로 사고 지점을 카메라로 찾아 그물 펜스 로봇을 어디에 세울지 계산하고, 전개 과정을 화면으로 보여 주는 프로젝트입니다.

- 탐지 결과(차량 박스, 상태, 불)를 받아 **사고 범위**를 도로 좌표로 계산합니다.
- 사고 범위에 맞춰 **로봇 5대의 배치와 이동 경로**를 계획합니다.
- 탐지 → 범위 확정 → 전개가 이어지는 **발표 화면**을 띄웁니다.

지금은 실물 로봇 없이 시뮬레이션으로만 동작합니다. 좌표와 치수는 1/3 축척입니다.

## 준비

[uv](https://docs.astral.sh/uv/)가 필요합니다. 파이썬은 uv가 알아서 맞춰 설치합니다.

```bash
git clone https://github.com/min2e-crypto/AI_Robotics.git
cd AI_Robotics
uv sync
```

잘 설치됐는지는 테스트로 확인합니다.

```bash
uv run pytest -q
```

## 바로 실행해 보기

모델이나 사진 없이, 저장소에 들어 있는 예제만으로 됩니다.

```bash
uv run fencebot present scenarios/lane2_basic.toml
```

창 하나에서 탐지 → 사고 범위 확정 → 로봇 전개가 차례로 이어집니다. 빨리 보려면 `--speed 8`을 붙입니다.

| 명령 | 하는 일 |
|---|---|
| `uv run fencebot present scenarios/lane2_basic.toml` | 발표 화면 |
| `uv run fencebot demo scenarios/lane2_basic.toml` | 전개만 자동 재생 |
| `uv run fencebot zone examples/detections/lane2_basic` | 탐지 결과로 사고 범위만 계산해 출력 |
| `uv run fencebot sim scenarios/lane2_basic.toml` | 로봇을 손으로 조작하는 창 |
| `uv run fencebot run scenarios/lane2_basic.toml` | 창 없이 실행하고 `logs/`에 기록 |
| `uv run fencebot view logs/lane2_basic.jsonl` | 기록을 창으로 재생 |

옵션은 `uv run fencebot <명령> --help`로 볼 수 있습니다.

## 실제 사진으로 해 보기

비전 모델(`share_model/`)로 사진에서 차량을 찾고, 그 결과를 발표 화면에 넣는 순서입니다.

**1. 모델 가중치 받기.** 가중치는 용량이 커서 저장소 대신 GitHub 릴리스에 있습니다. 로그인 없이 받아집니다.

```bash
python share_model/weights.py fetch
```

**2. 모델 돌리기.** 모델은 별도 패키지가 필요합니다. 자세한 내용은 [share_model/README.md](share_model/README.md)에 있습니다.

```bash
pip install ultralytics pyyaml opencv-python
python share_model/vision.py --source 사진.jpg --name test1
```

`share_model/runs/test1/`에 결과 그림과 `results.jsonl`이 생깁니다.

**3. 도로 4점 찍기.** 사진의 픽셀을 도로 좌표로 바꾸려면 바닥의 네 점이 필요합니다. 창에서 순서대로 누르고 enter로 저장합니다.

```bash
uv run fencebot calib 사진.jpg --rect 2500,0,7000,1167
```

`--rect`는 네 점이 가리키는 도로 위 사각형(mm)입니다. 갓길 옆 차로의 양쪽 차선이 y 0과 1167입니다. 결과는 `config/사진.toml`에 저장됩니다.

**4. 발표 화면에 넣기.**

```bash
uv run fencebot present scenarios/lane2_basic.toml --config config/사진.toml --detections share_model/runs/test1/results.jsonl --photo 사진.jpg --images 사진.jpg
```

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `src/fencebot/` | 사고 범위 계산, 배치, 경로 계획, 시뮬레이터, 화면 |
| `share_model/` | 비전 모델 실행 코드와 가중치 받기 명령 |
| `config/default.toml` | 도로, 로봇, 카메라 등 기본 설정 |
| `scenarios/` | 시나리오 (사고 위치, 차량 배치) |
| `examples/detections/` | 탐지 결과 예제와 형식 설명 |
| `tests/` | 테스트 |
| [ADR.md](ADR.md) | 설계 결정과 그 이유 (코드 주석의 "ADR-번호"가 가리키는 곳) |
| [배치_시뮬레이션_계획.md](배치_시뮬레이션_계획.md) | 진행 계획과 현재 상태 |

## 알아 둘 점

- 로봇 배치는 **갓길 바로 옆 차로의 사고**만 지원합니다. 다른 차로의 사고는 경고만 나옵니다.
- 사진 한 장으로 넣으면 여러 프레임에 걸친 확인을 거치지 않으므로, 그 한 장의 오탐이 그대로 사고 범위가 됩니다.
- 비전 모델의 상태 분류와 화재 탐지는 임시 버전입니다. 한계는 [share_model/README.md](share_model/README.md)에 정리돼 있습니다.
- 도로 4점은 지금은 손으로 찍습니다. 실물 시연 환경에서는 바닥 마커로 자동화할 예정입니다.

## 모델을 새로 올릴 때 (모델 담당)

저장소 쓰기 권한과 [GitHub CLI](https://cli.github.com) 로그인이 필요합니다.

```bash
python share_model/weights.py publish models-v2
```

올린 뒤 고쳐진 `share_model/models/weights.json`을 커밋합니다.
