# 가짜 Detection 예제 (팀원 인터페이스 초안)

배치_시뮬레이션_계획 2절의 제안안을 파일로 만든 것입니다. 팀원과 합의하면 이 문서와 `frame_000120.json`을 함께 고칩니다(0단계).

## 파일

- 프레임 하나에 JSON 파일 하나, 이름은 `frame_<6자리 프레임 번호>.json`
- 좌표는 모두 **이미지 픽셀** (원점 왼쪽 위, x 오른쪽, y 아래). 도로 좌표 변환은 배치 쪽에서 합니다.

## 필드

| 필드 | 형식 | 설명 |
|---|---|---|
| `frame` | 정수 | 프레임 번호 |
| `timestamp` | 실수(초) | 촬영 시각. 영상 시작 기준 |
| `detections[].cls` | 문자열 | `car`, `fire`, `smoke` (추가 시 합의) |
| `detections[].confidence` | 0~1 | 탐지 확신도 |
| `detections[].bbox_px` | [x1, y1, x2, y2] | 픽셀 박스 |
| `detections[].track_id` | 정수, 선택 | 차량 추적 ID. 프레임이 바뀌어도 같은 차량은 같은 값. 차량에는 필수, 화염·연기는 생략 가능 |
| `vehicle_states[].track_id` | 정수 | 위 차량의 추적 ID |
| `vehicle_states[].label` | 문자열 | `normal`, `damaged`, `overturned` |
| `vehicle_states[].scores` | {클래스: 확신도} | 상태별 확신도. 애매 판정은 배치 쪽에서 이 값으로 함 |
| `lanes_px` | {선 이름: [[x, y], ...]} | 선택. 차선 모델 출력. 선 이름은 `shoulder_line`, `lane_boundary`, `center_line` |

읽는 코드: `fencebot.detection_io.load_frame`

## 연속 프레임 예제

`lane2_basic/`은 `scenarios/lane2_basic.toml`에서 만든 가짜 Detection 10프레임입니다(`uv run fencebot fake scenarios/lane2_basic.toml examples/detections/lane2_basic --frames 10`). 설정 `[camera]`의 임시 카메라(갓길 쪽에서 비스듬히 내려다봄) 기준이라, `frame_000120.json`과는 좌표가 맞지 않습니다.

사고 범위 계산은 차량 박스의 **아랫변**을 바닥 접점으로 씁니다(ADR-015). 박스 아랫변이 차량이 바닥에 닿는 곳에 맞을수록 범위가 정확합니다.

## 팀원 모델 묶음(`share_model/vision.py`) 결과 읽기

모델 가중치(`share_model/models/*/*.pt`, 약 61MB)는 저장소에 넣지 않습니다. 팀원에게 받아 `share_model/models/` 아래 같은 폴더에 둡니다. 모델을 돌리려면 `share_model/README.md`의 설치 안내를 따릅니다(fencebot 환경과 따로).

`vision.py`가 쓰는 `runs/<name>/results.jsonl`(한 줄이 한 프레임)도 그대로 읽습니다. 폴더 대신 파일 경로를 주면 됩니다.

```bash
uv run fencebot zone share_model/runs/test1/results.jsonl
uv run fencebot demo scenarios/lane2_basic.toml --detections share_model/runs/test1/results.jsonl
```

옮기는 규칙 (`fencebot.detection_io.vision_frame`)

| vision.py | 여기 | 비고 |
|---|---|---|
| `vehicles[].box`, `det_conf` | `detections[].bbox_px`, `confidence` | |
| `vehicles[].vtype` (`vehicle`·`car`·`bus`·`truck`) | `cls = "car"` | 차종은 구분하지 않음 |
| `vehicles[].id` | `track_id` | 사진 안에서만 뜻이 있음. 프레임 사이 추적 아님 |
| `vehicles[].state`, `probs` | `vehicle_states[].label`, `scores` | `uncertain`은 그대로 넘어오고, 범위 계산에서는 사고 차량으로 보지 않음 |
| `fires[].cls`, `conf`, `box` | `detections[]` (`fire`·`smoke`) | |
| `frame` | `frame`, `timestamp = frame / 30` | 사진 폴더 결과(모두 0)는 줄 순번을 씀 |

판정 규칙은 팀원이 정한 것(`share_model/정해둔_규칙.md`)에 맞췄습니다. 연기는 쓰지 않고, 파손·전복 차량이 없으면 불만으로는 범위를 만들지 않습니다. 여유 거리만 그물 길이 때문에 다르게 둡니다 (ADR-016).
