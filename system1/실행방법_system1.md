# System1 실행 방법

## 1. 시스템 개요

System1은 운전자의 졸음 상태를 판단하는 Multi-task 시스템입니다.

하나의 모델에서 다음 세 항목을 동시에 추론합니다.

* Eye: 눈 감김 여부
* Yawn: 하품 여부
* Head Pose: 고개 숙임 여부

Risk Score는 다음 비율로 계산됩니다.

```text
Risk Score
= 0.45 × Eye Closed Ratio
+ 0.25 × Yawn Score
+ 0.30 × Head Down Ratio
```

상태 판정 기준은 다음과 같습니다.

```text
NORMAL  : Risk Score < 0.40
WARNING : 0.40 <= Risk Score < 0.70
DANGER  : Risk Score >= 0.70
```

프로그램 실행 직후 약 2초 동안 정면을 바라보면
Head Pose의 기준 Pitch 값이 자동으로 보정됩니다.

현재 Webcam과 저장 영상은 모두 `multitask_exe.py`를 사용합니다.

---

## 2. 가상환경 생성 및 패키지 설치

System1 폴더에서 실행합니다.

```bash
cd system1

python -m venv --system-site-packages venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jetson에서는 CUDA가 적용된 PyTorch와 시스템 환경을 사용하기 위해
`--system-site-packages` 옵션을 사용합니다.

CUDA 사용 여부 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

GPU를 정상적으로 사용하는 경우 다음과 같이 `True`가 출력됩니다.

```text
True
```

프로그램 실행 시 CUDA를 사용할 수 있으면 GPU를 자동으로 사용하고,
CUDA를 사용할 수 없는 환경에서는 CPU를 사용합니다.

---

## 3. 필요한 모델 파일

기본 시연 모델:

```text
mobilevit_xxs_multitask_best.pth
```

추가로 다음 모델도 사용할 수 있습니다.

```text
mobilenetv2_multitask_best.pth
resnet18_multitask_best.pth
```

기본 시연에서는 `MobileViT-XXS` 모델을 사용합니다.

```text
mobilevit_xxs_multitask_best.pth
```

현재 `multitask_exe.py`는 `.pth` 체크포인트를 불러와 추론하는 방식입니다.

---

## 4. Webcam 실행

Webcam 실행도 `multitask_exe.py`를 사용합니다.

기본 실행:

```bash
python multitask_exe.py \
  --model ./mobilevit_xxs_multitask_best.pth \
  --camera 0 \
  --yawn-window-sec 60 \
  --min-yawn-sec 0.7 \
  --yawns-for-full-score 2
```

실행 후 약 2초 동안 정면을 바라봅니다.

```text
프로그램 시작
    ↓
약 2초간 정면 자세 유지
    ↓
Head Pitch 기준값 Calibration
    ↓
졸음 상태 추론 시작
```

화면에서는 다음 정보를 확인할 수 있습니다.

* Eye 상태
* 최근 구간의 Eye Closed Ratio
* Yawn 상태
* 최근 하품 횟수
* Yawn Score
* Head Pose
* Head Down Ratio
* Risk Score
* NORMAL / WARNING / DANGER
* FPS

종료:

```text
q
```

카메라가 열리지 않는 경우 Camera Index를 변경합니다.

```bash
python multitask_exe.py \
  --model ./mobilevit_xxs_multitask_best.pth \
  --camera 1
```

Webcam은 `--source`를 이용해서도 실행할 수 있습니다.

```bash
python multitask_exe.py \
  --model ./mobilevit_xxs_multitask_best.pth \
  --source 0
```

---

## 5. 저장 영상 실행

저장 영상도 동일한 `multitask_exe.py`를 사용합니다.

```bash
python multitask_exe.py \
  --model ./mobilevit_xxs_multitask_best.pth \
  --source ./test_video.mp4 \
  --yawn-window-sec 60 \
  --min-yawn-sec 0.7 \
  --yawns-for-full-score 2
```

저장 영상에서도 실행 초기 약 2초 동안의 얼굴 자세를 이용해
Head Pose 기준값을 자동으로 Calibration합니다.

종료:

```text
q
```

---

## 6. 저장 영상 반복 재생

영상 시연을 반복해서 진행하려면 `--loop` 옵션을 사용합니다.

```bash
python multitask_exe.py \
  --model ./mobilevit_xxs_multitask_best.pth \
  --source ./test_video.mp4 \
  --loop \
  --yawn-window-sec 60 \
  --min-yawn-sec 0.7 \
  --yawns-for-full-score 2
```

영상이 끝나면 처음부터 다시 재생됩니다.

반복 재생이 시작될 때 다음 상태도 초기화됩니다.

* Eye / Head 상태 기록
* Yawn Event 기록
* Head Pose 기준값
* Calibration 상태
* Risk Score 계산 상태

따라서 영상이 다시 시작되면 Head Pose Calibration도 다시 진행됩니다.

영상 한 번만 재생하려면 `--loop` 옵션을 제거합니다.

---

## 7. 다른 모델로 실행

동일한 `multitask_exe.py`에서 모델 파일만 변경하여
MobileNetV2 또는 ResNet18을 사용할 수 있습니다.

### MobileNetV2

```bash
python multitask_exe.py \
  --model ./mobilenetv2_multitask_best.pth \
  --camera 0
```

### ResNet18

```bash
python multitask_exe.py \
  --model ./resnet18_multitask_best.pth \
  --camera 0
```

### MobileViT-XXS

```bash
python multitask_exe.py \
  --model ./mobilevit_xxs_multitask_best.pth \
  --camera 0
```

저장 영상에서도 동일하게 모델 파일만 변경할 수 있습니다.

예시:

```bash
python multitask_exe.py \
  --model ./resnet18_multitask_best.pth \
  --source ./test_video.mp4
```

---

## 8. 기본 판단 설정

`multitask_exe.py`의 주요 기본 설정은 다음과 같습니다.

| 항목                       |  기본값 | 의미                        |
| ------------------------ | ---: | ------------------------- |
| `--window-sec`           | 3.0초 | Eye / Head 상태 계산 구간       |
| `--yawn-window-sec`      |  60초 | 하품 횟수를 계산하는 구간            |
| `--min-yawn-sec`         | 0.7초 | 하품 1회로 인정하기 위한 최소 지속시간    |
| `--yawns-for-full-score` |   2회 | Yawn Score가 1.0이 되는 하품 횟수 |
| `--calibration-sec`      | 2.0초 | Head Pose 초기 보정 시간        |
| `--head-down-deg`        |  15도 | 고개 숙임으로 판단하는 기준           |
| `--closed-threshold`     |  0.5 | 눈 감김 판정 Threshold         |
| `--yawn-threshold`       |  0.5 | 하품 판정 Threshold           |
| `--warning-threshold`    | 0.40 | WARNING 기준                |
| `--danger-threshold`     | 0.70 | DANGER 기준                 |

---

## 9. Yawn Score 계산

Yawn은 한 프레임의 결과만 사용하는 것이 아니라
일정 시간 동안 발생한 하품 횟수를 이용합니다.

기본 설정:

```text
Yawn Window      : 최근 60초
최소 지속시간      : 0.7초
Full Score 기준   : 2회
```

`P(Yawn)`이 Threshold 이상으로 0.7초 이상 유지되면
하품 1회로 기록합니다.

기본 설정에서는:

```text
최근 60초 하품 0회 → Yawn Score = 0.0
최근 60초 하품 1회 → Yawn Score = 0.5
최근 60초 하품 2회 → Yawn Score = 1.0
```

2회를 초과하더라도 최대 Yawn Score는 `1.0`입니다.

---

## 10. Head Pose Calibration

프로그램 시작 직후 약 2초 동안 현재 운전자의 정면 자세를 측정합니다.

```text
정면 자세 유지
   ↓
약 2초간 Pitch 값 수집
   ↓
Pitch 중앙값 계산
   ↓
Baseline Pitch 설정
```

이후 현재 Pitch와 Baseline Pitch의 차이를 계산합니다.

기본적으로 차이가 약 15도 이상이면
고개를 숙인 상태로 판단합니다.

따라서 사람마다 정면 자세나 카메라 설치 각도가 조금 달라도
초기 Calibration을 기준으로 상대적인 고개 움직임을 판단할 수 있습니다.

---

## 11. Risk Score 계산

최근 Eye / Yawn / Head 상태를 조합하여 최종 Risk Score를 계산합니다.

```text
Risk Score
= 0.45 × Eye Closed Ratio
+ 0.25 × Yawn Score
+ 0.30 × Head Down Ratio
```

가중치는 다음과 같습니다.

| 항목               |  가중치 |
| ---------------- | ---: |
| Eye Closed Ratio | 0.45 |
| Yawn Score       | 0.25 |
| Head Down Ratio  | 0.30 |

최종 상태:

```text
Risk < 0.40
→ NORMAL

0.40 <= Risk < 0.70
→ WARNING

Risk >= 0.70
→ DANGER
```

초기 Calibration 및 필요한 상태 데이터가 충분히 쌓이기 전에는
`WARMUP` 상태로 표시될 수 있습니다.

---

## 12. 주요 실행 옵션

```text
--model
    사용할 .pth 모델 파일

--source
    Webcam 번호 또는 저장 영상 경로

--camera
    Webcam Camera Index

--loop
    저장 영상 반복 재생

--playback-speed
    저장 영상 재생 속도

--width
    Webcam 입력 가로 해상도

--height
    Webcam 입력 세로 해상도

--window-width
    출력 화면 가로 크기

--window-height
    출력 화면 세로 크기

--window-sec
    Eye / Head 상태 계산 시간

--yawn-window-sec
    하품 횟수 계산 시간

--min-yawn-sec
    하품 1회 인정 최소 지속시간

--yawns-for-full-score
    Yawn Score 1.0 기준 하품 횟수

--calibration-sec
    Head Pose 초기 Calibration 시간

--head-down-deg
    고개 숙임 판정 각도

--warning-threshold
    WARNING Risk Score 기준

--danger-threshold
    DANGER Risk Score 기준
```

---

## 13. 실행 시 확인사항

* 현재 Webcam과 저장 영상 실행 파일은 모두 `multitask_exe.py`입니다.
* 기본 시연 모델은 `mobilevit_xxs_multitask_best.pth`입니다.
* 실행 직후 약 2초 동안 정면을 바라봅니다.
* 기본 Webcam 번호는 `0`입니다.
* Webcam이 열리지 않으면 `--camera 1` 등으로 변경합니다.
* 기본 Webcam 입력 해상도는 1280×720입니다.
* 기본 출력 Window 크기는 1200×540입니다.
* Head Pose는 초기 정면 자세를 기준으로 자동 보정됩니다.
* Eye / Head 상태는 기본적으로 최근 3초 구간을 이용합니다.
* Yawn Score는 기본적으로 최근 60초의 하품 횟수를 이용합니다.
* 저장 영상 반복 재생에는 `--loop` 옵션을 사용합니다.
* 프로그램 종료는 `q` 키를 사용합니다.

---

## 14. 주요 실행 명령 요약

### MobileViT-XXS Webcam

```bash
python multitask_exe.py \
  --model ./mobilevit_xxs_multitask_best.pth \
  --camera 0 \
  --yawn-window-sec 60 \
  --min-yawn-sec 0.7 \
  --yawns-for-full-score 2
```

### MobileViT-XXS 저장 영상

```bash
python multitask_exe.py \
  --model ./mobilevit_xxs_multitask_best.pth \
  --source ./test_video.mp4 \
  --yawn-window-sec 60 \
  --min-yawn-sec 0.7 \
  --yawns-for-full-score 2
```

### 저장 영상 반복 재생

```bash
python multitask_exe.py \
  --model ./mobilevit_xxs_multitask_best.pth \
  --source ./test_video.mp4 \
  --loop
```

### MobileNetV2 Webcam

```bash
python multitask_exe.py \
  --model ./mobilenetv2_multitask_best.pth \
  --camera 0
```

### ResNet18 Webcam

```bash
python multitask_exe.py \
  --model ./resnet18_multitask_best.pth \
  --camera 0
```

### CUDA 확인

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
