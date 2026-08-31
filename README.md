# On-Device AI Vehicle Safety Monitoring System

Jetson Orin Nano에서 동작하는 **On-Device AI 기반 차량 통합 안전 모니터링 시스템**입니다.

운전자 졸음 상태, 차량 내부 잔류 탑승자, 차량 접근 보안을 각각 분석하고 Web UI 및 알림 시스템과 연동합니다.

## System Overview

전체 시스템은 세 개의 AI Subsystem으로 구성됩니다.

1. **Driver Monitoring** — 운전자 졸음 위험 판단
2. **Rear-seat Occupant Monitoring** — 아동 및 반려동물 잔류 감지
3. **Vehicle Security Camera** — 등록 사용자 / 미등록 사용자 / Spoof 판별

```text
Camera Input
     |
     +----------------------+----------------------+----------------------+
     |                      |                      |
     v                      v                      v
 Driver Monitoring     Rear-seat Monitoring   Security Camera
     |                      |                      |
     v                      v                      v
 Risk State            Occupant Stage       Identity / Spoof State
     |                      |                      |
     +----------------------+----------------------+
                            |
                            v
                    Alert / Web Server
                            |
                    Web UI / Telegram
```

## 1. Driver Monitoring

한 개의 Multi-task Model에서 다음 정보를 동시에 추론합니다.

- Eye: Open / Closed
- Yawn: Yawn / No Yawn
- Head Pose: Normal / Down

### Risk Score

```text
Risk Score
= 0.45 × Eye Closed Ratio
+ 0.25 × Yawn Score
+ 0.30 × Head Down Ratio
```

| State | Condition |
|---|---|
| NORMAL | Score < 0.40 |
| WARNING | 0.40 ≤ Score < 0.70 |
| DANGER | Score ≥ 0.70 |

시작 시 약 2초간 정면 자세를 기준으로 Head Pose Baseline을 자동 보정합니다.

### Compared Models

- ResNet18
- MobileNetV2
- MobileViT-XXS

## 2. Rear-seat Occupant Monitoring

```text
YOLO11
   ↓
person / dog / cat Detection
   ↓
MiVOLO V2
   ↓
Age Estimation
   ↓
Child(≤ 7 years) / Animal
   ↓
Vehicle State + Duration
   ↓
Risk Stage
   ↓
Alert Server
```

### Final Detection Model

**Fine-tuned YOLO11n**

| Model | Accuracy | Young Recall | Animal Recall |
|---|---:|---:|---:|
| YOLO11n | 0.791469 | 0.753086 | 0.866667 |
| YOLO11s | 0.819905 | 0.716049 | 0.950000 |
| YOLO11m | 0.796209 | 0.679012 | 0.950000 |
| **YOLO11n Fine-tuned** | **0.962085** | **0.938272** | **0.983333** |

아동과 동물의 인식 성능을 동시에 개선한 Fine-tuned YOLO11n을 최종 모델로 선정했습니다.

## 3. Vehicle Security Camera

```text
Camera
  ↓
YuNet Face Detection
  ↓
MiniFASNetV2 Anti-Spoofing
  ↓
SFace Face Recognition
  ↓
OWNER / GUEST / UNKNOWN / SPOOF
  ↓
Event / Alert
```

### Security Policy

- `OWNER` / `GUEST`: 등록 사용자
- `UNKNOWN`: 미등록 사용자
- `SPOOF`: 사진 / 화면 등을 이용한 위조 입력

`UNKNOWN` 상태가 지속될 경우:

- **10초** → Capture
- **30초** → Alarm

## AI Models

| Task | Model |
|---|---|
| Driver Multi-task Classification | ResNet18 / MobileNetV2 / MobileViT-XXS |
| Person / Animal Detection | YOLO11 |
| Age Estimation | MiVOLO V2 |
| Face Detection | YuNet |
| Anti-Spoofing | MiniFASNetV2 |
| Face Recognition | SFace |

## Optimization

Jetson Orin Nano의 실시간 추론 성능을 고려해 다음 실행 방식을 활용합니다.

- PyTorch CUDA
- TensorRT FP16
- TorchScript
- ONNX

## Hardware / Software

- NVIDIA Jetson Orin Nano
- JetPack 6.2.2
- CUDA 12.6
- Python
- PyTorch
- TensorRT
- OpenCV
- ONNX

## Repository Structure

```text
Car_Safety_Based_Jetson/
├── system1/        # Driver Monitoring
├── system2/        # Rear-seat Occupant Monitoring
├── system3/        # Security Camera
├── web/            # Web UI / Server
├── models/         # Model files
└── README.md
```

> 실제 디렉터리 구조에 맞게 조정해 주세요.

## Key Contributions

- 서로 다른 차량 안전 문제를 하나의 Edge AI Platform으로 통합
- Cloud 의존도를 낮춘 On-Device Inference
- Accuracy뿐 아니라 실제 실시간 실행 가능성을 고려한 Model Selection
- 상태 지속시간을 반영한 단계별 Risk / Security Policy
- Web UI 및 외부 알림 시스템 연동
