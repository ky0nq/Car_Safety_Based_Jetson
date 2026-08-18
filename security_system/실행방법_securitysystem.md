# Security System 실행 방법

## 1. 시스템 개요

Security System은 차량 내부의 얼굴을 분석하여
등록 사용자, 비등록 사용자, 위조 얼굴을 구분하는 보안 카메라 시스템입니다.

분류 결과:

```text
OWNER   : 등록된 차량 소유자
GUEST   : 등록된 허용 사용자
UNKNOWN : 등록되지 않은 사용자
SPOOF   : 사진 또는 화면 등의 위조 얼굴
```

주요 처리 과정:

```text
YuNet 얼굴 검출
  ↓
MiniFASNetV2 Anti-Spoofing
  ↓
SFace 얼굴 인식
  ↓
OWNER / GUEST / UNKNOWN / SPOOF
  ↓
Capture / Web Alarm
```

현재 실행 파일은 `security_exe.py`로 통합되어 있으며,
Webcam과 저장 영상 모두 해당 파일을 사용합니다.

`security_exe.py` 실행 시 Web UI 서버도 자동으로 시작됩니다.

---

## 2. 필요한 모델 파일

실행 전 다음 파일이 `models/` 폴더에 있어야 합니다.

```text
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
models/2.7_80x80_MiniFASNetV2.pth
```

---

## 3. 가상환경 생성 및 패키지 설치

```bash
cd security_system

python -m venv --system-site-packages venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

환경 및 모델 확인:

```bash
python verify_setup.py
```

---

## 4. OWNER / GUEST 등록

처음 사용하는 경우 얼굴을 등록합니다.

### OWNER 등록

```bash
python register_person.py \
  --name owner1 \
  --role OWNER \
  --camera 0
```

### GUEST 등록

```bash
python register_person.py \
  --name guest1 \
  --role GUEST \
  --camera 0
```

기본적으로 한 사람당 얼굴 샘플 15개를 수집합니다.

등록 정보는 다음 DB에 저장됩니다.

```text
database/face_database.sqlite3
```

등록 사용자 확인:

```bash
python manage_people.py list
```

이미 필요한 OWNER/GUEST가 등록되어 있다면 이 과정은 생략합니다.

---

## 5. Webcam 실행

현재 Webcam 실행 파일은 `security_exe.py`입니다.

```bash
python security_exe.py --camera 0
```

실행 시 Web UI 서버도 자동으로 시작되므로
`web_alarm.py`를 별도로 실행할 필요가 없습니다.

실행 키:

```text
o   : OWNER 등록
g   : GUEST 등록
l   : 등록 사용자 목록 확인
q   : 종료
ESC : 종료
```

프로그램 화면의 버튼을 이용해서도 OWNER/GUEST 등록 및 사용자 목록 확인이 가능합니다.

---

## 6. Webcam 기본 경고 기준

`security_exe.py`의 기본 경고 시간은 다음과 같습니다.

```text
UNKNOWN 감지
   ↓
10초 지속
   ↓
사진 Capture
   ↓
20초
Capture Cycle Reset
   ↓
30초 연속 지속
   ↓
Web Alarm
```

정리:

| 조건                   | 동작                  |
| -------------------- | ------------------- |
| UNKNOWN 10초 지속       | Capture             |
| Capture Cycle 20초 경과 | Capture Cycle Reset |
| UNKNOWN 30초 연속 지속    | Web Alarm           |
| SPOOF 3초 연속 지속       | Capture + Web Alarm |

20초 Reset은 UNKNOWN 전체 존재 시간을 초기화하는 것이 아니라
**Capture Cycle Timer만 초기화합니다.**

즉 UNKNOWN이 계속 존재하는 경우:

```text
0초
 ↓
10초 : Capture
 ↓
20초 : Capture Cycle Reset
 ↓
30초 : Web Alarm
```

UNKNOWN이 사라지거나 OWNER/GUEST가 나타나면
UNKNOWN 관련 타이머가 초기화됩니다.

---

## 7. Web UI 접속

Security System이 실행된 장치의 IP를 확인합니다.

```bash
hostname -I
```

같은 네트워크의 스마트폰 또는 PC에서 다음 주소로 접속합니다.

```text
http://<JETSON_IP>:5000
```

현재 시연 환경에서 사용한 주소 예시:

```text
http://10.10.20.54:5000
```

Web UI에서는 다음 정보를 확인할 수 있습니다.

* OWNER / GUEST / UNKNOWN / SPOOF 상태
* UNKNOWN 지속 시간
* Capture Event
* Alarm Event
* 저장된 이벤트 이미지

`security_exe.py`가 실행되면 Web UI 서버도 함께 시작되므로
별도로 `web_alarm.py`를 실행하지 않습니다.

---

## 8. 저장 영상 시연 실행

저장 영상도 `security_exe.py`를 사용합니다.

시연에서는 다음 시간 기준을 사용합니다.

```text
UNKNOWN 5초  : Capture
UNKNOWN 15초 : Web Alarm
20초         : Capture Cycle Reset
SPOOF 3초    : Capture + Web Alarm
```

따라서 다음과 같이 실행합니다.

```bash
python security_exe.py \
  --source ./sim_video.mp4 \
  --loop \
  --unknown-seconds 5 \
  --unknown-alarm-seconds 15 \
  --unknown-reset-seconds 20 \
  --spoof-seconds 3
```

`--loop` 옵션을 사용하면 영상이 끝난 뒤 처음부터 다시 재생됩니다.

영상 한 번만 재생하려면 `--loop` 옵션을 제거합니다.

```bash
python security_exe.py \
  --source ./sim_video.mp4 \
  --unknown-seconds 5 \
  --unknown-alarm-seconds 15 \
  --unknown-reset-seconds 20 \
  --spoof-seconds 3
```

종료:

```text
q 또는 ESC
```

---

## 9. Webcam을 시연용 시간으로 실행

Webcam에서도 저장 영상 Simulation과 동일하게
**5초 Capture / 15초 Alarm / 20초 Capture Cycle Reset** 기준을 사용하려면 다음과 같이 실행합니다.

```bash
python security_exe.py \
  --camera 0 \
  --unknown-seconds 5 \
  --unknown-alarm-seconds 15 \
  --unknown-reset-seconds 20 \
  --spoof-seconds 3
```

시연 정책:

```text
UNKNOWN 감지
   ↓
5초 지속
   ↓
사진 Capture
   ↓
15초 연속 지속
   ↓
Web Alarm

Capture Cycle은 20초마다 Reset
```

SPOOF의 경우:

```text
SPOOF 감지
   ↓
3초 연속 지속
   ↓
Capture + Web Alarm
```

---

## 10. 주요 실행 옵션

`security_exe.py`에서 사용할 수 있는 주요 옵션은 다음과 같습니다.

```text
--camera                Webcam 번호
--source                저장 영상 또는 Camera Source
--loop                  저장 영상 반복 재생
--unknown-seconds       UNKNOWN Capture 발생 시간
--unknown-alarm-seconds UNKNOWN Web Alarm 발생 시간
--unknown-reset-seconds Capture Cycle Reset 시간
--spoof-seconds         SPOOF Capture + Alarm 발생 시간
--web-port              Web UI 포트 번호
--playback-speed        저장 영상 재생 속도
```

예시:

```bash
python security_exe.py --camera 0
```

```bash
python security_exe.py --source ./sim_video.mp4 --loop
```

---

## 11. 실행 시 확인사항

* 현재 Webcam과 저장 영상 실행 파일은 모두 `security_exe.py`입니다.
* 처음 사용하는 OWNER/GUEST는 먼저 얼굴 등록을 진행합니다.
* Webcam이 열리지 않으면 `--camera 0`, `--camera 1` 등 카메라 번호를 변경합니다.
* Web UI는 `security_exe.py` 실행 시 자동으로 시작됩니다.
* `web_alarm.py`를 별도로 실행할 필요가 없습니다.
* 스마트폰과 Jetson은 같은 네트워크에 연결되어 있어야 합니다.
* `security_exe.py`의 기본 정책은 **10초 Capture / 30초 Alarm / 20초 Capture Cycle Reset**입니다.
* 시연 정책인 **5초 Capture / 15초 Alarm / 20초 Capture Cycle Reset**을 사용하려면 실행 옵션으로 시간을 지정해야 합니다.
* SPOOF는 기본적으로 3초 연속 감지 시 Capture와 Web Alarm이 발생합니다.
* 종료는 `q` 또는 `ESC`를 사용합니다.

---

## 12. 주요 실행 명령 요약

### 기본 Webcam

```bash
python security_exe.py --camera 0
```

### 시연용 Webcam

```bash
python security_exe.py \
  --camera 0 \
  --unknown-seconds 5 \
  --unknown-alarm-seconds 15 \
  --unknown-reset-seconds 20 \
  --spoof-seconds 3
```

### 시연 영상 반복 재생

```bash
python security_exe.py \
  --source ./sim_video.mp4 \
  --loop \
  --unknown-seconds 5 \
  --unknown-alarm-seconds 15 \
  --unknown-reset-seconds 20 \
  --spoof-seconds 3
```

### OWNER 등록

```bash
python register_person.py \
  --name owner1 \
  --role OWNER \
  --camera 0
```

### GUEST 등록

```bash
python register_person.py \
  --name guest1 \
  --role GUEST \
  --camera 0
```

### 등록 사용자 확인

```bash
python manage_people.py list
```

### 환경 확인

```bash
python verify_setup.py
```
