# System2 실행 방법

## 1. 시스템 개요

System2는 차량 내부의 사람과 동물을 검출하고,
사람의 나이를 추정하여 차량 내부 잔류 상황을 판단하는 시스템입니다.

잔류 위험 대상으로 판단하는 대상은 다음과 같습니다.

* **7세 이하 아동(CHILD)**
* **동물(ANIMAL: dog, cat)**

사람이 검출되면 MiVOLO V2로 나이를 추정하고,
추정 나이가 **7세 이하인 경우 CHILD**로 분류합니다.

dog 또는 cat은 나이 추정 없이 바로 잔류 감지 대상으로 처리하며,
화면에서는 `ANIMAL`로 통합하여 표시합니다.

동작 흐름:

```text id="gwlqhp"
YOLO11
  ↓
사람 / dog / cat 검출
  ↓
사람 Tracking
  ↓
MiVOLO V2
  ↓
사람 나이 추정
  ↓
7세 이하 CHILD 판정
  ↓
차량 시동 상태 + 잔류 대상 확인
  ↓
Stage 판단
  ↓
Alert Server
  ↓
Web UI / Telegram 앱 알림
```

기본 Age Threshold는 7세이며,
사람의 추정 나이가 7세 이하이면 `CHILD`로 판정합니다.

잔류 감지 대상:

```text id="5pfi0x"
CHILD  : 7세 이하로 추정된 사람
ANIMAL : YOLO에서 검출된 dog 또는 cat
```

최대 5명의 사람을 Tracking 및 나이 추정 대상으로 처리합니다.

현재 Webcam과 저장 영상은 모두 다음 실행 파일을 사용합니다.

```text id="ynoz0y"
yolo_mivolov2_exe.py
```

---

## 2. 데모 시간과 실제 시스템 운용 시간

현재 제출 코드의 기본값은 발표 및 시연 시간을 줄이기 위한 **데모용 시간**입니다.

| 단계      | 데모 기본값 | 실제 시스템 운용 제안값 | 동작              |
| ------- | -----: | ------------: | --------------- |
| Stage 1 |    10초 |    5분 | 1단계 저소음 경고      |
| Stage 2 |    15초 |   10분 | 보호자 Telegram 알림 |
| Stage 3 |    20초 |   20분 | 차량 공조 (에어컨) 동작 단계      |

즉, 시연에서는 다음과 같이 동작합니다.

```text id="1pkvbi"
10초 → 15초 → 20초
```

실제 차량 시스템에 적용할 경우 코드에서 제안하는 기준은 다음과 같습니다.

```text id="ha1zmo"
5분 → 10분 → 20분
```

현재 Stage 3의 에어컨 가동 단계는 **실제 차량 에어컨을 직접 제어하는 것이 아니라 시연 화면에서 차량 공조 (에어컨) 동작 단계로 표시하는 구조**입니다.

---

## 3. 가상환경 생성 및 패키지 설치

System2 폴더에서 실행합니다.

```bash id="svfflb"
cd system2

python -m venv --system-site-packages venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jetson에서는 CUDA가 적용된 PyTorch와 시스템 환경을 사용하기 위해
`--system-site-packages` 옵션을 사용합니다.

CUDA 사용 여부 확인:

```bash id="0r2j85"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

GPU를 정상적으로 사용할 수 있는 경우 다음과 같이 출력됩니다.

```text id="7xhp9t"
True
```

---

## 4. 필요한 모델 파일

기본 시연에서는 다음 모델 파일을 사용합니다.

```text id="iwtca4"
yolo11n.engine
mivolo_v2.pt
```

역할:

```text id="z7dw4i"
yolo11n.engine
→ 사람 / dog / cat 객체 검출

mivolo_v2.pt
→ 검출된 사람의 나이 추정
```

MiVOLO V2는 TorchScript 모델을 사용하며,
CUDA 사용이 가능한 경우 GPU에서 추론합니다.

---

## 5. Web UI / Telegram 서버

System2의 Web UI와 Telegram 알림 기능은
`alert_server.py`가 실행되는 서버를 통해 동작합니다.

구성:

```text id="jg68jw"
yolo_mivolov2_exe.py
        ↓
   Alert Client
        ↓
 alert_server.py
     ↙       ↘
 Web UI    Telegram
```

### 현재 프로젝트 시연 환경

현재 시연 환경에서 사용한 Alert Server 주소:

```text id="5erm19"
http://10.10.20.58:8000
```

Web UI 및 Telegram 알림 기능을 사용하려면
해당 Alert Server가 실행되어 있어야 합니다.

Alert Server를 사용하지 않는 경우에도
YOLO + MiVOLO의 로컬 영상 추론 자체는 실행할 수 있습니다.

이 경우 다음 옵션을 사용합니다.

```text id="8l98v8"
--no-alert-server
```

---

## 6. Alert Server 실행

Alert Server를 실행할 장치에서 다음과 같이 설정합니다.

```bash id="5hajec"
cd system2
source venv/bin/activate
```

Telegram Bot 정보를 환경변수로 설정합니다.

```bash id="9ntofh"
export TELEGRAM_BOT_TOKEN="BOT_TOKEN"
export TELEGRAM_CHAT_ID="CHAT_ID"
```

Alert Server 실행:

```bash id="vkrtyh"
python alert_server.py
```

기본 Port:

```text id="la2sm5"
8000
```

서버 상태 확인:

```bash id="r7l04w"
curl http://127.0.0.1:8000/api/status
```

브라우저에서 Web UI 접속:

```text id="x2cobi"
http://<SERVER_IP>:8000
```

현재 시연 환경 예시:

```text id="brz84v"
http://10.10.20.58:8000
```

Telegram Bot Token 또는 Chat ID가 설정되어 있지 않으면
Telegram 메시지는 실제로 전송되지 않습니다.

---

## 7. Webcam 실행

현재 Webcam 실행 파일은 `yolo_mivolov2_exe.py`입니다.

Alert Server가 실행 중인 상태에서 새 터미널을 엽니다.

```bash id="dnqrxv"
cd system2
source venv/bin/activate
```

### Alert Server와 감지 프로그램이 같은 장치인 경우

```bash id="g1knxs"
python yolo_mivolov2_exe.py \
  --mivolo ./mivolo_v2.pt \
  --detector ./yolo11n.engine \
  --camera 0 \
  --fullscreen \
  --alert-server-url http://127.0.0.1:8000 \
  --no-display-crops
```

### Alert Server가 별도 장치에서 실행되는 경우

현재 시연 서버를 사용할 경우:

```bash id="f9nyv4"
python yolo_mivolov2_exe.py \
  --mivolo ./mivolo_v2.pt \
  --detector ./yolo11n.engine \
  --camera 0 \
  --fullscreen \
  --alert-server-url http://10.10.20.58:8000 \
  --no-display-crops
```

`--source`를 지정하지 않으면 자동으로 Webcam 모드로 실행됩니다.

기본 Webcam 번호:

```text id="azhm01"
0
```

---

## 8. 차량 시동 및 경고 단계 시연

프로그램은 기본적으로 **차량 시동이 ON인 상태**로 시작합니다.

실행 키:

```text id="gtevco"
e   : 차량 시동 ON / OFF 전환
r   : Person Tracking 및 저장된 Age 정보 초기화
q   : 종료
ESC : 종료
```

시연을 시작하려면 `e` 키를 눌러 차량 시동을 OFF 상태로 변경합니다.

```text id="lvnvlh"
프로그램 시작
    ↓
ENGINE ON
    ↓
'e' 입력
    ↓
ENGINE OFF
```

시동이 OFF인 상태에서 다음 대상 중 하나라도 계속 감지되면
잔류 타이머가 시작됩니다.

```text id="yi7zfx"
CHILD
또는
ANIMAL
```

즉:

```text id="of5rvj"
ENGINE OFF
    +
7세 이하 CHILD 또는 dog/cat 감지
    ↓
잔류 타이머 시작
```

데모 기본값:

```text id="ysjwkb"
0 ~ 10초 : Monitoring

10초
→ Stage 1

15초
→ Stage 2
→ 보호자 Telegram 알림

20초
→ Stage 3
→ 차량 공조 (에어컨) 동작 단계
```

시동을 다시 ON으로 변경하거나 잔류 대상이 모두 사라지면
잔류 타이머와 Stage가 초기화됩니다.

```text id="kiwm6k"
ENGINE ON
또는
CHILD / ANIMAL 미검출
    ↓
Stage 0
    ↓
잔류 타이머 Reset
```

---

## 9. 실제 시스템 시간으로 실행

코드에서 제안하는 실제 운용 기준인
**5분 / 10분 / 20분**으로 실행하려면 다음과 같이 실행합니다.

```bash id="y8stwg"
python yolo_mivolov2_exe.py \
  --mivolo ./mivolo_v2.pt \
  --detector ./yolo11n.engine \
  --camera 0 \
  --alert-server-url http://10.10.20.58:8000 \
  --stage1-seconds 300 \
  --stage2-seconds 600 \
  --stage3-seconds 1200 \
  --no-display-crops
```

동작:

```text id="nk6zaa"
5분
→ Stage 1

10분
→ Stage 2
→ 보호자 Telegram 알림

20분
→ Stage 3
→ 차량 공조 (에어컨) 동작 단계
```

---

## 10. 저장 영상 실행

저장 영상도 Webcam과 동일하게
`yolo_mivolov2_exe.py`를 사용합니다.

`--source` 옵션에 영상 경로를 지정합니다.

### 기본 데모 시간으로 실행

```bash id="1yqelc"
python yolo_mivolov2_exe.py \
  --mivolo ./mivolo_v2.pt \
  --detector ./yolo11n.engine \
  --source ./sim_video.mp4 \
  --fullscreen \
  --alert-server-url http://10.10.20.58:8000 \
  --no-display-crops
```

저장 영상은 **기본적으로 반복 재생**됩니다.

즉 별도의 `--loop` 옵션을 추가하지 않아도 영상이 끝나면 다시 재생됩니다.

```text id="el1z2o"
sim_video.mp4 재생
      ↓
영상 종료
      ↓
처음부터 다시 재생
```

---

## 11. 저장 영상 한 번만 재생

영상을 반복하지 않고 한 번만 실행하려면
`--play-once` 옵션을 사용합니다.

```bash id="6ji6uh"
python yolo_mivolov2_exe.py \
  --mivolo ./mivolo_v2.pt \
  --detector ./yolo11n.engine \
  --source ./sim_video.mp4 \
  --play-once \
  --fullscreen \
  --alert-server-url http://10.10.20.58:8000 \
  --no-display-crops
```

`--play-once`를 사용하지 않으면 저장 영상은 기본적으로 반복됩니다.

---

## 12. 서버 없이 모델만 실행

Web UI와 Telegram 알림을 사용하지 않고
로컬 추론 화면만 확인하려면 `--no-alert-server` 옵션을 사용합니다.

### Webcam

```bash id="82347p"
python yolo_mivolov2_exe.py \
  --mivolo ./mivolo_v2.pt \
  --detector ./yolo11n.engine \
  --camera 0 \
  --no-alert-server \
  --no-display-crops
```

### 저장 영상

```bash id="u1a6ne"
python yolo_mivolov2_exe.py \
  --mivolo ./mivolo_v2.pt \
  --detector ./yolo11n.engine \
  --source ./sim_video.mp4 \
  --no-alert-server \
  --no-display-crops
```

이 경우 다음 기능은 동작하지 않습니다.

```text id="9glyf6"
Web UI 연동
Telegram 앱 알림
Alert Server 상태 관리
```

다만 다음 기능은 로컬에서 그대로 동작합니다.

```text id="20qhhp"
YOLO 객체 검출
MiVOLO 나이 추정
CHILD / ADULT 판정
ANIMAL 판정
잔류 시간 계산
Stage 0 / 1 / 2 / 3 판정
화면 출력
```

---

## 13. 사람 나이 추정 방식

YOLO에서 사람이 검출되면 각 사람에게 Tracking ID를 부여합니다.

MiVOLO는 모든 사람을 매 Frame마다 추론하지 않고,
보이는 사람 중 한 명씩 순차적으로 나이를 추정합니다.

기본 설정:

```text id="eijtyy"
Age Interval : 1.0초
Age Window   : 5
Max People   : 5명
Age Threshold: 7세
```

나이 추정값은 각 사람별로 저장되며
최근 나이 결과의 Median을 사용하여 결과를 안정화합니다.

```text id="eoh852"
YOLO Person Detection
        ↓
Tracking ID 할당
        ↓
MiVOLO Age Prediction
        ↓
최근 Age 값 저장
        ↓
Median Smoothing
        ↓
7세 이하 → CHILD
7세 초과 → ADULT
```

---

## 14. YOLO 동작 설정

기본 YOLO 설정:

```text id="4bda9r"
Input Size       : 640
Confidence       : 0.35
YOLO Every       : 2 Frame
Person Labels    : person, adult, child, baby
Animal Labels    : dog, cat
```

YOLO는 기본적으로 2 Frame마다 한 번 실행하며,
중간 Frame에서는 최근 Detection 결과를 재사용합니다.

dog와 cat은 내부적으로 각각 검출되지만
최종 화면에서는 다음과 같이 표시됩니다.

```text id="smlpd6"
ANIMAL
```

---

## 15. 주요 실행 옵션

```text id="b7s2rp"
--mivolo
    MiVOLO TorchScript 모델 경로

--detector
    YOLO 모델 경로

--source
    저장 영상 경로
    지정하지 않으면 Webcam 사용

--camera
    Webcam Camera Index
    기본값: 0

--fullscreen
    전체 화면으로 표시

--play-once
    저장 영상을 한 번만 재생

--detector-imgsz
    YOLO 입력 이미지 크기
    기본값: 640

--detector-conf
    YOLO Confidence Threshold
    기본값: 0.35

--yolo-every
    YOLO를 실행할 Frame 간격
    기본값: 2

--age-interval
    MiVOLO 나이 추론 간격
    기본값: 1.0초

--age-window
    나이 Median Smoothing Window
    기본값: 5

--age-threshold
    CHILD 판정 나이
    기본값: 7세

--max-people
    최대 Tracking 사람 수
    기본값: 5

--alert-server-url
    Alert Server 주소
    기본값: http://127.0.0.1:8000

--no-alert-server
    Alert Server 없이 로컬 추론만 실행

--stage1-seconds
    Stage 1 진입 시간
    기본값: 10초

--stage2-seconds
    Stage 2 진입 시간
    기본값: 15초

--stage3-seconds
    Stage 3 진입 시간
    기본값: 20초

--no-display-crops
    별도 Crop 화면을 표시하지 않음
```

---

## 16. 실행 시 확인사항

* 현재 Webcam과 저장 영상 실행 파일은 모두 `yolo_mivolov2_exe.py`입니다.
* Webcam은 `--camera`를 사용합니다.
* 저장 영상은 `--source`를 사용합니다.
* 저장 영상은 기본적으로 반복 재생됩니다.
* 영상을 한 번만 재생하려면 `--play-once`를 사용합니다.
* Web UI와 Telegram 알림을 사용하려면 `alert_server.py`가 실행 중이어야 합니다.
* 현재 시연 구성의 Alert Server 주소는 `10.10.20.58:8000`입니다.
* 서버 IP가 변경되면 `--alert-server-url`도 변경해야 합니다.
* 프로그램은 기본적으로 차량 시동 ON 상태로 시작합니다.
* 잔류 감지 시연 시 `e` 키를 눌러 차량 시동을 OFF로 변경합니다.
* 7세 이하로 추정된 사람은 `CHILD`로 판정합니다.
* 7세 초과로 추정된 사람은 `ADULT`로 판정합니다.
* dog와 cat은 잔류 감지 대상이며 화면에서는 `ANIMAL`로 표시됩니다.
* 최대 5명의 사람을 Tracking 및 나이 추정 대상으로 처리합니다.
* Stage 2 진입 시 Telegram 보호자 알림이 발생합니다.
* Stage 3은 현재 시연에서 차량 공조 (에어컨) 동작 단계까지 표시하며 실제 차량 에어컨을 직접 제어하지는 않습니다.
* `r` 키를 누르면 Person Tracking ID와 저장된 나이 정보가 초기화됩니다.
* 종료는 `q` 또는 `ESC`를 사용합니다.

---

## 17. 주요 실행 명령 요약

### Alert Server 실행

```bash id="p6ps4o"
export TELEGRAM_BOT_TOKEN="BOT_TOKEN"
export TELEGRAM_CHAT_ID="CHAT_ID"

python alert_server.py
```

### Webcam + Alert Server

```bash id="tz0gjd"
python yolo_mivolov2_exe.py \
  --mivolo ./mivolo_v2.pt \
  --detector ./yolo11n.engine \
  --camera 0 \
  --fullscreen \
  --alert-server-url http://10.10.20.58:8000 \
  --no-display-crops
```

### 저장 영상 반복 재생

```bash id="7brqbs"
python yolo_mivolov2_exe.py \
  --mivolo ./mivolo_v2.pt \
  --detector ./yolo11n.engine \
  --source ./sim_video.mp4 \
  --fullscreen \
  --alert-server-url http://10.10.20.58:8000 \
  --no-display-crops
```

### 저장 영상 한 번만 재생

```bash id="c9ts2l"
python yolo_mivolov2_exe.py \
  --mivolo ./mivolo_v2.pt \
  --detector ./yolo11n.engine \
  --source ./sim_video.mp4 \
  --play-once \
  --fullscreen \
  --alert-server-url http://10.10.20.58:8000 \
  --no-display-crops
```

### 서버 없이 Webcam 실행

```bash id="umkdr0"
python yolo_mivolov2_exe.py \
  --mivolo ./mivolo_v2.pt \
  --detector ./yolo11n.engine \
  --camera 0 \
  --no-alert-server \
  --no-display-crops
```

### 실제 운용 제안 시간
--stage1-seconds, --stage2-seconds, --stage3-seconds 에 대해서는 초 (seconds) 단위로 작성합니다.

```bash id="1hfgr1"
python yolo_mivolov2_exe.py \
  --mivolo ./mivolo_v2.pt \
  --detector ./yolo11n.engine \
  --camera 0 \
  --alert-server-url http://10.10.20.58:8000 \
  --stage1-seconds 300 \
  --stage2-seconds 600 \
  --stage3-seconds 1200 \
  --no-display-crops
```
