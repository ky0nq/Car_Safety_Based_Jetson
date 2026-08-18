#!/usr/bin/env python3
"""
차량 방치 감지 알림 서버 (Guardian Alert Server)
=================================================

역할
----
1. Jetson에서 돌아가는 감지 스크립트(detector_main.py)가 프레임마다 보내주는 상태(JSON)를 받아서 서버 메모리에 저장한다.
2. 웹 대시보드(dashboard.html)가 WebSocket으로 붙어 있으면 상태가 바뀔 때마다 실시간으로 밀어준다(push). 즉, 이 서버가 감지 스크립트와 웹 UI 사이의 다리 역할.
3. 경고 단계가 2단계(보호자 알림)에 도달하면 텔레그램으로 보호자 휴대폰에 실제 메시지를 전송한다.

실행
----
    pip install fastapi "uvicorn[standard]" requests
    export TELEGRAM_BOT_TOKEN="123456:ABC-your-bot-token"
    export TELEGRAM_CHAT_ID="123456789"
    python3 alert_server.py
    # -> http://<서버IP>:8000  에서 대시보드 확인 가능
    # -> ws://<서버IP>:8000/ws 로 실시간 상태 수신

감지 스크립트(detector_main.py)는 이 서버의 POST /api/status 로
매 프레임(또는 N프레임마다) 현재 상태를 보내주기만 하면 된다.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

DASHBOARD_HTML_PATH = Path(__file__).parent / "dashboard.html"

app = FastAPI(title="Car Guardian Alert Server")


# --------------------------------------------------------------------------
# 상태 모델
# --------------------------------------------------------------------------

class OccupantIn(BaseModel):
    type: str  # "child" | "animal" | "adult"
    age: Optional[float] = None


class StatusIn(BaseModel):
    engine_on: bool
    stage: int  # 0=안전, 1=저소음 알림, 2=보호자 알림, 3=공조(시연: 화면표시)
    elapsed_seconds: float
    person_count: int
    child_count: int
    animal_count: int
    occupants: list[OccupantIn] = []


class EngineState:
    """서버가 들고 있는 '현재 상태' 하나(단일 차량 데모용)."""

    def __init__(self) -> None:
        self.engine_on: bool = True
        self.stage: int = 0
        self.elapsed_seconds: float = 0.0
        self.person_count: int = 0
        self.child_count: int = 0
        self.animal_count: int = 0
        self.occupants: list[dict] = []
        self.updated_at: float = time.time()

        self._last_alerted_stage: int = 0  
        self.history: list[dict] = []  

    def to_dict(self) -> dict:
        return {
            "engine_on": self.engine_on,
            "stage": self.stage,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "person_count": self.person_count,
            "child_count": self.child_count,
            "animal_count": self.animal_count,
            "occupants": self.occupants,
            "updated_at": self.updated_at,
            "history": self.history[-20:],
        }


state = EngineState()
state_lock = asyncio.Lock()

connected_clients: set[WebSocket] = set()


# --------------------------------------------------------------------------
# telegram 
# --------------------------------------------------------------------------

def send_telegram_message(text: str) -> bool:
    """텔레그램으로 보호자 휴대폰에 실제 메시지 전송 동작 구현.

    BotFather에서 만든 봇의 TELEGRAM_BOT_TOKEN과, 보호자가 그 봇과 대화를 한 번이라도 나눈 뒤 얻은 TELEGRAM_CHAT_ID가 필요하다.
    두 값이 설정되어 있지 않으면 콘솔에만 출력하고 False를 반환 (하드웨어/네트워크가 없는 자리에서도 코드가 죽지 않도록).
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[TELEGRAM 미설정] 아래 메시지를 보내려 했음:\n  {text}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
            },
            timeout=5,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as error:
        print(f"[TELEGRAM 전송 실패] {error}")
        return False


def build_alert_text(status: StatusIn) -> str:
    occupant_lines = []
    for occupant in status.occupants:
        if occupant.type == "child":
            age_text = f" (추정 {occupant.age:.1f}세)" if occupant.age is not None else ""
            occupant_lines.append(f"- 아동{age_text}")
        elif occupant.type == "animal":
            occupant_lines.append("- 반려동물")
        else:
            occupant_lines.append("- 탑승자")

    occupant_text = "\n".join(occupant_lines) if occupant_lines else "- (탑승자 정보 없음)"

    return (
        "🚨 <b>차량 방치 경고</b>\n"
        f"시동이 꺼진 뒤 {status.elapsed_seconds:.0f}초 동안 아래 탑승자가 "
        "차량 뒷자리에 남아있습니다.\n\n"
        f"{occupant_text}\n\n"
        "지금 바로 차량을 확인해주세요."
    )


# --------------------------------------------------------------------------
# WebSocket broadcast
# --------------------------------------------------------------------------

async def broadcast_state() -> None:
    if not connected_clients:
        return

    payload = json.dumps({"type": "status", "data": state.to_dict()}, ensure_ascii=False)
    dead_clients = []

    for client in connected_clients:
        try:
            await client.send_text(payload)
        except Exception:
            dead_clients.append(client)

    for client in dead_clients:
        connected_clients.discard(client)


# --------------------------------------------------------------------------
# REST API — 감지 스크립트(detector_main.py)가 호출하는 엔드포인트
# --------------------------------------------------------------------------

@app.post("/api/status")
async def post_status(status: StatusIn):
    async with state_lock:
        state.engine_on = status.engine_on
        state.stage = status.stage
        state.elapsed_seconds = status.elapsed_seconds
        state.person_count = status.person_count
        state.child_count = status.child_count
        state.animal_count = status.animal_count
        state.occupants = [occupant.model_dump() for occupant in status.occupants]
        state.updated_at = time.time()

        # 시동이 켜지거나(엔진 ON), 아무도 안 남아있으면(stage 0) 사건 종료 -> 재무장
        if status.engine_on or status.stage == 0:
            state._last_alerted_stage = 0

        # 2단계(보호자 알림)에 "새로" 도달한 순간에만 텔레그램 발송 (스팸 방지)
        if status.stage >= 2 and state._last_alerted_stage < 2:
            alert_text = build_alert_text(status)
            sent = send_telegram_message(alert_text)
            state.history.append(
                {
                    "time": time.strftime("%H:%M:%S"),
                    "stage": status.stage,
                    "message": alert_text,
                    "telegram_sent": sent,
                }
            )
            state._last_alerted_stage = status.stage

    await broadcast_state()
    return {"ok": True}


@app.get("/api/status")
async def get_status():
    return state.to_dict()


@app.post("/api/reset")
async def reset_state(clear_history: bool = False):
    async with state_lock:
        state.stage = 0
        state.elapsed_seconds = 0.0
        state._last_alerted_stage = 0

        if clear_history:
            state.history.clear()
    await broadcast_state()
    return {"ok": True}


# --------------------------------------------------------------------------
# WebSocket — 대시보드(dashboard.html)가 붙는 엔드포인트
# --------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)

    await websocket.send_text(
        json.dumps({"type": "status", "data": state.to_dict()}, ensure_ascii=False)
    )

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        
# --------------------------------------------------------------------------
# 기본 static page 
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    if DASHBOARD_HTML_PATH.exists():
        return DASHBOARD_HTML_PATH.read_text(encoding="utf-8")
    return "<h1>dashboard.html이 없습니다. alert_server.py와 같은 폴더에 두세요.</h1>"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)