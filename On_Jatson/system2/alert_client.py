"""
alert_client.py
================
감지 스크립트(detector_main.py)가 매 프레임 계산해내는 상태를
알림 서버(alert_server.py)로 보내는 아주 얇은 클라이언트.

카메라 처리 루프를 절대 막으면 안 되기 때문에(네트워크가 느리면 FPS가 뚝뚝
떨어짐), 실제 HTTP 전송은 별도 스레드에서 처리하고, 큐에는 "가장 최신 상태
1개만" 유지한다. 즉 전송이 밀리면 오래된 상태는 버리고 최신 것만 보낸다.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import requests


class AlertClient:
    def __init__(self, server_url: str, timeout: float = 1.5) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

        self._queue: "queue.Queue[dict]" = queue.Queue(maxsize=1)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "AlertClient":
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def send(self, status: dict) -> None:
        """최신 상태로 큐를 덮어쓴다(오래된 미전송 상태는 버림)."""
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(status)
        except queue.Full:
            pass

    def _worker(self) -> None:
        while self._running:
            try:
                status = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                requests.post(
                    f"{self.server_url}/api/status",
                    json=status,
                    timeout=self.timeout,
                )
            except requests.RequestException as error:
                print(f"[alert_client] 서버 전송 실패: {error}")
