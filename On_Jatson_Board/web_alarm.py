from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import logging
from pathlib import Path
import threading
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory


logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__, static_folder="static", static_url_path="/static")

_lock = threading.Lock()
_server_started = False
_event_root = Path("security_events").resolve()
_event_seq = 0
_events: deque[dict[str, Any]] = deque(maxlen=50)
_last_alarm_seq = 0
_live_status: dict[str, Any] = {
    "owner_count": 0,
    "guest_count": 0,
    "unknown_count": 0,
    "spoof_count": 0,
    "unknown_cycle_seconds": 0.0,
    "unknown_presence_seconds": 0.0,
    "updated_at": "",
}


HTML = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<meta name="theme-color" content="#f5f7fb">
<title>Cabin Guard</title>
<style>
:root {
  color-scheme: light;
  --page:#e9edf3; --app:#f7f8fb; --card:#ffffff; --ink:#111827; --sub:#7b8493;
  --line:#e8ebf0; --navy:#151c2b; --green:#25b66f; --red:#ef4b59; --amber:#f4a62a;
  --purple:#8b5cf6; --blue:#377dff; --shadow:0 10px 30px rgba(24,32,48,.08);
}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;min-height:100%;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans KR",Arial,sans-serif;background:var(--page);color:var(--ink)}
body{display:flex;justify-content:center}
.phone{width:100%;max-width:430px;min-height:100dvh;background:var(--app);position:relative;overflow:hidden}
.safe-top{height:max(10px,env(safe-area-inset-top))}
.header{padding:12px 18px 8px;display:flex;align-items:center;justify-content:space-between}
.title-wrap small{display:block;color:#98a1af;font-size:11px;font-weight:700;letter-spacing:.08em;margin-bottom:3px}
.title{font-size:22px;font-weight:850;letter-spacing:-.03em}
.conn{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:750;color:#667085;background:#fff;border:1px solid var(--line);padding:7px 9px;border-radius:999px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 4px rgba(37,182,111,.10)}
.content{padding:8px 14px 92px}
.hero{background:linear-gradient(145deg,#182132,#111827);color:#fff;border-radius:26px;padding:20px;box-shadow:0 16px 34px rgba(17,24,39,.18)}
.hero-top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}
.eyebrow{font-size:11px;color:#aab4c3;font-weight:750;letter-spacing:.08em}
.state{font-size:26px;font-weight:900;letter-spacing:-.04em;margin-top:4px}
.updated{font-size:11px;color:#aab4c3;text-align:right;line-height:1.4}
.live-pill{display:inline-flex;gap:6px;align-items:center;background:rgba(255,255,255,.08);padding:7px 10px;border-radius:999px;margin-top:13px;font-size:11px;color:#d9e0ea}
.live-pill i{width:6px;height:6px;border-radius:50%;background:#ff4d57;display:block}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:18px}
.stat{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.07);border-radius:16px;padding:11px 5px;text-align:center}
.stat b{display:block;font-size:20px}.stat span{display:block;font-size:9px;color:#aeb8c6;margin-top:3px;letter-spacing:.02em}
.timer-row{display:flex;gap:8px;margin-top:10px}.timer{flex:1;background:#fff;border:1px solid var(--line);border-radius:18px;padding:13px 14px;box-shadow:0 4px 18px rgba(21,28,43,.04)}
.timer span{display:block;color:var(--sub);font-size:10px;font-weight:700}.timer b{display:block;font-size:18px;margin-top:3px}
.section{display:flex;align-items:end;justify-content:space-between;margin:22px 3px 10px}.section h2{margin:0;font-size:18px;letter-spacing:-.02em}.section span{font-size:11px;color:var(--sub)}
.empty{background:#fff;border:1px solid var(--line);border-radius:22px;padding:34px 20px;text-align:center;color:#9aa2af;font-size:13px}.empty-icon{font-size:26px;margin-bottom:8px}
.event{background:var(--card);border:1px solid var(--line);border-radius:24px;margin-bottom:12px;overflow:hidden;box-shadow:0 6px 22px rgba(30,41,59,.045)}
.event.alarm{border-color:#ffd7da;box-shadow:0 8px 26px rgba(239,75,89,.08)}
.event-head{padding:15px 15px 12px}.event-top{display:flex;align-items:center;justify-content:space-between;gap:10px}.type-wrap{display:flex;align-items:center;gap:9px}.event-icon{width:36px;height:36px;border-radius:12px;display:grid;place-items:center;font-weight:900}.unknown-icon{background:#fff5df;color:#cf8300}.spoof-icon{background:#f2ebff;color:#7c3aed}.alarm-icon{background:#ffe8ea;color:#dd3644}
.event-type{font-size:13px;font-weight:850}.event-sub{font-size:10px;color:var(--sub);margin-top:2px}.time{font-size:10px;color:#9aa2af;text-align:right}.message{font-size:13px;line-height:1.45;margin-top:11px;color:#3d4654;font-weight:620}
.media-row{display:flex;gap:9px;padding:0 15px 14px;overflow-x:auto}.face{width:96px;height:96px;border-radius:18px;object-fit:cover;background:#eef1f5;border:1px solid #e5e8ed;flex:0 0 auto}.face-placeholder{width:96px;height:96px;border-radius:18px;background:#f0f2f6;display:grid;place-items:center;color:#a0a8b5;font-size:11px;flex:0 0 auto}
details{border-top:1px solid var(--line)}summary{list-style:none;padding:12px 15px;font-size:12px;font-weight:750;color:#526071;cursor:pointer;display:flex;justify-content:space-between}summary::-webkit-details-marker{display:none}.full{width:100%;display:block;background:#111827;max-height:360px;object-fit:contain}
.notice{position:fixed;z-index:20;top:max(14px,env(safe-area-inset-top));left:50%;transform:translateX(-50%) translateY(-130%);width:min(calc(100% - 28px),402px);background:#fff;border:1px solid #ffd7da;border-radius:18px;padding:13px 14px;box-shadow:0 16px 38px rgba(17,24,39,.18);transition:.28s ease}.notice.show{transform:translateX(-50%) translateY(0)}.notice strong{font-size:12px;color:var(--red)}.notice div{font-size:11px;color:#657083;margin-top:3px;line-height:1.4}
.bottom{position:fixed;bottom:0;left:50%;transform:translateX(-50%);width:100%;max-width:430px;background:rgba(255,255,255,.94);backdrop-filter:blur(16px);border-top:1px solid var(--line);padding:8px 24px max(9px,env(safe-area-inset-bottom));display:flex;justify-content:space-around}.tab{text-align:center;color:#a0a8b5;font-size:9px;font-weight:700}.tab b{display:grid;place-items:center;width:31px;height:27px;margin:0 auto 2px;border-radius:10px;font-size:16px}.tab.active{color:var(--blue)}.tab.active b{background:#edf3ff}
.note{font-size:10px;color:#a0a8b5;line-height:1.5;padding:8px 4px 0}
@media(min-width:560px){.phone{margin:24px 0;min-height:calc(100dvh - 48px);border-radius:32px;box-shadow:0 24px 70px rgba(22,30,44,.18)}.bottom{bottom:24px;border-radius:0 0 32px 32px}}

/* ===== Phone-frame UI override ===== */
:root{
  color-scheme: dark;
  --page:#020713;
  --app:#06101d;
  --card:#0b1728;
  --ink:#f4f8ff;
  --sub:#8fa7c8;
  --line:#1c3150;
  --navy:#0d1728;
  --green:#4ade80;
  --red:#ff647c;
  --amber:#ffb44f;
  --purple:#aa86ff;
  --blue:#60a5fa;
  --shadow:0 16px 36px rgba(0,0,0,.30);
}

html,body{
  width:100%;
  height:100%;
  min-height:100%;
  overflow:hidden;
  background:
    radial-gradient(circle at 50% -10%,rgba(56,122,238,.28),transparent 42%),
    radial-gradient(circle at 20% 85%,rgba(35,86,171,.12),transparent 35%),
    linear-gradient(180deg,#07162b 0%,#020713 100%);
}

body{
  display:flex;
  align-items:center;
  justify-content:center;
  padding:18px;
}

/* 실제 휴대폰 기기처럼 보이는 바깥 프레임 */
.phone{
  width:390px;
  max-width:100%;
  height:min(830px,94dvh);
  min-height:0;
  margin:0;
  display:flex;
  flex-direction:column;
  position:relative;
  overflow:hidden;

  background:
    radial-gradient(circle at 50% -10%,rgba(53,108,194,.15),transparent 35%),
    linear-gradient(180deg,#081424 0%,#040b16 100%);

  border:10px solid #05080d;
  border-radius:44px;
  box-shadow:
    0 0 0 1px rgba(255,255,255,.08),
    0 0 0 6px rgba(255,255,255,.018),
    0 30px 90px rgba(0,0,0,.62);
}

/* Dynamic Island */
.phone::before{
  content:"";
  position:absolute;
  z-index:80;
  top:10px;
  left:50%;
  transform:translateX(-50%);
  width:116px;
  height:29px;
  border-radius:18px;
  background:#010204;
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.025);
}

/* 홈 인디케이터 */
.phone::after{
  content:"";
  position:absolute;
  z-index:80;
  bottom:8px;
  left:50%;
  transform:translateX(-50%);
  width:112px;
  height:4px;
  border-radius:999px;
  background:rgba(255,255,255,.35);
}

.safe-top{
  height:39px;
  flex:0 0 39px;
}

/* 앱 상단 */
.header{
  flex:0 0 auto;
  padding:7px 16px 9px;
}
.title-wrap small{color:#6fa8ff}
.title{color:#f4f8ff;font-size:24px}
.conn{
  color:#b8cae4;
  background:rgba(255,255,255,.035);
  border-color:rgba(143,167,200,.16);
}
.dot{background:var(--green)}

.content{
  flex:1 1 auto;
  min-height:0;
  overflow-y:auto;
  padding:8px 14px 92px;
  scrollbar-width:none;
}
.content::-webkit-scrollbar{display:none}

/* 상태 카드 */
.hero{
  background:
    linear-gradient(145deg,rgba(16,33,57,.98),rgba(7,18,34,.98));
  border:1px solid rgba(96,165,250,.18);
  color:#fff;
  border-radius:25px;
  box-shadow:0 16px 34px rgba(0,0,0,.30);
}
.eyebrow,.updated{color:#8fa7c8}
.stat{
  background:rgba(0,0,0,.20);
  border-color:rgba(143,167,200,.13);
}
.stat span{color:#8fa7c8}

.timer{
  background:rgba(11,23,40,.92);
  border-color:rgba(143,167,200,.13);
  color:#f4f8ff;
  box-shadow:none;
}
.timer span{color:#8fa7c8}

.section h2{color:#f4f8ff}
.section span{color:#8fa7c8}

.empty{
  color:#8fa7c8;
  background:rgba(11,23,40,.75);
  border-color:rgba(143,167,200,.18);
}
.empty-icon{
  color:#4ade80;
}

/* 이벤트 카드 */
.event{
  background:
    linear-gradient(180deg,rgba(13,27,47,.98),rgba(8,18,32,.98));
  border-color:rgba(143,167,200,.14);
  box-shadow:0 12px 25px rgba(0,0,0,.20);
}
.event.alarm{border-color:rgba(255,100,124,.34)}
.event-sub,.time{color:#8fa7c8}
.message{color:#c7d7ec}
.face{
  background:#020713;
  border-color:rgba(255,255,255,.07);
}
.face-placeholder{
  background:#081425;
  color:#8fa7c8;
}
details{border-top-color:rgba(143,167,200,.12)}
summary{color:#9eb6d8}
.full{background:#000}

.unknown-icon{background:rgba(255,180,79,.10);color:#ffb44f}
.spoof-icon{background:rgba(170,134,255,.11);color:#aa86ff}
.alarm-icon{background:rgba(255,100,124,.11);color:#ff647c}

/* 알림 팝업도 휴대폰 내부에 고정 */
.notice{
  position:absolute;
  top:47px;
  left:14px;
  right:14px;
  width:auto;
  transform:translateY(-150%);
  background:rgba(21,12,21,.97);
  border-color:rgba(255,100,124,.28);
  box-shadow:0 16px 38px rgba(0,0,0,.38);
}
.notice.show{transform:translateY(0)}
.notice strong{color:#ff647c}
.notice div{color:#c7d3e4}

/* 하단 앱 탭바 */
.bottom{
  position:absolute;
  left:0;
  bottom:0;
  transform:none;
  width:100%;
  max-width:none;
  height:72px;
  padding:8px 24px 20px;
  background:rgba(4,10,19,.94);
  border-top-color:rgba(143,167,200,.10);
  backdrop-filter:blur(16px);
}
.tab{color:#627c9d}
.tab.active{color:#60a5fa}
.tab.active b{background:rgba(96,165,250,.11)}

.note{
  color:#7288a8;
  padding-bottom:8px;
}

/* 데스크톱에서는 '진짜 폰'으로, 실제 휴대폰에서는 화면 전체를 앱으로 */
@media(max-width:480px){
  html,body{
    overflow:auto;
    background:#040b16;
  }

  body{
    display:block;
    padding:0;
  }

  .phone{
    width:100%;
    max-width:none;
    height:100dvh;
    min-height:100dvh;
    border:0;
    border-radius:0;
    box-shadow:none;
  }

  .phone::before{
    top:8px;
  }

  .phone::after{
    bottom:6px;
  }

  .bottom{
    bottom:0;
  }
}

@media(max-height:700px) and (min-width:481px){
  .phone{height:96dvh}
  .title{font-size:20px}
  .state{font-size:23px}
  .hero{padding:15px}
  .stats{margin-top:12px}
}


.sound-control{
  margin-top:10px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  padding:10px 12px;
  border-radius:16px;
  background:rgba(96,165,250,.08);
  border:1px solid rgba(96,165,250,.16);
}
.sound-copy{
  min-width:0;
}
.sound-copy strong{
  display:block;
  font-size:10px;
  color:#dbeafe;
}
.sound-copy span{
  display:block;
  margin-top:2px;
  font-size:9px;
  color:#8299ba;
  line-height:1.35;
}
.sound-btn{
  appearance:none;
  border:1px solid rgba(96,165,250,.28);
  background:rgba(96,165,250,.14);
  color:#cfe2ff;
  font-size:9px;
  font-weight:850;
  border-radius:999px;
  padding:8px 10px;
  cursor:pointer;
  white-space:nowrap;
}
.sound-btn.enabled{
  color:#b8ffd2;
  background:rgba(74,222,128,.11);
  border-color:rgba(74,222,128,.24);
}

</style>
</head>
<body>
<div class="phone">
  <div class="safe-top"></div>
  <div id="notice" class="notice"><strong id="noticeTitle">보안 알림</strong><div id="noticeText"></div></div>
  <header class="header">
    <div class="title-wrap"><small>CABIN GUARD</small><div class="title">내 차량 보안</div></div>
    <div class="conn"><span class="dot" id="statusDot"></span><span id="connectionText">연결됨</span></div>
  </header>

  <div class="sound-control">
    <div class="sound-copy">
      <strong>15초 경고 알림음</strong>
      <span id="soundStatus">UNKNOWN이 15초 지속되면 재생됩니다. 먼저 한 번 활성화해 주세요.</span>
    </div>
    <button id="soundButton" class="sound-btn" type="button" onclick="enableSound()">알림음 켜기</button>
  </div>

  <main class="content">
    <section class="hero">
      <div class="hero-top">
        <div><div class="eyebrow">LIVE CABIN STATUS</div><div id="systemState" class="state">MONITORING</div></div>
        <div id="lastUpdate" class="updated">--</div>
      </div>
      <div class="live-pill"><i></i> 실시간 차량 내부 감시 중</div>
      <div class="stats">
        <div class="stat"><b id="ownerCount">0</b><span>OWNER</span></div>
        <div class="stat"><b id="guestCount">0</b><span>GUEST</span></div>
        <div class="stat"><b id="unknownCount">0</b><span>UNKNOWN</span></div>
        <div class="stat"><b id="spoofCount">0</b><span>SPOOF</span></div>
      </div>
    </section>

    <div class="timer-row">
      <div class="timer"><span>UNKNOWN 체류 · 15초 알림</span><b id="presenceTimer">0.0s</b></div>
      <div class="timer"><span>캡처 대기 · 5초</span><b id="cycleTimer">0.0s</b></div>
    </div>

    <div class="section"><h2>최근 보안 기록</h2><span id="eventCount">0건</span></div>
    <div id="eventList"><div id="empty" class="empty"><div class="empty-icon">✓</div>아직 저장된 보안 이벤트가 없습니다.</div></div>
    <div class="note">Jetson과 휴대폰이 같은 Wi‑Fi/LAN에 연결되어 있으면 상태가 자동으로 갱신됩니다. UNKNOWN 인원이 5초 연속 감지되면 사진을 캡처하고, 15초 연속 감지되면 휴대폰 경고 알림과 알림음이 발생합니다. SPOOF는 별도의 보안 이벤트로 처리됩니다.</div>
  </main>

  <nav class="bottom">
    <div class="tab active"><b>⌂</b>홈</div>
    <div class="tab"><b>◫</b>이벤트</div>
    <div class="tab"><b>●</b>연결</div>
  </nav>
</div>

<script>
let lastSeq = 0;
let totalEvents = 0;

const alarmAudio = new Audio("/static/alarm.wav");
alarmAudio.preload = "auto";
alarmAudio.volume = 1.0;

let soundEnabled = false;

async function enableSound(){
  const button = document.getElementById("soundButton");
  const status = document.getElementById("soundStatus");

  try{
    alarmAudio.currentTime = 0;
    await alarmAudio.play();

    soundEnabled = true;
    button.textContent = "알림음 ON";
    button.classList.add("enabled");
    status.textContent = "활성화됨 · UNKNOWN 15초 경고 발생 시 휴대폰에서 재생됩니다.";

    // 활성화 확인용으로 한 번 재생한 뒤 자동 종료되는 WAV입니다.
  }catch(err){
    soundEnabled = false;
    button.textContent = "다시 시도";
    button.classList.remove("enabled");
    status.textContent = "브라우저가 소리 재생을 차단했습니다. 버튼을 다시 눌러 주세요.";
    console.log("Enable alarm sound failed:", err);
  }
}

async function playAlarm(){
  if(!soundEnabled){
    console.log("Alarm event received, but sound is not enabled.");
    return;
  }

  try{
    alarmAudio.pause();
    alarmAudio.currentTime = 0;
    await alarmAudio.play();
  }catch(err){
    soundEnabled = false;
    const button = document.getElementById("soundButton");
    const status = document.getElementById("soundStatus");
    button.textContent = "알림음 켜기";
    button.classList.remove("enabled");
    status.textContent = "소리 재생이 차단되었습니다. 다시 활성화해 주세요.";
    console.log("Alarm playback failed:", err);
  }
}
function esc(v){return String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function niceType(t){if((t||'').includes('SPOOF'))return '위조 얼굴 감지';if((t||'').includes('ALARM'))return '15초 미등록자 경고';return '5초 미등록자 캡처';}
function iconInfo(ev){if(ev.alarm)return ['!','alarm-icon'];if((ev.event_type||'').includes('SPOOF'))return ['S','spoof-icon'];return ['?','unknown-icon'];}
function showNotice(title,text){const n=document.getElementById('notice');document.getElementById('noticeTitle').innerText=title;document.getElementById('noticeText').innerText=text;n.classList.add('show');setTimeout(()=>n.classList.remove('show'),4500);}
function cardHtml(ev){
  const [icon,ic]=iconInfo(ev);
  const faces=(ev.face_urls||[]).map(u=>`<img class="face" src="${u}" alt="face crop" loading="lazy">`).join('') || '<div class="face-placeholder">FACE</div>';
  const full=ev.full_frame_url?`<details><summary><span>전체 카메라 화면 보기</span><span>›</span></summary><img class="full" src="${ev.full_frame_url}" loading="lazy"></details>`:'';
  const meta=[]; if(ev.duration_seconds!=null)meta.push(`${Number(ev.duration_seconds).toFixed(1)}초 감지`); if(ev.face_count!=null)meta.push(`얼굴 ${ev.face_count}명`);
  return `<article class="event ${ev.alarm?'alarm':''}" data-seq="${ev.seq}"><div class="event-head"><div class="event-top"><div class="type-wrap"><div class="event-icon ${ic}">${icon}</div><div><div class="event-type">${niceType(ev.event_type)}</div><div class="event-sub">${esc(meta.join(' · ')||ev.event_type)}</div></div></div><div class="time">${esc(ev.time)}</div></div><div class="message">${esc(ev.message)}</div></div><div class="media-row">${faces}</div>${full}</article>`;
}
function addEvent(ev){
  const list=document.getElementById('eventList');
  document.getElementById('empty')?.remove();

  list.insertAdjacentHTML('afterbegin',cardHtml(ev));

  while(list.children.length>20){
    list.removeChild(list.lastChild);
  }

  totalEvents++;
  document.getElementById('eventCount').innerText=`${totalEvents}건`;

  if(ev.alarm){
    playAlarm();
    showNotice(
      niceType(ev.event_type),
      ev.message||'보안 이벤트가 감지되었습니다.'
    );
  }
}
async function pollEvents(){try{const r=await fetch(`/api/events?after=${lastSeq}`,{cache:'no-store'});const d=await r.json();document.getElementById('statusDot').style.background='var(--green)';document.getElementById('connectionText').innerText='연결됨';for(const ev of d.events){lastSeq=Math.max(lastSeq,ev.seq);addEvent(ev);}}catch(e){document.getElementById('statusDot').style.background='var(--red)';document.getElementById('connectionText').innerText='연결 끊김';}}
async function pollStatus(){try{const r=await fetch('/api/status',{cache:'no-store'});const s=await r.json();ownerCount.innerText=s.owner_count??0;guestCount.innerText=s.guest_count??0;unknownCount.innerText=s.unknown_count??0;spoofCount.innerText=s.spoof_count??0;presenceTimer.innerText=`${Number(s.unknown_presence_seconds||0).toFixed(1)}s`;cycleTimer.innerText=`${Number(s.unknown_cycle_seconds||0).toFixed(1)}s`;lastUpdate.innerText=(s.updated_at||'--')+' 업데이트';const state=(s.spoof_count||0)>0?'SPOOF':(s.unknown_count||0)>0?'UNKNOWN':'SAFE';systemState.innerText=state;systemState.style.color=state==='SAFE'?'#61e59b':state==='SPOOF'?'#c4a0ff':'#ffbd55';}catch(e){}}
setInterval(pollEvents,700);setInterval(pollStatus,700);pollEvents();pollStatus();
</script>
</body>
</html>
"""


def _safe_relative(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(_event_root).as_posix()
    except (ValueError, OSError):
        return None


def _read_event_metadata(event_dir: Path) -> dict[str, Any]:
    metadata_path = event_dir / "event.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _event_payload(
    *,
    seq: int,
    message: str,
    event_type: str,
    alarm: bool,
    event_dir: Path | str | None,
) -> dict[str, Any]:
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload: dict[str, Any] = {
        "seq": seq,
        "event_type": event_type,
        "message": message,
        "time": now_text,
        "alarm": bool(alarm),
        "face_urls": [],
        "full_frame_url": None,
        "duration_seconds": None,
        "face_count": None,
    }

    if event_dir is None:
        return payload

    event_path = Path(event_dir)
    meta = _read_event_metadata(event_path)

    captured_at = meta.get("captured_at")
    if captured_at:
        try:
            payload["time"] = datetime.fromisoformat(captured_at).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    rel_dir = _safe_relative(event_path)
    if rel_dir is None:
        return payload

    full_frame = meta.get("full_frame")
    if full_frame:
        payload["full_frame_url"] = f"/event-files/{rel_dir}/{full_frame}"

    crops = meta.get("face_crops") or []
    payload["face_urls"] = [f"/event-files/{rel_dir}/{name}" for name in crops]

    if "unknown_duration_seconds" in meta:
        payload["duration_seconds"] = meta.get("unknown_duration_seconds")
        payload["face_count"] = meta.get("unknown_count", len(crops))
    elif "spoof_duration_seconds" in meta:
        payload["duration_seconds"] = meta.get("spoof_duration_seconds")
        payload["face_count"] = meta.get("spoof_count", len(crops))

    return payload


def _append_event(
    *,
    message: str,
    event_type: str,
    alarm: bool,
    event_dir: Path | str | None = None,
) -> int:
    global _event_seq, _last_alarm_seq
    with _lock:
        _event_seq += 1
        seq = _event_seq
        payload = _event_payload(
            seq=seq,
            message=message,
            event_type=event_type,
            alarm=alarm,
            event_dir=event_dir,
        )
        _events.append(payload)
        if alarm:
            _last_alarm_seq = seq
    prefix = "WEB ALERT" if alarm else "WEB EVENT"
    print(f"[{prefix}] event #{seq}: {message}")
    return seq


@app.route("/")
def index() -> Response:
    return Response(HTML, mimetype="text/html")


@app.route("/api/events")
def api_events():
    try:
        after = int(request.args.get("after", "0"))
    except ValueError:
        after = 0
    with _lock:
        items = [dict(item) for item in _events if int(item["seq"]) > after]
    return jsonify({"events": items, "latest_seq": _event_seq})


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify(dict(_live_status))


# Backward-compatible endpoint used by older browser pages.
@app.route("/api/alarm")
def get_alarm():
    with _lock:
        last = next((e for e in reversed(_events) if e.get("alarm")), None)
        if last is None:
            return jsonify({"seq": 0, "message": "", "time": ""})
        return jsonify({"seq": last["seq"], "message": last["message"], "time": last["time"]})


@app.route("/event-files/<path:filename>")
def event_file(filename: str):
    # send_from_directory uses safe_join internally and blocks path traversal.
    return send_from_directory(str(_event_root), filename)


def publish_capture_event(
    event_dir: Path | str,
    message: str,
    event_type: str = "UNKNOWN_CAPTURE",
) -> int:
    """Publish a capture card to the phone UI."""
    return _append_event(
        message=message,
        event_type=event_type,
        alarm=False,
        event_dir=event_dir,
    )


def trigger_web_alarm(
    message: str = "Security alarm",
    event_dir: Path | str | None = None,
    event_type: str = "SECURITY_ALARM",
) -> int:
    """Publish a high-priority visual alert event to connected phone browsers."""
    return _append_event(
        message=message,
        event_type=event_type,
        alarm=True,
        event_dir=event_dir,
    )


def update_live_status(
    *,
    owner_count: int,
    guest_count: int,
    unknown_count: int,
    spoof_count: int,
    unknown_cycle_seconds: float = 0.0,
    unknown_presence_seconds: float = 0.0,
) -> None:
    with _lock:
        _live_status.update(
            {
                "owner_count": int(owner_count),
                "guest_count": int(guest_count),
                "unknown_count": int(unknown_count),
                "spoof_count": int(spoof_count),
                "unknown_cycle_seconds": round(float(unknown_cycle_seconds), 2),
                "unknown_presence_seconds": round(float(unknown_presence_seconds), 2),
                "updated_at": datetime.now().strftime("%H:%M:%S"),
            }
        )


def start_web_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    event_root: Path | str | None = None,
) -> None:
    global _server_started, _event_root

    with _lock:
        if _server_started:
            return
        if event_root is not None:
            _event_root = Path(event_root).resolve()
        _event_root.mkdir(parents=True, exist_ok=True)
        _server_started = True

    def run() -> None:
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    thread = threading.Thread(target=run, name="web-security-dashboard", daemon=True)
    thread.start()
    print(f"[WEB] Mobile security dashboard started on port {port}")
    print(f"[WEB] Phone: http://<JETSON_IP>:{port}")