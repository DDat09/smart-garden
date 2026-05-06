from datetime import datetime
from flask import Flask, request, jsonify, Response
from collections import deque, Counter
import os
import time
import numpy as np
import cv2

import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO

# ===== AUTO ACTION RULES (AI -> ACTUATORS) =====
AUTO_ENABLE = True
ALARM_LABELS = {"sau_benh"}
WATER_LABELS = {"heo_xanh", "la_vang"}
AUTO_MIN_CONF = 0.30
AUTO_COOLDOWN_SEC = 10
_last_auto_ts = {"bom": 0.0, "coi": 0.0}

print("=== SERVER ALL-IN-ONE: FIREBASE + WEB PRO + ESP32 API + YOLOv8 ===")

# ======================================================
# 1) FIREBASE CONFIG
# ======================================================
FIREBASE_KEY_PATH = os.environ.get("FIREBASE_KEY_PATH", r"D:/espcam/yolo/firebase_key.json")
FIREBASE_DB_URL = os.environ.get(
    "FIREBASE_DB_URL",
    "https://tem-iot-94142-default-rtdb.asia-southeast1.firebasedatabase.app/"
)

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
print("[OK] Firebase Admin connected")

def now_str():
    return datetime.now().strftime("%H:%M:%S %d-%m-%Y")

def push_history(sensor, value):
    db.reference(f"vuon/history/{sensor}").push({
        "time": now_str(),
        "value": value
    })

def _set_cmd(path, state: str):
    try:
        db.reference(path).set("ON" if str(state).upper() == "ON" else "OFF")
    except Exception as e:
        print("[AUTO] set cmd error:", path, e)

def auto_actuate_from_ai(label: str, conf: float):
    if not AUTO_ENABLE:
        return
    now = time.time()
    if conf < AUTO_MIN_CONF:
        return
    if label in ALARM_LABELS:
        if now - _last_auto_ts["coi"] >= AUTO_COOLDOWN_SEC:
            _set_cmd("vuon/lenh/coi", "ON")
            _last_auto_ts["coi"] = now
            print(f"[AUTO] ALARM ON (label={label}, conf={conf:.2f})")
    if label in WATER_LABELS:
        if now - _last_auto_ts["bom"] >= AUTO_COOLDOWN_SEC:
            _set_cmd("vuon/lenh/bom", "ON")
            _last_auto_ts["bom"] = now
            print(f"[AUTO] PUMP ON (label={label}, conf={conf:.2f})")

# ======================================================
# 2) YOLO MODEL
# ======================================================
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", r"D:\espcam\yolo\bestt.pt")
yolo = YOLO(MODEL_PATH)
print("🔥 YOLO Model Loaded:", MODEL_PATH)
print("📌 Classes:", yolo.names)

# ======================================================
# 3) HTML DASHBOARD - FULL VERSION
# ======================================================
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Smart Garden Pro - Full</title>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">

  <script src="https://www.gstatic.com/firebasejs/8.10.0/firebase-app.js"></script>
  <script src="https://www.gstatic.com/firebasejs/8.10.0/firebase-database.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

  <style>
    :root {
      --green-400: #66bb6a;
      --green-500: #4caf50;
      --green-600: #388e3c;
      --teal-600: #00796b;
      --red-400: #ef5350;
      --blue-400: #42a5f5;
      --amber-400: #ffa726;
      --bg-main: #0d1117;
      --bg-card: #161b22;
      --bg-card2: #1c2333;
      --border: rgba(255,255,255,0.08);
      --text-primary: #e6edf3;
      --text-secondary: #8b949e;
      --text-muted: #484f58;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'DM Sans', sans-serif;
      background: var(--bg-main);
      color: var(--text-primary);
      min-height: 100vh;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1.2rem 2rem;
      background: var(--bg-card);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .logo {
      display: flex;
      align-items: center;
      gap: 10px;
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1.2rem;
      font-weight: 600;
      color: var(--green-400);
    }

    .logo-icon {
      width: 36px; height: 36px;
      background: linear-gradient(135deg, var(--green-600), var(--teal-600));
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 18px;
    }

    .header-meta {
      display: flex;
      align-items: center;
      gap: 1rem;
      font-size: 13px;
      color: var(--text-secondary);
    }

    .status-dot {
      width: 8px; height: 8px;
      background: var(--green-400);
      border-radius: 50%;
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%,100% { opacity: 1; }
      50% { opacity: 0.4; }
    }

    .main {
      padding: 1.5rem 2rem;
      max-width: 1600px;
      margin: 0 auto;
    }

    .section-title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 12px;
      font-weight: 500;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 1rem;
    }

    .sensor-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 1.5rem;
    }

    .sensor-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.2rem 1.4rem;
      position: relative;
      overflow: hidden;
    }

    .sensor-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
      background: linear-gradient(90deg, var(--red-400), var(--amber-400));
    }

    .sensor-label {
      font-size: 12px;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: .5rem;
    }

    .sensor-value {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 2.4rem;
      font-weight: 600;
      line-height: 1;
    }

    .sensor-unit {
      font-size: 14px;
      font-weight: 400;
      color: var(--text-secondary);
      margin-left: 4px;
    }

    .sensor-trend {
      margin-top: 8px;
      font-size: 11px;
      color: var(--text-muted);
    }

    .three-col {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr;
      gap: 16px;
      margin-bottom: 1.5rem;
    }

    @media (max-width: 1200px) {
      .three-col { grid-template-columns: 1fr 1fr; }
    }

    .two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 1.5rem;
    }

    @media (max-width: 900px) {
      .two-col { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
    }

    .card-header {
      padding: 1rem 1.25rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .card-title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 14px;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .card-title i { color: var(--green-400); }

    .card-body { padding: 1.25rem; }

    .camera-wrapper {
      position: relative;
      background: #000;
      border-radius: 12px;
      overflow: hidden;
      aspect-ratio: 16/9;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 12px;
    }

    #cameraStream {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    .camera-badge {
      position: absolute;
      bottom: 10px; left: 10px;
      background: rgba(0,0,0,0.7);
      border: 1px solid rgba(76,175,80,0.4);
      border-radius: 20px;
      padding: 4px 12px;
      font-size: 11px;
      color: var(--green-400);
      display: flex;
      align-items: center;
      gap: 5px;
    }

    .cam-live-dot {
      width: 6px; height: 6px;
      background: var(--red-400);
      border-radius: 50%;
      animation: pulse 1s infinite;
    }

    .ai-result-box {
      background: var(--bg-card2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1rem;
    }

    .ai-result-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }

    .ai-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-muted);
    }

    .ai-disease-name {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 1.4rem;
      font-weight: 600;
    }

    .ai-disease-name.healthy { color: var(--green-400); }
    .ai-disease-name.sick { color: var(--red-400); }

    .confidence-bar-wrap {
      background: rgba(255,255,255,0.06);
      border-radius: 4px;
      height: 6px;
      overflow: hidden;
      margin-top: 4px;
    }

    .confidence-bar {
      height: 100%;
      background: linear-gradient(90deg, var(--green-500), #26a69a);
      border-radius: 4px;
      transition: width .5s ease;
    }

    .device-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .device-toggle {
      background: var(--bg-card2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1rem;
      cursor: pointer;
      transition: all .2s;
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 8px;
    }

    .device-toggle:hover {
      border-color: rgba(76,175,80,0.3);
    }

    .device-toggle.active {
      border-color: var(--green-500);
      background: rgba(76,175,80,0.08);
    }

    .device-icon {
      width: 36px; height: 36px;
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 16px;
      background: rgba(255,255,255,0.06);
    }

    .device-toggle.active .device-icon {
      background: rgba(76,175,80,0.2);
      color: var(--green-400);
    }

    .device-name {
      font-size: 13px;
      font-weight: 500;
    }

    .device-status {
      font-size: 11px;
      color: var(--text-muted);
    }

    .device-toggle.active .device-status {
      color: var(--green-400);
    }

    .schedule-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }

    .time-input-wrap {
      display: flex;
      flex-direction: column;
      gap: 4px;
      flex: 1;
      min-width: 100px;
    }

    .time-input-label {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: .5px;
    }

    input[type="time"] {
      background: var(--bg-card2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 9px 12px;
      color: var(--text-primary);
      font-family: 'DM Sans', sans-serif;
      font-size: 14px;
      width: 100%;
      color-scheme: dark;
      outline: none;
      transition: border-color .2s;
    }

    input[type="time"]:focus {
      border-color: var(--green-500);
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 9px 16px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--bg-card2);
      color: var(--text-primary);
      font-size: 13px;
      font-family: 'DM Sans', sans-serif;
      cursor: pointer;
      transition: all .2s;
    }

    .btn:hover {
      border-color: var(--green-600);
      background: rgba(76,175,80,0.08);
    }

    .btn-primary {
      background: var(--green-600);
      border-color: var(--green-600);
      color: #fff;
    }

    .btn-primary:hover {
      background: var(--green-500);
    }

    .btn-full { width: 100%; justify-content: center; }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 500;
    }

    .badge-green {
      background: rgba(76,175,80,0.15);
      color: var(--green-400);
      border: 1px solid rgba(76,175,80,0.3);
    }

    .badge-red {
      background: rgba(239,83,80,0.15);
      color: var(--red-400);
      border: 1px solid rgba(239,83,80,0.3);
    }

    .chart-container {
      position: relative;
      height: 200px;
    }

    .log-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .log-item {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      padding: 8px 10px;
      background: var(--bg-card2);
      border-radius: 8px;
      font-size: 12px;
    }

    .log-time {
      color: var(--text-muted);
      min-width: 55px;
      flex-shrink: 0;
    }

    .log-msg { color: var(--text-secondary); line-height: 1.4; }
  </style>
</head>

<body>

  <header>
    <div class="logo">
      <div class="logo-icon">🌿</div>
      Smart Garden Pro - Full Version
    </div>
    <div class="header-meta">
      <div class="status-dot"></div>
      <span>Firebase Connected</span>
      <span id="header-time" style="color:var(--text-muted)"></span>
    </div>
  </header>

  <div class="main">

    <!-- SENSORS -->
    <p class="section-title">Cảm biến môi trường</p>
    <div class="sensor-grid">
      <div class="sensor-card">
        <div class="sensor-label"><i class="fa-solid fa-temperature-half"></i> Nhiệt độ</div>
        <div class="sensor-value" id="temp">--<span class="sensor-unit">°C</span></div>
        <div class="sensor-trend" id="temp-trend">Đang tải...</div>
      </div>
      <div class="sensor-card">
        <div class="sensor-label"><i class="fa-solid fa-droplet"></i> Độ ẩm không khí</div>
        <div class="sensor-value" id="hum">--<span class="sensor-unit">%</span></div>
        <div class="sensor-trend">RH</div>
      </div>
      <div class="sensor-card">
        <div class="sensor-label"><i class="fa-solid fa-seedling"></i> Độ ẩm đất</div>
        <div class="sensor-value" id="soil">--<span class="sensor-unit">%</span></div>
        <div class="sensor-trend" id="soil-status">Đang tải...</div>
      </div>
    </div>

    <!-- CAMERA + CONTROL + SCHEDULE -->
    <p class="section-title">Camera AI & Điều khiển</p>
    <div class="three-col">

      <!-- CAMERA -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <i class="fa-solid fa-camera"></i>
            Camera ESP32-CAM (LIVE)
          </div>
          <span id="cam-status" class="badge badge-red"><i class="fa-solid fa-circle" style="font-size:7px"></i> Offline</span>
        </div>
        <div class="card-body">
          <div class="camera-wrapper">
            <img id="cameraStream" />
            <div class="camera-badge">
              <div class="cam-live-dot" id="liveDot"></div>
              <span id="camBadgeText">LIVE</span>
            </div>
          </div>

          <div class="ai-result-box">
            <div class="ai-result-row">
              <span class="ai-label">Kết quả AI</span>
              <span class="ai-label" id="ai-time-label">--</span>
            </div>
            <div class="ai-disease-name" id="ai-benh">Đang phân tích...</div>
            <div style="margin-top:8px">
              <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted); margin-bottom:3px">
                <span>Độ tin cậy</span>
                <span id="ai-conf-pct">0%</span>
              </div>
              <div class="confidence-bar-wrap">
                <div class="confidence-bar" id="confBar" style="width:0%"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- DEVICE -->
      <div class="card">
        <div class="card-header">
          <div class="card-title"><i class="fa-solid fa-sliders"></i> Điều khiển</div>
        </div>
        <div class="card-body">
          <div class="device-grid">
            <div class="device-toggle" id="toggle-bom" onclick="toggle('bom','toggle-bom')">
              <div class="device-icon"><i class="fa-solid fa-faucet"></i></div>
              <div>
                <div class="device-name">Bơm nước</div>
                <div class="device-status" id="status-bom">Tắt</div>
              </div>
            </div>
            <div class="device-toggle" id="toggle-suoi" onclick="toggle('suoi','toggle-suoi')">
              <div class="device-icon"><i class="fa-solid fa-fire"></i></div>
              <div>
                <div class="device-name">Sưởi ấm</div>
                <div class="device-status" id="status-suoi">Tắt</div>
              </div>
            </div>
            <div class="device-toggle" id="toggle-coi" onclick="toggle('coi','toggle-coi')">
              <div class="device-icon"><i class="fa-solid fa-bell"></i></div>
              <div>
                <div class="device-name">Còi báo</div>
                <div class="device-status" id="status-coi">Tắt</div>
              </div>
            </div>
            <div class="device-toggle" id="toggle-den" onclick="toggle('den','toggle-den')">
              <div class="device-icon"><i class="fa-solid fa-lightbulb"></i></div>
              <div>
                <div class="device-name">Đèn chiếu</div>
                <div class="device-status" id="status-den">Tắt</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- SCHEDULE -->
      <div class="card">
        <div class="card-header">
          <div class="card-title"><i class="fa-solid fa-calendar-check"></i> Lịch tưới</div>
        </div>
        <div class="card-body">
          <div class="schedule-row">
            <div class="time-input-wrap">
              <span class="time-input-label">Bắt đầu</span>
              <input type="time" id="start" />
            </div>
            <div class="time-input-wrap">
              <span class="time-input-label">Kết thúc</span>
              <input type="time" id="end" />
            </div>
          </div>
          <button class="btn btn-primary btn-full" onclick="saveSchedule()">
            <i class="fa-solid fa-floppy-disk"></i> Lưu
          </button>
          <div id="schedule-display" style="margin-top:10px; font-size:12px; color:var(--text-muted); text-align:center"></div>
        </div>
      </div>

    </div>

    <!-- CHARTS -->
    <p class="section-title">Biểu đồ dữ liệu</p>
    <div class="two-col">
      <div class="card">
        <div class="card-header">
          <div class="card-title"><i class="fa-solid fa-chart-line"></i> Realtime (20 điểm)</div>
        </div>
        <div class="card-body">
          <div class="chart-container">
            <canvas id="chartRealtime"></canvas>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <div class="card-title"><i class="fa-solid fa-chart-area"></i> Lịch sử 24 giờ</div>
        </div>
        <div class="card-body">
          <div class="chart-container">
            <canvas id="chartHistory"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- LOG -->
    <p class="section-title">Nhật ký sự kiện</p>
    <div class="card">
      <div class="card-header">
        <div class="card-title"><i class="fa-solid fa-list-ul"></i> Activity Log</div>
      </div>
      <div class="card-body">
        <div class="log-list" id="logList">
          <div class="log-item">
            <span class="log-time">--:--</span>
            <span class="log-msg">Dashboard khởi động...</span>
          </div>
        </div>
      </div>
    </div>

  </div>

  <script>
    // FIREBASE
    const firebaseConfig = {
      databaseURL: "https://tem-iot-94142-default-rtdb.asia-southeast1.firebasedatabase.app/"
    };
    firebase.initializeApp(firebaseConfig);
    const db = firebase.database();

    // CLOCK
    function updateClock() {
      document.getElementById('header-time').textContent =
        new Date().toLocaleTimeString('vi-VN');
    }
    updateClock();
    setInterval(updateClock, 1000);

    // LOG
    function addLog(msg) {
      const list = document.getElementById('logList');
      const now = new Date().toLocaleTimeString('vi-VN', {hour:'2-digit',minute:'2-digit'});
      const div = document.createElement('div');
      div.className = 'log-item';
      div.innerHTML = `<span class="log-time">${now}</span><span class="log-msg">${msg}</span>`;
      list.insertBefore(div, list.firstChild);
      if (list.children.length > 8) list.removeChild(list.lastChild);
    }

    // SENSORS
    db.ref("vuon/thongso/nhietdo").on("value", s => {
      const v = s.val();
      document.getElementById('temp').innerHTML = (v ?? '--') + '<span class="sensor-unit">°C</span>';
      if (v !== null) {
        const t = document.getElementById('temp-trend');
        t.textContent = v > 35 ? '⚠ Cao' : v < 18 ? '❄ Thấp' : '✓ OK';
      }
      addRealtime(v, 0);
    });

    db.ref("vuon/thongso/doam").on("value", s => {
      const v = s.val();
      document.getElementById('hum').innerHTML = (v ?? '--') + '<span class="sensor-unit">%</span>';
      addRealtime(v, 1);
    });

    db.ref("vuon/thongso/dat").on("value", s => {
      const v = s.val();
      document.getElementById('soil').innerHTML = (v ?? '--') + '<span class="sensor-unit">%</span>';
      if (v !== null) {
        const st = document.getElementById('soil-status');
        st.textContent = v < 30 ? '⚠ Khô' : v > 80 ? '💧 Ẩm' : '✓ OK';
      }
      addRealtime(v, 2);
    });

    // DEVICE
    function toggle(dev, btnId) {
      const ref = db.ref("vuon/lenh/" + dev);
      ref.once("value").then(s => {
        const newState = s.val() === "ON" ? "OFF" : "ON";
        ref.set(newState);
        addLog(`${dev.toUpperCase()}: ${newState}`);
      });
    }

    function bindDevice(dev, toggleId, statusId) {
      db.ref("vuon/lenh/" + dev).on("value", s => {
        const on = s.val() === "ON";
        const el = document.getElementById(toggleId);
        const st = document.getElementById(statusId);
        if (el) el.className = on ? "device-toggle active" : "device-toggle";
        if (st) st.textContent = on ? "🟢 Bật" : "⚫ Tắt";
      });
    }

    bindDevice('bom',  'toggle-bom',  'status-bom');
    bindDevice('suoi', 'toggle-suoi', 'status-suoi');
    bindDevice('coi',  'toggle-coi',  'status-coi');
    bindDevice('den',  'toggle-den',  'status-den');

    // SCHEDULE
    function saveSchedule() {
      const s = document.getElementById('start').value;
      const e = document.getElementById('end').value;
      if (!s || !e) { alert('Nhập đủ giờ'); return; }
      db.ref("vuon/lenh/lich").set({ start: s, end: e });
      document.getElementById('schedule-display').textContent = `Lịch: ${s} → ${e}`;
      addLog(`Lịch tưới: ${s} → ${e}`);
    }

    db.ref("vuon/lenh/lich").on("value", s => {
      const v = s.val();
      if (v && v.start && v.end) {
        document.getElementById('start').value = v.start;
        document.getElementById('end').value = v.end;
        document.getElementById('schedule-display').textContent = `Lịch: ${v.start} → ${v.end}`;
      }
    });

    // AI RESULT
    db.ref("vuon/ai").on("value", snap => {
      const data = snap.val() ?? {};
      const benh = data.benh ?? "--";
      const time  = data.time  ?? "--";
      const conf  = data.confidence;

      const nameEl = document.getElementById('ai-benh');
      nameEl.textContent = benh;
      const sick = benh === "la_vang" || benh === "sau_benh";
      nameEl.className = "ai-disease-name " + (sick ? "sick" : (benh === "--" ? "" : "healthy"));

      if (conf !== null && conf !== undefined) {
        const c = Number(conf);
        if (Number.isFinite(c)) {
          const pct = c <= 1 ? c * 100 : c;
          document.getElementById('ai-conf-pct').textContent = pct.toFixed(1) + '%';
          document.getElementById('confBar').style.width = pct.toFixed(1) + '%';
        }
      }

      document.getElementById('ai-time-label').textContent = time;
    });

    // CAMERA LIVE
    let lastUpdate = 0;
    function displayLatestImage() {
      const now = Date.now();
      if (now - lastUpdate < 300) return;
      lastUpdate = now;

      fetch('/get_latest_image')
        .then(r => r.blob())
        .then(blob => {
          const img = document.getElementById('cameraStream');
          img.src = URL.createObjectURL(blob);
          document.getElementById('cam-status').innerHTML = '<i class="fa-solid fa-circle" style="font-size:7px"></i> Live';
          document.getElementById('cam-status').className = 'badge badge-green';
          document.getElementById('liveDot').style.background = '#ef5350';
        })
        .catch(err => {});
    }

    setInterval(displayLatestImage, 300);

    // CHARTS
    const chartDefaults = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#8b949e', boxWidth: 12, font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: '#484f58', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#484f58', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } }
      }
    };

    const realtimeChart = new Chart(document.getElementById('chartRealtime'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Temp °C', data: [], borderColor: '#ef5350', backgroundColor: 'rgba(239,83,80,0.1)', fill: true, tension: 0.4, pointRadius: 2 },
          { label: 'Hum %',   data: [], borderColor: '#42a5f5', backgroundColor: 'rgba(66,165,245,0.08)', fill: true, tension: 0.4, pointRadius: 2 },
          { label: 'Soil %',  data: [], borderColor: '#66bb6a', backgroundColor: 'rgba(102,187,106,0.08)', fill: true, tension: 0.4, pointRadius: 2 }
        ]
      },
      options: chartDefaults
    });

    function addRealtime(value, dataset) {
      if (value === null || value === undefined) return;
      const now = new Date().toLocaleTimeString('vi-VN', {hour:'2-digit',minute:'2-digit',second:'2-digit'});
      if (realtimeChart.data.labels.length > 20) {
        realtimeChart.data.labels.shift();
        realtimeChart.data.datasets.forEach(d => d.data.shift());
      }
      realtimeChart.data.labels.push(now);
      realtimeChart.data.datasets[dataset].data.push(value);
      realtimeChart.update('none');
    }

    const historyChart = new Chart(document.getElementById('chartHistory'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Temp °C',    data: [], borderColor: '#ef5350', tension: 0.4, pointRadius: 1 },
          { label: 'Humidity %', data: [], borderColor: '#42a5f5', tension: 0.4, pointRadius: 1 },
          { label: 'Soil %',     data: [], borderColor: '#66bb6a', tension: 0.4, pointRadius: 1 }
        ]
      },
      options: chartDefaults
    });

    function loadHistory(sensor, idx) {
      db.ref("vuon/history/" + sensor).limitToLast(50).on("value", snap => {
        const data = snap.val() ?? {};
        const labels = [], values = [];
        Object.keys(data).forEach(k => {
          labels.push(data[k].time);
          values.push(data[k].value);
        });
        historyChart.data.labels = labels;
        historyChart.data.datasets[idx].data = values;
        historyChart.update('none');
      });
    }

    loadHistory('nhietdo', 0);
    loadHistory('doam', 1);
    loadHistory('dat', 2);

    addLog('Dashboard khởi động ✓');
  </script>
</body>
</html>
"""

app = Flask(__name__)

latest_image = None

@app.before_request
def log_req():
    print(f"[REQ] {request.remote_addr} {request.method} {request.path}")

@app.get("/")
def index():
    return Response(DASHBOARD_HTML, mimetype="text/html")

@app.get("/health")
def health():
    return jsonify({"ok": True, "time": now_str()})

@app.get("/get_latest_image")
def get_latest_image():
    global latest_image
    if latest_image is None:
        return Response(b"", status=204)
    return Response(latest_image, mimetype="image/jpeg")

@app.post("/api/telemetry")
def telemetry():
    d = request.get_json(force=True, silent=True) or {}
    temp = float(d.get("temp", 0))
    hum  = float(d.get("hum", 0))
    soil = int(d.get("soil", 0))

    db.reference("vuon/thongso/nhietdo").set(temp)
    db.reference("vuon/thongso/doam").set(hum)
    db.reference("vuon/thongso/dat").set(soil)

    push_history("nhietdo", temp)
    push_history("doam", hum)
    push_history("dat", soil)

    return jsonify({"ok": True})

@app.get("/api/pull_cmd")
def pull_cmd():
    def g(x): return "ON" if str(x).upper() == "ON" else "OFF"

    lich = db.reference("vuon/lenh/lich").get() or {}
    start = lich.get("start", "")
    end = lich.get("end", "")

    return jsonify({
        "bom": g(db.reference("vuon/lenh/bom").get()),
        "suoi": g(db.reference("vuon/lenh/suoi").get()),
        "coi": g(db.reference("vuon/lenh/coi").get()),
        "den": g(db.reference("vuon/lenh/den").get()),
        "lich": {"start": start, "end": end},
    })

@app.post("/predict")
def predict():
    global latest_image
    
    print("🔔 [RECEIVED] POST /predict")  # ← THÊM DÒNG NÀY
    
    img_bytes = request.data
    print(f"📦 Image size: {len(img_bytes)} bytes")  # ← THÊM DÒNG NÀY
    
    if not img_bytes or len(img_bytes) < 100:
        print("❌ Image too small")  # ← THÊM DÒNG NÀY
        return jsonify({"error": "empty"}), 400

    latest_image = img_bytes

    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("❌ Decode failed")  # ← THÊM DÒNG NÀY
        return jsonify({"error": "decode"}), 400

    print("✓ Image decoded OK")  # ← THÊM DÒNG NÀY

    t0 = time.perf_counter()
    result = yolo(img)[0]
    infer_ms = (time.perf_counter() - t0) * 1000.0

    label = "khong_xac_dinh"
    conf = 0.0

    if hasattr(result, "probs") and result.probs is not None:
        class_id = int(result.probs.top1)
        label = str(yolo.names.get(class_id, class_id))
        conf = float(result.probs.top1conf)
    elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        confs = boxes.conf.detach().cpu().numpy()
        idx = int(np.argmax(confs))
        best = boxes[idx]
        conf = float(best.conf.detach().cpu().numpy().item())
        cls_id = int(best.cls.detach().cpu().numpy().item())
        label = str(yolo.names.get(cls_id, cls_id))

    db.reference("vuon/ai/benh").set(label)
    db.reference("vuon/ai/confidence").set(float(conf))
    db.reference("vuon/ai/time").set(now_str())
    auto_actuate_from_ai(label, float(conf))

    print(f"✅ [PREDICT] {label} | {conf:.2f}% | {infer_ms:.1f}ms")  # ← THÊM DÒNG NÀY

    return jsonify({"ok": True})

@app.post("/api/ai")
def api_ai():
    d = request.get_json(force=True, silent=True) or {}
    benh = str(d.get("label") or d.get("benh") or "").strip()
    conf = d.get("confidence", None)

    if not benh:
        return jsonify({"ok": False, "error": "missing_label"}), 400

    db.reference("vuon/ai/benh").set(benh)
    db.reference("vuon/ai/time").set(now_str())

    if conf is not None:
        try:
            db.reference("vuon/ai/confidence").set(float(conf))
        except:
            pass

    return jsonify({"ok": True})

@app.get("/api/ai_state")
def ai_state():
    data = db.reference("vuon/ai").get() or {}
    return jsonify({
        "benh": data.get("benh", "--"),
        "confidence": data.get("confidence", None),
        "time": data.get("time", "--"),
    })

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌿 SMART GARDEN PRO - FULL VERSION WITH LIVE CAMERA")
    print("="*70)
    print(f"WEB Dashboard : http://127.0.0.1:5000")
    print(f"LAN           : http://192.168.1.24:5000")
    print(f"\nFeatures:")
    print(f"  ✓ Camera LIVE Streaming")
    print(f"  ✓ AI Disease Detection (Real-time)")
    print(f"  ✓ Sensor Data (Temp, Humidity, Soil)")
    print(f"  ✓ Device Control (Pump, Heater, Alarm, Light)")
    print(f"  ✓ Auto Watering Schedule")
    print(f"  ✓ Charts (Realtime + History)")
    print(f"  ✓ Firebase Sync")
    print(f"  ✓ Activity Log")
    print("="*70 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)