#!/usr/bin/env python3
# server_asr_ctrl_test_lazy.py
# 懒加载 ASR + 中文口令稳健识别 + 唤醒词门控 + 时长/能量过滤（无MQTT）

import os, json, wave, datetime, math, re, threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

# ===================== 基础配置 =====================
SAVE_DIR   = Path("./rx_audio")
RATE       = 16000
CHANNELS   = 1
SAMPWIDTH  = 2  # PCM16

SAVE_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI()

# ===================== ASR 档位/环境变量 =====================
# 运行档位（可覆盖下方详细变量）: ASR_PROFILE=tiny|base|small|medium
ASR_PROFILE   = os.environ.get("ASR_PROFILE", "").lower().strip()

# 逐项指定（优先级更高于 PROFILE）
ASR_MODEL_NAME = os.environ.get("ASR_MODEL", None)      # e.g. "small"
ASR_DEVICE     = os.environ.get("ASR_DEVICE", "auto")   # "auto"/"cpu"/"cuda"
ASR_COMPUTE    = os.environ.get("ASR_COMPUTE", None)    # "int8"/"int8_float16"/"float16"/"float32"

# 假ASR模式（链路自测用，1=不跑模型）
USE_FAKE_ASR   = int(os.environ.get("USE_FAKE_ASR", "0"))

# 语音段保护：超过该秒数强制提交（防误长按）
MAX_UTT_SEC    = float(os.environ.get("MAX_UTT_SEC", "10"))

# 过滤阈值：短段/低能量直接丢弃；唤醒词门控
MIN_UTT_SEC   = float(os.environ.get("MIN_UTT_SEC", "0.6"))   # 最短有效语音时长（秒）
MIN_RMS       = float(os.environ.get("MIN_RMS", "800"))       # 最低能量阈值（幅值）
REQUIRE_WAKE  = int(os.environ.get("REQUIRE_WAKE", "1"))      # 是否需要唤醒词(1/0)
WAKE_WORDS    = [w for w in os.environ.get("WAKE_WORDS", "机器人").split(",") if w.strip()]

# 行走/转向参数（解析到动作的缺省值/上限）
DEFAULT_VX        = 0.25
MAX_VX            = 0.6
DEFAULT_TURN_RATE = math.radians(60)
MAX_TURN_RATE     = math.radians(120)
MAX_DURATION      = 10.0

# ===================== ASR 懒加载 =====================
asr_model = None
model_loaded = False
asr_lock = threading.Lock()

def _defaults_from_profile(profile: str):
    profile = profile.lower().strip()
    if profile == "small":
        return ("small", "int8")
    if profile == "base":
        return ("base", "int8")
    if profile == "medium":
        return ("medium", "int8_float16")
    return ("tiny", "int8")

def ensure_asr_loaded():
    global asr_model, model_loaded, ASR_MODEL_NAME, ASR_COMPUTE
    if USE_FAKE_ASR or model_loaded:
        return
    with asr_lock:
        if model_loaded or USE_FAKE_ASR:
            return
        if ASR_MODEL_NAME is None or ASR_COMPUTE is None:
            m, c = _defaults_from_profile(ASR_PROFILE)
            ASR_MODEL_NAME = ASR_MODEL_NAME or m
            ASR_COMPUTE    = ASR_COMPUTE or c
        print(f"[ASR] Loading WhisperModel(name={ASR_MODEL_NAME}, device={ASR_DEVICE}, compute={ASR_COMPUTE}) ...")
        from faster_whisper import WhisperModel
        asr_model = WhisperModel(ASR_MODEL_NAME, device=ASR_DEVICE, compute_type=ASR_COMPUTE)
        model_loaded = True
        print("[ASR] Model loaded.")

# ===================== 工具函数 =====================
@app.get("/", response_class=PlainTextResponse)
def index():
    return "OK: /ws; send PCM16 16k mono in ~20ms frames, then send {\"event\":\"end_of_utterance\"}\n"

@app.get("/status", response_class=JSONResponse)
def status():
    return {
        "model_loaded": bool(model_loaded),
        "fake_asr": bool(USE_FAKE_ASR),
        "profile": ASR_PROFILE or "(none)",
        "model": ASR_MODEL_NAME or "(auto)",
        "device": ASR_DEVICE,
        "compute": ASR_COMPUTE or "(auto)",
        "max_utt_sec": MAX_UTT_SEC,
        "min_utt_sec": MIN_UTT_SEC,
        "min_rms": MIN_RMS,
        "require_wake": bool(REQUIRE_WAKE),
        "wake_words": WAKE_WORDS,
    }

def save_wav_and_metrics(buf: bytes) -> dict:
    if not buf:
        return {"path": None, "duration_s": 0.0, "bytes": 0, "rms": 0.0}
    day_dir = SAVE_DIR / datetime.date.today().strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%H%M%S_%f")
    wav_path = day_dir / f"utt_{ts}.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(RATE)
        wf.writeframes(buf)
    a = np.frombuffer(buf[:len(buf) - (len(buf) % 2)], dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
    dur = len(buf) / (RATE * CHANNELS * SAMPWIDTH)
    return {"path": str(wav_path), "duration_s": round(dur, 3), "bytes": len(buf), "rms": round(rms, 2)}

# 纠错/归一化（典型错字替换 + 简繁统一）
def normalize_zh_cmd(text: str) -> str:
    t = text.strip()
    repl = {
        # 同音/错字
        "前進": "前进", "後退": "后退",
        "移民": "一米", "一名": "一米", "已米": "一米", "亿米": "一米", "壹米": "一米", "一迷": "一米",
        # 简繁
        "準備": "准备", "姿態": "姿态", "轉": "转",
    }
    for k, v in repl.items():
        t = t.replace(k, v)
    return t

def run_asr_pcm16(buf: bytes) -> str:
    if USE_FAKE_ASR:
        return "前进一米（FAKE）"
    ensure_asr_loaded()
    if not buf:
        return ""
    audio = np.frombuffer(buf[:len(buf) - (len(buf) % 2)], dtype=np.int16).astype(np.float32) / 32768.0

    # 稳健中文配置：beam + 固定语言 + 无跨句 + 热词先验 + 低温度
    initial_prompt = (
        "指令：前进、后退、左转、右转、停止、站立、准备姿态、左移、右移、"
        "米、半米、零点五米、0.5米、角度、度、九十度、四十五度。"
    )
    segments, _ = asr_model.transcribe(
        audio,
        language="zh",
        beam_size=5,                       # 可改 8 提升准确率（更慢）
        best_of=5,
        vad_filter=False,                  # 已由 PTT 截断
        condition_on_previous_text=False,  # 不使用跨句上下文
        initial_prompt=initial_prompt,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
    )
    text = "".join(seg.text for seg in segments).strip()
    return text

# ===================== 指令解析（中文） =====================
NUM_MAP = {"零":0,"〇":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}

def cn_num_to_float(s: str) -> float:
    s = s.replace("。","点")
    if s == "十": return 10.0
    if "十" in s and len(s) <= 3:
        a, b = s.split("十")
        if a=="" and b!="": return 10 + NUM_MAP.get(b, 0)
        if a!="" and b=="": return NUM_MAP.get(a, 0) * 10
        if a!="" and b!="": return NUM_MAP.get(a, 0) * 10 + NUM_MAP.get(b, 0)
    val = 0.0; frac = 0.0; base = 0.1; inf = False
    for ch in s:
        if ch in NUM_MAP and not inf:
            val = val*10 + NUM_MAP[ch]
        elif ch in ("点","."):
            inf = True
        elif ch in NUM_MAP and inf:
            frac += NUM_MAP[ch]*base; base *= 0.1
    return val + frac if (val + frac) > 0 else 0.0

def extract_number(text: str) -> float:
    m = re.search(r"(\d+(\.\d+)?)", text)
    if m: return float(m.group(1))
    m2 = re.search(r"([零〇一二两三四五六七八九十点\.]+)", text)
    return cn_num_to_float(m2.group(1)) if m2 else 0.0

def clamp(v, lo, hi): return max(lo, min(hi, v))

def fuzzy_has(patterns, s):
    return any(re.search(p, s) for p in patterns)

# 容错关键词
FWD_PAT   = [r"前.?进", r"向.?前", r"往.?前", r"走.{0,2}前"]
BACK_PAT  = [r"后.?退", r"向.?后", r"往.?后", r"倒.?退"]
LTURN_PAT = [r"左.?转", r"向.?左.?转", r"往.?左.?转"]
RTURN_PAT = [r"右.?转", r"向.?右.?转", r"往.?右.?转"]
LSHIFT_PAT= [r"左.?移", r"向.?左.?移", r"侧.?移.?左"]
RSHIFT_PAT= [r"右.?移", r"向.?右.?移", r"侧.?移.?右"]
STOP_PAT  = [r"停止", r"停下", r"紧急停止", r"别动"]
POSE_PAT  = [r"站.?好", r"站.?立", r"回到?.?准备", r"回到?.?home"]

def parse_command_cn(text: str) -> dict | None:
    t = text.replace(" ", "")
    t = t.replace("停止运动","停止").replace("停一下","停止")
    if fuzzy_has(STOP_PAT, t):
        return {"tool":"robot.stop","args":{}}
    if fuzzy_has(POSE_PAT, t):
        return {"tool":"robot.pose","args":{"name":"stand_ready"}}

    if fuzzy_has(FWD_PAT, t):
        dist = extract_number(t) or 0.5
        vx = clamp(DEFAULT_VX, 0.05, MAX_VX)
        dur = clamp(dist / vx, 0.5, MAX_DURATION)
        return {"tool":"robot.walk","args":{"vx": +vx, "vy": 0.0, "omega": 0.0, "duration": round(dur,2)}}

    if fuzzy_has(BACK_PAT, t):
        dist = extract_number(t) or 0.5
        vx = clamp(DEFAULT_VX, 0.05, MAX_VX)
        dur = clamp(dist / vx, 0.5, MAX_DURATION)
        return {"tool":"robot.walk","args":{"vx": -vx, "vy": 0.0, "omega": 0.0, "duration": round(dur,2)}}

    if fuzzy_has(LTURN_PAT, t):
        deg = extract_number(t) or 45.0
        om  = clamp(DEFAULT_TURN_RATE, 0.2, MAX_TURN_RATE)
        dur = clamp(math.radians(deg)/om, 0.3, MAX_DURATION)
        return {"tool":"robot.walk","args":{"vx": 0.0, "vy": 0.0, "omega": +om, "duration": round(dur,2)}}

    if fuzzy_has(RTURN_PAT, t):
        deg = extract_number(t) or 45.0
        om  = clamp(DEFAULT_TURN_RATE, 0.2, MAX_TURN_RATE)
        dur = clamp(math.radians(deg)/om, 0.3, MAX_DURATION)
        return {"tool":"robot.walk","args":{"vx": 0.0, "vy": 0.0, "omega": -om, "duration": round(dur,2)}}

    if fuzzy_has(LSHIFT_PAT, t):
        dist = extract_number(t) or 0.3
        vy   = clamp(DEFAULT_VX, 0.05, MAX_VX)
        dur  = clamp(dist / vy, 0.5, MAX_DURATION)
        return {"tool":"robot.walk","args":{"vx": 0.0, "vy": +vy, "omega": 0.0, "duration": round(dur,2)}}

    if fuzzy_has(RSHIFT_PAT, t):
        dist = extract_number(t) or 0.3
        vy   = clamp(DEFAULT_VX, 0.05, MAX_VX)
        dur  = clamp(dist / vy, 0.5, MAX_DURATION)
        return {"tool":"robot.walk","args":{"vx": 0.0, "vy": -vy, "omega": 0.0, "duration": round(dur,2)}}

    return None

# ===================== WebSocket 处理 =====================
def _handle_and_reply(ws, buf: bytearray):
    """公共提交逻辑：保存 -> 过滤 -> ASR -> 唤醒词 -> 解析 -> 回复"""
    saved = save_wav_and_metrics(buf)
    print(f"[WAV] saved to {saved['path']} dur={saved['duration_s']}s rms={saved['rms']}")

    # ① 时长/能量过滤
    if saved["duration_s"] < MIN_UTT_SEC or saved["rms"] < MIN_RMS:
        print(f"[DROP] ignored short/silent utt (dur={saved['duration_s']}s, rms={saved['rms']})")
        return {"type":"ignored","reason":"short_or_silent","saved":saved}, None

    # ② 识别 & 归一化
    text = run_asr_pcm16(buf)
    text = normalize_zh_cmd(text)

    # ③ 唤醒词门控（可关）
    if REQUIRE_WAKE:
        if not any(text.startswith(w) for w in WAKE_WORDS):
            print(f"[DROP] missing wake word: {text!r}")
            return {"type":"ignored","reason":"no_wake_word","asr_text":text,"saved":saved}, None
        # 去掉唤醒词前缀（便于解析）
        for w in WAKE_WORDS:
            if text.startswith(w):
                text = text[len(w):].lstrip("，。,.：: ")
                break

    cmd = parse_command_cn(text) if text else None
    print("[ASR]", text, "->", cmd)
    return {"type":"asr_ctrl","asr_text":text,"cmd":cmd,"saved":saved}, cmd

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text(json.dumps({"type":"hello","msg":"server ready (lazy ASR, wake+filters)"}))
    print("[WS] connection open")

    buf = bytearray()
    frames = 0
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                print("[WS] disconnect")
                break

            if (b := msg.get("bytes")) is not None:
                buf.extend(b); frames += 1
                # 超长保护：超过 MAX_UTT_SEC 自动切分提交一次
                max_bytes = int(MAX_UTT_SEC * RATE * CHANNELS * SAMPWIDTH)
                if len(buf) >= max_bytes:
                    print(f"[WS] force-commit by MAX_UTT_SEC, bytes={len(buf)}")
                    payload, _ = _handle_and_reply(ws, buf)
                    await ws.send_text(json.dumps(payload))
                    buf.clear(); frames = 0
                elif frames % 50 == 0:
                    print(f"[WS] received frames={frames}, bytes_total={len(buf)}")
                continue

            if (t := msg.get("text")) is not None:
                try:
                    meta = json.loads(t)
                except Exception:
                    await ws.send_text(json.dumps({"type":"ack_raw","text":t}))
                    continue

                if meta.get("event") == "end_of_utterance":
                    print(f"[WS] end_of_utterance received; total_bytes={len(buf)}")
                    payload, _ = _handle_and_reply(ws, buf)
                    await ws.send_text(json.dumps(payload))
                    buf.clear(); frames = 0
                else:
                    await ws.send_text(json.dumps({"type":"ack","recv":meta}))
    except WebSocketDisconnect:
        print("[WS] client disconnected")
    finally:
        if buf:
            saved = save_wav_and_metrics(buf)
            print("[INFO] saved on disconnect:", saved)

if __name__ == "__main__":
    print("[BOOT] starting uvicorn ...")
    uvicorn.run("server_asr_ctrl_test_lazy:app", host="0.0.0.0", port=8765, reload=False)

