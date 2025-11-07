#!/usr/bin/env python3
# Unified voice client: --mode keyboard (PC) / --mode gpio (Zero2W)
import os, sys, asyncio, json, signal, queue, threading, time, datetime, wave
from pathlib import Path
import argparse

import numpy as np
import sounddevice as sd
import websockets

# ====== 默认参数 ======
RATE = 16000
CHANNELS = 1
FRAME_MS = 20                                # 低延迟
FRAME_SAMPLES = int(RATE * FRAME_MS / 1000)  # 320
DTYPE = "int16"

DEFAULT_WS = os.environ.get("SERVER_WS", "ws://127.0.0.1:8765/ws")
DEFAULT_AUDIO_KEYS = [k.strip().lower() for k in os.environ.get(
    "AUDIO_KEYS", "USB,mini,mic,microphone,UGREEN").split(",")
]

SAVE_LOCAL_DEFAULT = int(os.environ.get("SAVE_LOCAL", "0"))
SAVE_DIR = Path(os.environ.get("SAVE_DIR", "/dev/shm/rec"))  # RAM 盘，重启清空

# ====== 公共工具 ======
def pick_input_device(keys):
    """按名称关键词挑选输入设备；找不到返回 None 用系统默认。"""
    try:
        devs = sd.query_devices()
        for idx, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                name = (d.get("name") or "").lower()
                if any(k in name for k in keys):
                    return idx
    except Exception:
        pass
    return None

def save_wav_and_report(pcm: bytes):
    if not pcm:
        return None
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    day = SAVE_DIR / datetime.date.today().strftime("%Y%m%d")
    day.mkdir(exist_ok=True)
    p = day / f"utt_{datetime.datetime.now().strftime('%H%M%S_%f')}.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # S16_LE
        wf.setframerate(RATE)
        wf.writeframes(pcm)
    a = np.frombuffer(pcm[:len(pcm)-(len(pcm)%2)], dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(a*a))) if a.size else 0.0
    dur = len(pcm) / (RATE * CHANNELS * 2)
    print(f"[SAVE] {p} dur={dur:.2f}s rms={rms:.1f}")
    return {"path": str(p), "duration_s": round(dur, 3), "rms": round(rms, 1)}

def build_stream(device_index, q_frames):
    """创建 sounddevice 输入流，回调推进队列。"""
    def audio_cb(indata, frames, time_info, status):
        try:
            q_frames.put_nowait(indata.copy().tobytes())
        except queue.Full:
            pass  # 为实时性允许丢帧
    stream = sd.InputStream(
        samplerate=RATE, channels=CHANNELS, dtype=DTYPE,
        blocksize=FRAME_SAMPLES, callback=audio_cb,
        device=device_index, latency="low"
    )
    stream.start()
    print(f"[Audio] 打开麦克风成功 device={device_index} (None=default)")
    return stream

async def ws_loop(server_ws, speaking_event, stop_event, q_frames, save_local):
    """通用 WS 发送循环：讲话时发 20ms 帧；静音边沿发 end_of_utterance。"""
    reconnect = 3
    pcm_buf = bytearray()
    was_speaking = False

    while not stop_event.is_set():
        try:
            print(f"[WS] 连接 {server_ws} ...")
            async with websockets.connect(
                server_ws, max_size=2**23, ping_interval=20, ping_timeout=20
            ) as ws:
                print("[WS] 已连接。按下开始发送，松开提交。")
                async def recv_task():
                    try:
                        async for m in ws:
                            print("[SRV]", m)
                    except Exception as e:
                        print("[WS] 接收结束:", e)
                rt = asyncio.create_task(recv_task())

                while not stop_event.is_set():
                    try:
                        frame = q_frames.get(timeout=0.1)
                    except queue.Empty:
                        frame = None

                    if speaking_event.is_set():
                        if frame is not None:
                            await ws.send(frame)
                            if save_local:
                                pcm_buf.extend(frame)
                        was_speaking = True
                    else:
                        if was_speaking:
                            await ws.send(json.dumps({"event": "end_of_utterance"}))
                            print("[PTT] 已发送 end_of_utterance")
                            if save_local and len(pcm_buf) > 0:
                                save_wav_and_report(pcm_buf)
                                pcm_buf.clear()
                            was_speaking = False
                        await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[WS] 连接异常: {e} -> {reconnect}s 后重连")
            await asyncio.sleep(reconnect)

# ====== 键盘模式（PC） ======
def run_keyboard_mode(args):
    try:
        from pynput import keyboard
    except Exception as e:
        print("缺少 pynput：请先安装 `python3 -m pip install --user pynput`")
        sys.exit(1)

    speaking = threading.Event()
    stop_event = threading.Event()
    q_frames: "queue.Queue[bytes]" = queue.Queue(maxsize=200)

    # 键盘监听：空格 PTT，ESC 退出
    def on_press(key):
        if key == keyboard.Key.space:
            if not speaking.is_set():
                speaking.set()
                print("[PTT] 空格按下 -> 开始发送")
    def on_release(key):
        if key == keyboard.Key.space:
            if speaking.is_set():
                speaking.clear()
                print("[PTT] 空格松开 -> 提交")
        elif key == keyboard.Key.esc:
            print("[Exit] ESC")
            stop_event.set()
            return False
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # 采集流
    dev_idx = args.device if args.device is not None else pick_input_device(args.audio_keys)
    try:
        stream = build_stream(dev_idx, q_frames)
    except Exception as e:
        print("[Audio] 打开失败：", e)
        try: print(sd.query_devices())
        except Exception: pass
        sys.exit(1)

    def _sigterm(*_):
        print("[Exit] SIGTERM"); stop_event.set()
    signal.signal(signal.SIGTERM, _sigterm)

    try:
        asyncio.run(ws_loop(args.server, speaking, stop_event, q_frames, args.save_local))
    finally:
        try: stream.stop(); stream.close()
        except Exception: pass
        listener.stop()
        print("[INFO] 退出完成。")

# ====== GPIO 模式（Zero2W） ======
def run_gpio_mode(args):
    try:
        from gpiozero import Button, LED
    except Exception:
        print("缺少 gpiozero：请先安装 `sudo apt-get install -y python3-gpiozero`")
        sys.exit(1)

    speaking = threading.Event()
    stop_event = threading.Event()
    q_frames: "queue.Queue[bytes]" = queue.Queue(maxsize=200)

    btn = Button(args.btn_pin, pull_up=True, bounce_time=0.02)
    led = None
    try:
        led = LED(args.led_pin); led.off()
    except Exception:
        pass

    btn.when_pressed  = lambda: (speaking.set(),  led and led.on(),  print("[PTT] 按下 -> 开始发送"))
    btn.when_released = lambda: (speaking.clear(), led and led.off(), print("[PTT] 松开 -> 提交"))

    dev_idx = args.device if args.device is not None else pick_input_device(args.audio_keys)
    try:
        stream = build_stream(dev_idx, q_frames)
    except Exception as e:
        print("[Audio] 打开失败：", e)
        try: print(sd.query_devices())
        except Exception: pass
        sys.exit(1)

    def _sigterm(*_):
        print("[Exit] SIGTERM"); stop_event.set()
    signal.signal(signal.SIGTERM, _sigterm)

    try:
        asyncio.run(ws_loop(args.server, speaking, stop_event, q_frames, args.save_local))
    finally:
        try: stream.stop(); stream.close()
        except Exception: pass
        print("[INFO] 退出完成。")

# ====== main ======
def main():
    ap = argparse.ArgumentParser(description="Unified voice PTT client (keyboard/gpio)")
    ap.add_argument("--mode", choices=["keyboard","gpio"], default="keyboard",
                    help="keyboard=按空格录音（PC）；gpio=GPIO按键录音（Zero2W）")
    ap.add_argument("--server", default=DEFAULT_WS, help="WebSocket server, e.g. ws://<ip>:8765/ws")
    ap.add_argument("--device", type=int, default=None, help="sounddevice 输入设备索引")
    ap.add_argument("--audio-keys", nargs="*", default=DEFAULT_AUDIO_KEYS,
                    help="按名称关键词自动选麦（空格分隔多个）")
    ap.add_argument("--save-local", type=int, default=SAVE_LOCAL_DEFAULT,
                    help="每句话保存本地WAV到 /dev/shm/rec (0/1)")
    # GPIO only:
    ap.add_argument("--btn-pin", type=int, default=int(os.environ.get("BTN_PIN","17")))
    ap.add_argument("--led-pin", type=int, default=int(os.environ.get("LED_PIN","27")))
    args = ap.parse_args()

    if args.mode == "keyboard":
        run_keyboard_mode(args)
    else:
        run_gpio_mode(args)

if __name__ == "__main__":
    main()

