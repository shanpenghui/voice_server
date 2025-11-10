#!/usr/bin/env python3
# voice_client.py
# PTT（按键发言）客户端：支持 keyboard（PC）/ gpio（Zero2W）
# 关键特性：
#  - 自动按硬件原生采样率打开麦克风（避免 Invalid sample rate）
#  - 在客户端重采样到 16 kHz（RATE_OUT）再发给服务器
#  - WebSocket 链路：发送 PCM 分帧 + {"event":"end_of_utterance","seq":N}
#  - 规范回包处理：asr_ctrl / ignored / nack（含 ok/seq/reason）
#  - 可选本地保存每段语音（重采样后的16k）

import os, sys, asyncio, json, signal, queue, threading, time, datetime, wave
from pathlib import Path
import argparse
import numpy as np
import sounddevice as sd
import websockets

# ========= 发送侧采样率（固定 16k）=========
RATE_OUT = 16000            # 发送给服务器的目标采样率
CHANNELS = 1
FRAME_MS = 20               # 回调帧时长（毫秒），仅影响输入端 blocksize
DTYPE = "int16"

DEFAULT_WS = os.environ.get("SERVER_WS", "ws://127.0.0.1:8765/ws")
DEFAULT_AUDIO_KEYS = [k.strip().lower() for k in os.environ.get(
    "AUDIO_KEYS", "USB,mini,mic,microphone,UGREEN").split(",")
]

SAVE_LOCAL_DEFAULT = int(os.environ.get("SAVE_LOCAL", "0"))
SAVE_DIR = Path(os.environ.get("SAVE_DIR", "/dev/shm/rec"))

# ========= 工具 =========
def pick_input_device(keys):
    """按关键词在设备名里筛一个输入设备索引；找不到返回 None 使用默认。"""
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

def save_wav_and_report(pcm_16k: bytes):
    """保存重采样后的 16k PCM 到本地，便于复盘链路。"""
    if not pcm_16k:
        return None
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    day = SAVE_DIR / datetime.date.today().strftime("%Y%m%d")
    day.mkdir(exist_ok=True)
    p = day / f"utt_{datetime.datetime.now().strftime('%H%M%S_%f')}.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE_OUT)
        wf.writeframes(pcm_16k)
    a = np.frombuffer(pcm_16k[:len(pcm_16k)-(len(pcm_16k)%2)], dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(a*a))) if a.size else 0.0
    dur = len(pcm_16k) / (RATE_OUT * CHANNELS * 2)
    print(f"[SAVE] {p} dur={dur:.2f}s rms={rms:.1f}")
    return {"path": str(p), "duration_s": round(dur, 3), "rms": round(rms, 1)}

def resample_to_16k(pcm_bytes: bytes, rate_in: int) -> bytes:
    """把任意 rate_in 的 int16 PCM（单声道）重采样到 16k。
       - 48k → 16k：整比抽取（3:1），足够用于命令语音
       - 44.1k/其他 → 16k：线性插值
    """
    if rate_in == RATE_OUT:
        return pcm_bytes

    a = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

    if rate_in == 48000:
        # 简单整比抽取（可选加低通，命令语音足够）
        a = a[::3]
    else:
        n_in  = a.shape[0]
        n_out = int(round(n_in * (RATE_OUT / float(rate_in))))
        if n_in == 0 or n_out <= 0:
            return b""
        x  = np.linspace(0.0, 1.0, num=n_in,  endpoint=False)
        xi = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        a  = np.interp(xi, x, a).astype(np.float32)

    a = np.clip(a, -32768.0, 32767.0).astype(np.int16)
    return a.tobytes()

def build_stream(device_index, q_frames):
    """按设备默认采样率打开输入流，并返回 (stream, rate_in)。"""
    devinfo = sd.query_devices(device_index if device_index is not None else None, 'input')
    rate_in = int(devinfo.get("default_samplerate", 48000) or 48000)
    frame_samples = int(rate_in * FRAME_MS / 1000)

    def audio_cb(indata, frames, time_info, status):
        # 回调里不可阻塞；若队列满，直接丢弃本帧，避免卡顿
        try:
            q_frames.put_nowait(indata.copy().tobytes())
        except queue.Full:
            pass

    stream = sd.InputStream(
        samplerate=rate_in, channels=CHANNELS, dtype=DTYPE,
        blocksize=frame_samples, callback=audio_cb,
        device=device_index, latency="low"
    )
    stream.start()
    print(f"[Audio] 打开麦克风成功 device={device_index} rate_in={rate_in} (None=default)")
    return stream, rate_in

# ========= WS 主循环 =========
async def ws_loop(server_ws, speaking_event, stop_event, q_frames, save_local, rate_in):
    reconnect = 3
    pcm_buf_16k = bytearray()   # 本地可选保存的 16k 流
    was_speaking = False
    seq = 0
    last_cmd = None

    while not stop_event.is_set():
        try:
            print(f"[WS] 连接 {server_ws} ...")
            async with websockets.connect(
                server_ws, max_size=2**23, ping_interval=20, ping_timeout=20
            ) as ws:
                print("[WS] 已连接。按下开始发送，松开提交。")

                async def recv_task():
                    nonlocal last_cmd
                    try:
                        async for m in ws:
                            try:
                                data = json.loads(m)
                            except Exception:
                                print("[SRV/raw]", m); continue

                            t = data.get("type")
                            if t == "hello":
                                print("[HELLO]", data.get("msg"))
                            elif t == "ignored":
                                print(f"[IGNORED] seq={data.get('seq')} reason={data.get('reason')} saved={data.get('saved')}")
                            elif t == "asr_ctrl":
                                if data.get("ok"):
                                    cmd = data.get("cmd") or {}
                                    last_cmd = cmd
                                    print(f"[CMD] seq={data.get('seq')} {data.get('asr_text')} -> {cmd}")
                                    # TODO: 在这里对接机器人控制（openminiduck/ROS2/MQTT），此处仅打印
                                else:
                                    print(f"[NACK] seq={data.get('seq')} reason={data.get('reason')} text={data.get('asr_text')}")
                            else:
                                print("[SRV]", data)
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
                            # 采集端到 16k 的重采样
                            frame16 = resample_to_16k(frame, rate_in)
                            if frame16:
                                await ws.send(frame16)
                                if save_local:
                                    pcm_buf_16k.extend(frame16)
                        was_speaking = True
                    else:
                        if was_speaking:
                            seq += 1
                            await ws.send(json.dumps({"event":"end_of_utterance","seq":seq}))
                            print(f"[PTT] 已发送 end_of_utterance seq={seq}")
                            if save_local and len(pcm_buf_16k) > 0:
                                save_wav_and_report(pcm_buf_16k)
                                pcm_buf_16k.clear()
                            was_speaking = False
                        await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[WS] 连接异常: {e} -> {reconnect}s 后重连")
            await asyncio.sleep(reconnect)

# ========= 键盘模式（PC） =========
def run_keyboard_mode(args):
    try:
        from pynput import keyboard
    except Exception:
        print("缺少 pynput：请先安装 `python3 -m pip install --user pynput`")
        sys.exit(1)

    speaking = threading.Event()
    stop_event = threading.Event()
    q_frames: "queue.Queue[bytes]" = queue.Queue(maxsize=200)

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
            print("[Exit] ESC"); stop_event.set(); return False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    dev_idx = args.device if args.device is not None else pick_input_device(args.audio_keys)
    try:
        stream, rate_in = build_stream(dev_idx, q_frames)
    except Exception as e:
        print("[Audio] 打开失败：", e)
        try: 
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                print(f"{i:3d} {d.get('name','')}, ALSA ({d.get('max_input_channels',0)} in, {d.get('max_output_channels',0)} out)")
        except Exception:
            pass
        sys.exit(1)

    def _sigterm(*_): print("[Exit] SIGTERM"); stop_event.set()
    signal.signal(signal.SIGTERM, _sigterm)

    try:
        asyncio.run(ws_loop(args.server, speaking, stop_event, q_frames, args.save_local, rate_in))
    finally:
        try: stream.stop(); stream.close()
        except Exception: pass
        listener.stop()
        print("[INFO] 退出完成。")

# ========= GPIO 模式（Zero2W） =========
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
        stream, rate_in = build_stream(dev_idx, q_frames)
    except Exception as e:
        print("[Audio] 打开失败：", e)
        try: 
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                print(f"{i:3d} {d.get('name','')}, ALSA ({d.get('max_input_channels',0)} in, {d.get('max_output_channels',0)} out)")
        except Exception:
            pass
        sys.exit(1)

    def _sigterm(*_): print("[Exit] SIGTERM"); stop_event.set()
    signal.signal(signal.SIGTERM, _sigterm)

    try:
        asyncio.run(ws_loop(args.server, speaking, stop_event, q_frames, args.save_local, rate_in))
    finally:
        try: stream.stop(); stream.close()
        except Exception: pass
        print("[INFO] 退出完成。")

# ========= main =========
def main():
    ap = argparse.ArgumentParser(description="Unified voice PTT client (keyboard/gpio)")
    ap.add_argument("--mode", choices=["keyboard","gpio"], default="keyboard",
                    help="keyboard=按空格录音（PC）；gpio=GPIO按键录音（Zero2W）")
    ap.add_argument("--server", default=DEFAULT_WS, help="WebSocket server, e.g. ws://<ip>:8765/ws")
    ap.add_argument("--host", default=None, help="服务器IP，简写用法（会拼成 ws://<host>:8765/ws）")
    ap.add_argument("--port", type=int, default=8765, help="与 --host 搭配使用的端口（默认 8765）")
    ap.add_argument("--device", type=int, default=None, help="sounddevice 输入设备索引")
    ap.add_argument("--audio-keys", nargs="*", default=DEFAULT_AUDIO_KEYS,
                    help="按名称关键词自动选麦（空格分隔多个）")
    ap.add_argument("--save-local", type=int, default=SAVE_LOCAL_DEFAULT,
                    help="保存每段 16k WAV 到 /dev/shm/rec (0/1)")
    # GPIO only:
    ap.add_argument("--btn-pin", type=int, default=int(os.environ.get("BTN_PIN","17")))
    ap.add_argument("--led-pin", type=int, default=int(os.environ.get("LED_PIN","27")))
    args = ap.parse_args()

    # 简写：--host 优先生效（便于经常换 IP 的场景）
    if args.host:
        args.server = f"ws://{args.host}:{args.port}/ws"

    if args.mode == "keyboard":
        run_keyboard_mode(args)
    else:
        run_gpio_mode(args)

if __name__ == "__main__":
    main()
