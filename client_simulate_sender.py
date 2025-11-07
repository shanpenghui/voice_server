#!/usr/bin/env python3
import argparse
import asyncio
import json
import math
import struct
from pathlib import Path

import numpy as np
import websockets

RATE = 16000          # 必须与 server_rx_test.py 一致
CHANNELS = 1
SAMPWIDTH = 2         # 16-bit
FRAME_MS = 30         # 与 Zero 端保持一致
FRAME_SAMPLES = int(RATE * FRAME_MS / 1000)  # 480 samples @ 16kHz

def gen_tone_pcm(duration_s: float, freq_hz: float = 440.0, amplitude: float = 0.5) -> bytes:
    """生成 16kHz/16-bit/mono 的纯音 PCM"""
    n = int(RATE * duration_s)
    t = np.arange(n, dtype=np.float32) / RATE
    # 剪裁到 int16
    x = (amplitude * np.sin(2 * math.pi * freq_hz * t) * 32767.0).astype(np.int16)
    return x.tobytes()

def read_wav_pcm16_mono(path: Path) -> bytes:
    """读取已为 16kHz/16-bit/mono 的 WAV，返回其数据段（不做重采样）"""
    import wave
    with wave.open(str(path), "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        assert ch == 1, f"WAV需单声道，实际={ch}"
        assert sr == RATE, f"WAV采样率需 {RATE} Hz，实际={sr}"
        assert sw == SAMPWIDTH, f"WAV采样位宽需 16-bit，实际={sw*8} bit"
        data = wf.readframes(wf.getnframes())
    return data

async def send_pcm_over_ws(server_url: str, pcm: bytes, multiple_utterances: int = 1, gap_s: float = 0.6):
    """将 PCM 切成 30ms 帧发给服务器，然后发 end_of_utterance；可循环多次"""
    async with websockets.connect(server_url, max_size=2**24, ping_interval=20, ping_timeout=20) as ws:
        print(f"[WS] Connected: {server_url}")

        async def recv_task():
            try:
                async for msg in ws:
                    print("[SRV]", msg)
            except Exception as e:
                print("[WS] recv ended:", e)

        rt = asyncio.create_task(recv_task())

        # 将整段 PCM 切片成 30ms 帧
        frame_bytes = FRAME_SAMPLES * SAMPWIDTH
        total_len = len(pcm)
        for k in range(multiple_utterances):
            sent = 0
            while sent < total_len:
                chunk = pcm[sent:sent + frame_bytes]
                if not chunk:
                    break
                # 如果最后一帧不足一整帧，用静音补齐
                if len(chunk) < frame_bytes:
                    chunk += b"\x00" * (frame_bytes - len(chunk))
                await ws.send(chunk)
                sent += frame_bytes
                await asyncio.sleep(FRAME_MS / 1000.0)  # 模拟实时发送

            # 一句话结束 → 发送 end_of_utterance
            await ws.send(json.dumps({"event": "end_of_utterance"}))
            print("[WS] sent end_of_utterance")

            # 话轮之间留点空隙
            await asyncio.sleep(gap_s)

        await asyncio.sleep(0.5)  # 等待服务器最后一个回包
        rt.cancel()

def main():
    ap = argparse.ArgumentParser(description="Simulate Zero-side sender for server_rx_test.py")
    ap.add_argument("--server", required=True, help="ws://HOST:PORT/ws")
    ap.add_argument("--mode", choices=["tone", "wav"], default="tone")
    ap.add_argument("--duration", type=float, default=2.0, help="seconds (tone mode)")
    ap.add_argument("--freq", type=float, default=440.0, help="Hz (tone mode)")
    ap.add_argument("--wav_path", type=str, help="path to 16kHz/mono/16-bit wav (wav mode)")
    ap.add_argument("--repeat", type=int, default=1, help="send the utterance N times")
    ap.add_argument("--gap", type=float, default=0.6, help="gap seconds between utterances")
    args = ap.parse_args()

    if args.mode == "tone":
        pcm = gen_tone_pcm(args.duration, args.freq)
        print(f"[PCM] tone {args.freq} Hz, {args.duration:.2f}s, bytes={len(pcm)}")
    else:
        assert args.wav_path, "--wav_path is required in wav mode"
        pcm = read_wav_pcm16_mono(Path(args.wav_path))
        print(f"[PCM] wav '{args.wav_path}', bytes={len(pcm)}")

    asyncio.run(send_pcm_over_ws(args.server, pcm, multiple_utterances=args.repeat, gap_s=args.gap))

if __name__ == "__main__":
    main()

