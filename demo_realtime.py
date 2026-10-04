#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-time Text-to-Speech Demo for FastThaiG2P in PyThaiTTS.

Demonstrates:
1. Low-latency streaming TTS from text string (chunk by chunk).
2. Streaming TTS from an LLM-like token generator.
3. Streaming 16-bit PCM bytes for audio pipelines/WebSockets.
4. FastThaiG2PEngine integration with KoljaB/RealtimeTTS.
"""

import time
from pythaitts import TTS, RealtimeTTS, FastThaiG2PEngine


def main():
    print("=" * 65)
    print("PyThaiTTS - Real-time TTS Demo (FastThaiG2P)")
    print("=" * 65)
    print()

    # 1. Initialize TTS model
    print("Initializing FastThaiG2P TTS model...")
    t_start = time.time()
    tts = TTS(pretrained="fastthaig2p")
    print(f"✓ Model loaded in {time.time() - t_start:.2f}s")
    print()

    # 2. Streaming from a full text string
    sample_text = (
        "สวัสดีครับ ยินดีต้อนรับสู่ระบบสังเคราะห์เสียงภาษาไทยแบบเรียลไทม์ "
        "ระบบนี้ช่วยให้สร้างเสียงพูดได้อย่างรวดเร็วและต่อเนื่อง "
        "โดยเริ่มส่งสัญญาณเสียงได้ทันทีตั้งแต่ข้อความส่วนแรกประมวลผลเสร็จ"
    )
    print("-----------------------------------------------------------------")
    print("Demo 1: Real-time Streaming from Full Text")
    print("-----------------------------------------------------------------")
    print(f"Input text:\n{sample_text}\n")

    t0 = time.time()
    total_audio_samples = 0
    chunk_count = 0
    ttfa = None  # Time to First Audio

    for audio_chunk in tts.stream(sample_text, return_type="waveform"):
        chunk_count += 1
        elapsed = time.time() - t0
        if ttfa is None:
            ttfa = elapsed
            print(f"⚡ Time to First Audio (TTFA): {ttfa * 1000:.1f} ms!")

        duration = len(audio_chunk) / 24000.0
        total_audio_samples += len(audio_chunk)
        print(
            f"  [Chunk {chunk_count}] Received {len(audio_chunk)} samples "
            f"({duration:.2f}s of audio) at +{elapsed:.2f}s"
        )

    total_time = time.time() - t0
    total_audio_sec = total_audio_samples / 24000.0
    rtf = total_time / total_audio_sec if total_audio_sec > 0 else 0
    print(f"\n✓ Generated {total_audio_sec:.2f}s of audio across {chunk_count} chunks in {total_time:.2f}s")
    print(f"  Real-Time Factor (RTF): {rtf:.3f} (< 1.0 means faster than real-time)")
    print()

    # 3. Streaming from simulated LLM token stream
    print("-----------------------------------------------------------------")
    print("Demo 2: Real-time Streaming from LLM Token Stream")
    print("-----------------------------------------------------------------")

    def simulate_llm_stream():
        tokens = [
            "สวัสดี", "ครับ", " ", "นี่", "คือ", "การ", "ทดสอบ", " ",
            "การ", "สตรีม", "มิ่ง", " ", "ข้อความ", "จาก", " ", "โมเดล",
            "ภาษา", "ขนาด", "ใหญ่", " ", "แบบ", "เรียล", "ไทม์", "ครับ"
        ]
        print("Streaming tokens from LLM: ", end="", flush=True)
        for tok in tokens:
            print(tok, end="", flush=True)
            time.sleep(0.04)  # simulate LLM generation delay
            yield tok
        print("\n")

    t0 = time.time()
    stream_chunks = 0
    for audio_chunk in tts.stream(simulate_llm_stream(), return_type="waveform"):
        stream_chunks += 1
        elapsed = time.time() - t0
        dur = len(audio_chunk) / 24000.0
        print(f"  [Audio Chunk {stream_chunks}] Duration: {dur:.2f}s at +{elapsed:.2f}s")

    print(f"✓ LLM streaming synthesis complete in {time.time() - t0:.2f}s\n")

    # 4. Streaming 16-bit PCM bytes
    print("-----------------------------------------------------------------")
    print("Demo 3: Streaming 16-bit PCM Bytes (for WebSockets / PyAudio)")
    print("-----------------------------------------------------------------")
    byte_chunks = 0
    total_bytes = 0
    for pcm in tts.stream("ระบบเสียงภาษาไทย คุณภาพสูง", return_type="bytes"):
        byte_chunks += 1
        total_bytes += len(pcm)
        print(f"  [PCM Chunk {byte_chunks}] Received {len(pcm)} bytes of raw 16-bit PCM")

    print(f"✓ Total raw PCM data: {total_bytes} bytes (24 kHz, 16-bit, mono)\n")

    # 5. RealtimeTTS Library compatibility
    print("-----------------------------------------------------------------")
    print("Demo 4: RealtimeTTS Engine (KoljaB/RealtimeTTS compatibility)")
    print("-----------------------------------------------------------------")
    try:
        from RealtimeTTS import TextToAudioStream
        print("RealtimeTTS library detected. Initializing FastThaiG2PEngine...")
        engine = FastThaiG2PEngine()
        stream = TextToAudioStream(engine)
        print("✓ FastThaiG2PEngine ready with RealtimeTTS TextToAudioStream!")
    except ImportError:
        print("KoljaB/RealtimeTTS library is not installed.")
        print("To use with RealtimeTTS:")
        print("  pip install pythaitts[realtime]")
        print("  or: pip install RealtimeTTS")
        print("\nFastThaiG2PEngine can then be used directly:")
        print("  from RealtimeTTS import TextToAudioStream")
        print("  from pythaitts.realtime import FastThaiG2PEngine")
        print("  engine = FastThaiG2PEngine()")
        print("  stream = TextToAudioStream(engine)")
        print("  stream.feed('สวัสดีครับ')")
        print("  stream.play()")

    print()
    print("=" * 65)
    print("Real-time TTS Demo completed successfully!")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    exit(main())
