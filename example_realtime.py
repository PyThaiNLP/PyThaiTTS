#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example for Real-time TTS using FastThaiG2P.

Usage:
    python example_realtime.py
"""

from pythaitts import TTS

def main():
    print("PyThaiTTS Realtime TTS Example (fastthaig2p)")
    print("-" * 50)

    # Initialize FastThaiG2P TTS
    tts = TTS(pretrained="fastthaig2p")

    # 1. Stream speech from text string chunk-by-chunk
    text = "สวัสดีครับ ยินดีต้อนรับสู่ระบบสังเคราะห์เสียงภาษาไทยแบบเรียลไทม์"
    print(f"Streaming text: {text}\n")

    for i, audio_chunk in enumerate(tts.stream(text, return_type="waveform"), start=1):
        duration = len(audio_chunk) / 24000.0
        print(f"  Chunk {i}: {len(audio_chunk)} samples ({duration:.2f}s of audio)")

    print("\nDone!")

if __name__ == "__main__":
    main()
