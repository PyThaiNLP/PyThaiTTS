#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple demo script for PyThaiTTS
This script demonstrates Thai text-to-speech synthesis using PyThaiTTS.
"""

from pythaitts import TTS

def main():
    print("=" * 60)
    print("PyThaiTTS Demo - Thai Text-to-Speech")
    print("=" * 60)
    print()
    
    # Initialize TTS with default model (fastthaig2p)
    print("Initializing TTS model (fastthaig2p)...")
    try:
        tts = TTS()
        print("✓ TTS model loaded successfully!")
        print()
        
        # Sample Thai text
        text = "สวัสดีครับ ยินดีต้อนรับสู่ พายไทยทีทีเอส"
        print(f"Input text: {text}")
        print()
        
        # Generate speech and save to file
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output_{timestamp}.wav"
        print(f"Generating speech and saving to {output_file}...")
        result = tts.tts(text, filename=output_file)
        print(f"✓ Speech generated successfully!")
        print(f"Output saved to: {result}")
        print()
        
        # Also demonstrate getting waveform
        print("Generating waveform...")
        waveform = tts.tts(text, return_type="waveform")
        print(f"✓ Waveform generated successfully!")
        print(f"Waveform shape: {waveform.shape}")
        print()
        
        print("=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
