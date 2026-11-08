# -*- coding: utf-8 -*-
"""
Unit tests for MakHamTTS and MakHamTTS Paradee models
"""
import os
import tempfile
import unittest
import numpy as np

from pythaitts import TTS, MakHamTTS, MakHamTTSParadee


class TestMakHamTTS(unittest.TestCase):
    """Test MakHamTTS and MakHamTTSParadee models"""

    def test_paradee_direct_init(self):
        tts = MakHamTTSParadee()
        self.assertEqual(tts.sample_rate, 24000)
        self.assertEqual(tts.speed, 1.0)
        audio = tts.generate("สวัสดีครับ")
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_paradee_empty_text(self):
        tts = MakHamTTSParadee()
        audio = tts.generate("")
        self.assertEqual(len(audio), 0)

    def test_paradee_file_output(self):
        tts = MakHamTTSParadee()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            temp_path = fp.name
        try:
            out_file = tts(text="สวัสดีครับ", return_type="file", filename=temp_path)
            self.assertEqual(out_file, temp_path)
            self.assertTrue(os.path.exists(temp_path))
            self.assertGreater(os.path.getsize(temp_path), 44)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_paradee_tts_wrapper(self):
        tts = TTS(pretrained="makham_paradee")
        wav = tts.tts("สวัสดีครับ", return_type="waveform")
        self.assertIsInstance(wav, np.ndarray)
        self.assertGreater(len(wav), 0)

        tts_alias = TTS(pretrained="MakHamTTS_paradee")
        wav_alias = tts_alias.tts("สวัสดีครับ", return_type="waveform")
        self.assertIsInstance(wav_alias, np.ndarray)
        self.assertGreater(len(wav_alias), 0)

    def test_paradee_custom_dict(self):
        tts = TTS(pretrained="makham_paradee", custom_dict={"ทดสอบ": "ทบ-สอบ"})
        wav = tts.tts("ทดสอบ", return_type="waveform")
        self.assertIsInstance(wav, np.ndarray)
        self.assertGreater(len(wav), 0)

    def test_paradee_stream(self):
        tts = TTS(pretrained="makham_paradee")
        chunks = list(tts.stream("สวัสดีครับ ยินดีต้อนรับ"))
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, np.ndarray)
            self.assertGreater(len(chunk), 0)

    def test_makham_base_tts_wrapper(self):
        tts = TTS(pretrained="makham")
        wav = tts.tts("สวัสดีครับ", speaker_idx="newvoice_female", return_type="waveform")
        self.assertIsInstance(wav, np.ndarray)
        self.assertGreater(len(wav), 0)

        tts_alias = TTS(pretrained="MakHamTTS")
        wav_alias = tts_alias.tts("สวัสดีครับ", return_type="waveform")
        self.assertIsInstance(wav_alias, np.ndarray)
        self.assertGreater(len(wav_alias), 0)

    def test_makham_base_stream(self):
        tts = TTS(pretrained="makham")
        chunks = list(tts.stream("สวัสดีครับ ยินดีต้อนรับ", speaker_idx="newvoice_female"))
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, np.ndarray)
            self.assertGreater(len(chunk), 0)


if __name__ == "__main__":
    unittest.main()
