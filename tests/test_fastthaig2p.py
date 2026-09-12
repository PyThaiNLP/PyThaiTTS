# -*- coding: utf-8 -*-
"""
Unit tests for FastThaiG2P TTS integration
"""
import os
import unittest
from unittest.mock import Mock, patch
import numpy as np

from pythaitts import TTS
from pythaitts.pretrained.fastthaig2p import G2P, Tokenizer, normalize, ipa_to_kokoro, FastThaiG2P


class TestFastThaiG2PComponents(unittest.TestCase):
    """Test individual FastThaiG2P components (Normalizer, Kokoro, Tokenizer, G2P)"""

    def test_normalizer_plain_text(self):
        self.assertEqual(normalize("สวัสดีครับ"), "สวัสดีครับ")

    def test_normalizer_numbers(self):
        self.assertEqual(normalize("มี 42 คน"), "มี สี่สิบสอง คน")

    def test_normalizer_decimal(self):
        self.assertEqual(normalize("3.14"), "สามจุดหนึ่งสี่")

    def test_normalizer_mai_yamok(self):
        self.assertEqual(normalize("เด็กๆ"), "เด็กเด็ก")

    def test_normalizer_abbreviation(self):
        self.assertIn("มกราคม", normalize("ม.ค."))
        self.assertIn("ด็อกเตอร์", normalize("ดร."))

    def test_kokoro_ipa_mapping(self):
        result = ipa_to_kokoro("/sa˨˩.wat̚˨˩.diː˧/ /kʰrap̚˦˥/")
        self.assertEqual(result, "sa↓.wat↓.diː→ kʰrap↑")

    def test_kokoro_distinct_tones(self):
        tones = {
            "˧": "→",    # mid
            "˨˩": "↓",   # low
            "˥˩": "↘",   # falling
            "˦˥": "↑",   # high
            "˩˩˦": "↗",  # rising
        }
        for chao, arrow in tones.items():
            self.assertEqual(ipa_to_kokoro(f"/maː{chao}/"), f"maː{arrow}")

    def test_tokenizer_and_g2p(self):
        g2p = G2P()
        result = g2p.convert("กินข้าว")
        self.assertIn("kin˧", result)
        self.assertIn("kʰaːw˥˩", result)


class TestFastThaiG2PIntegration(unittest.TestCase):
    """Test FastThaiG2P integration via pythaitts.TTS"""

    @patch('pythaitts.pretrained.fastthaig2p.FastThaiG2P')
    def test_fastthaig2p_initialization(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_fastthaig2p_cls.return_value = mock_instance

        tts = TTS(pretrained="fastthaig2p")
        self.assertIsNotNone(tts.model)
        self.assertEqual(tts.pretrained, "fastthaig2p")
        mock_fastthaig2p_cls.assert_called_once_with(device="cpu")

    @patch('pythaitts.pretrained.fastthaig2p.FastThaiG2P')
    def test_fastthaig2p_default_initialization(self, mock_fastthaig2p_cls):
        """Test that FastThaiG2P is initialized by default when no pretrained is specified"""
        mock_instance = Mock()
        mock_fastthaig2p_cls.return_value = mock_instance

        tts = TTS()
        self.assertIsNotNone(tts.model)
        self.assertEqual(tts.pretrained, "fastthaig2p")
        mock_fastthaig2p_cls.assert_called_once_with(device="cpu")

    @patch('pythaitts.pretrained.fastthaig2p.FastThaiG2P')
    def test_fastthaig2p_case_insensitive_name(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_fastthaig2p_cls.return_value = mock_instance

        tts = TTS(pretrained="FastThaiG2P")
        self.assertIsNotNone(tts.model)
        self.assertEqual(tts.pretrained, "FastThaiG2P")

    @patch('pythaitts.pretrained.fastthaig2p.FastThaiG2P')
    def test_fastthaig2p_tts_file_call(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_instance.return_value = "/tmp/output.wav"
        mock_fastthaig2p_cls.return_value = mock_instance

        tts = TTS(pretrained="fastthaig2p")
        result = tts.tts("สวัสดีครับ", filename="/tmp/test.wav")

        mock_instance.assert_called_once()
        call_args = mock_instance.call_args
        self.assertEqual(call_args.kwargs['text'], "สวัสดีครับ")
        self.assertEqual(call_args.kwargs['speaker_idx'], "thai_som")
        self.assertEqual(call_args.kwargs['filename'], "/tmp/test.wav")
        self.assertEqual(call_args.kwargs['return_type'], "file")
        self.assertEqual(result, "/tmp/output.wav")

    @patch('pythaitts.pretrained.fastthaig2p.FastThaiG2P')
    def test_fastthaig2p_tts_waveform_call(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        dummy_audio = np.zeros(24000, dtype=np.float32)
        mock_instance.return_value = dummy_audio
        mock_fastthaig2p_cls.return_value = mock_instance

        tts = TTS(pretrained="fastthaig2p")
        result = tts.tts("สวัสดีครับ", return_type="waveform")

        call_args = mock_instance.call_args
        self.assertEqual(call_args.kwargs['return_type'], "waveform")
        self.assertEqual(result.shape, (24000,))

    @patch('pythaitts.pretrained.fastthaig2p.FastThaiG2P')
    def test_fastthaig2p_speaker_default_mapping(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_fastthaig2p_cls.return_value = mock_instance

        tts = TTS(pretrained="fastthaig2p")
        # Default speaker_idx in tts() is Linda, should map to thai_som
        tts.tts("สวัสดีครับ")
        call_args = mock_instance.call_args
        self.assertEqual(call_args.kwargs['speaker_idx'], "thai_som")

    @patch('pythaitts.pretrained.fastthaig2p.FastThaiG2P')
    def test_fastthaig2p_with_preprocessing(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_instance.return_value = "/tmp/output.wav"
        mock_fastthaig2p_cls.return_value = mock_instance

        tts = TTS(pretrained="fastthaig2p")
        tts.tts("มี 5 คนๆ", preprocess=True)

        call_args = mock_instance.call_args
        processed_text = call_args.kwargs['text']
        self.assertNotIn("5", processed_text)
        self.assertNotIn("ๆ", processed_text)
        self.assertIn("ห้า", processed_text)
        self.assertIn("คนคน", processed_text)


class TestFastThaiG2PDirect(unittest.TestCase):
    """Direct unit tests for FastThaiG2P class in fastthaig2p/tts.py"""

    @patch('onnxruntime.InferenceSession')
    @patch('pythaitts.pretrained.fastthaig2p.tts._default_asset')
    def test_direct_tts_call_waveform(self, mock_default_asset, mock_session_cls):
        import tempfile, json
        # Create a dummy config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
            json.dump({"vocab": {"a": 1, "b": 2}}, fp)
            dummy_config = fp.name
        # Create dummy voicepack
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as fp:
            np.save(fp.name, np.zeros((510, 1, 256), dtype=np.float32))
            dummy_voicepack = fp.name
        # Create dummy onnx
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as fp:
            dummy_onnx = fp.name

        try:
            model = FastThaiG2P(
                model_path=dummy_onnx,
                voicepack_path=dummy_voicepack,
                config_path=dummy_config,
            )

            # Mock generate
            dummy_audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
            model.generate = Mock(return_value=dummy_audio)

            # Test waveform return
            out = model("สวัสดี", return_type="waveform")
            self.assertTrue(np.array_equal(out, dummy_audio))

            # Test invalid speaker
            with self.assertRaises(ValueError):
                model("สวัสดี", speaker_idx="unsupported_voice")

            # Test Linda maps to thai_som
            model.synthesize = Mock(return_value="test.wav")
            model("สวัสดี", speaker_idx="Linda", filename="test.wav")
            model.synthesize.assert_called_once_with("สวัสดี", "test.wav")

        finally:
            for p in (dummy_config, dummy_voicepack, dummy_onnx):
                if os.path.exists(p):
                    os.unlink(p)


if __name__ == '__main__':
    unittest.main()
