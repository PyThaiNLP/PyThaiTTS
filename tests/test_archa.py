# -*- coding: utf-8 -*-
"""
Unit tests for ArchaTTS integration
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pythaitts import TTS


class TestArchaIntegration(unittest.TestCase):
    """Test ArchaTTS integration"""

    @patch('pythaitts.pretrained.archa_tts.ArchaTTS')
    def test_archa_model_initialization(self, mock_archa):
        """Test that ArchaTTS model can be initialized"""
        tts = TTS(pretrained="archa")
        self.assertIsNotNone(tts.model)
        self.assertEqual(tts.pretrained, "archa")

    @patch('pythaitts.pretrained.archa_tts.ArchaTTS')
    def test_archa_tts_call(self, mock_archa_class):
        """Test calling tts method with archa model"""
        mock_instance = Mock()
        mock_instance.return_value = "/tmp/output.wav"
        mock_archa_class.return_value = mock_instance

        tts = TTS(pretrained="archa")
        result = tts.tts("สวัสดีครับ", filename="/tmp/test.wav")

        mock_instance.assert_called_once()
        call_args = mock_instance.call_args
        self.assertEqual(call_args.kwargs['text'], "สวัสดีครับ")
        self.assertEqual(call_args.kwargs['filename'], "/tmp/test.wav")
        self.assertEqual(call_args.kwargs['return_type'], "file")

    @patch('pythaitts.pretrained.archa_tts.ArchaTTS')
    def test_archa_with_preprocessing(self, mock_archa_class):
        """Test that preprocessing works with archa model"""
        mock_instance = Mock()
        mock_instance.return_value = "/tmp/output.wav"
        mock_archa_class.return_value = mock_instance

        tts = TTS(pretrained="archa")
        tts.tts("มี 5 คนๆ", preprocess=True)

        mock_instance.assert_called_once()
        call_args = mock_instance.call_args
        processed_text = call_args.kwargs['text']

        # Text should have numbers converted and ๆ expanded
        self.assertNotIn("5", processed_text)
        self.assertNotIn("ๆ", processed_text)
        self.assertIn("ห้า", processed_text)
        self.assertIn("คนคน", processed_text)

    @patch('pythaitts.pretrained.archa_tts.ArchaTTS')
    def test_archa_waveform_return(self, mock_archa_class):
        """Test waveform return type for archa model"""
        mock_instance = Mock()
        mock_waveform = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        mock_instance.return_value = mock_waveform
        mock_archa_class.return_value = mock_instance

        tts = TTS(pretrained="archa")
        result = tts.tts("สวัสดี", return_type="waveform")

        mock_instance.assert_called_once()
        call_args = mock_instance.call_args
        self.assertEqual(call_args.kwargs['return_type'], "waveform")

    @patch('pythaitts.pretrained.archa_tts.ArchaTTS')
    def test_archa_no_filename_returns_temp_file(self, mock_archa_class):
        """Test that archa model returns a temp file path when filename is None"""
        mock_instance = Mock()
        mock_instance.return_value = "/tmp/tmpXXXXXX.wav"
        mock_archa_class.return_value = mock_instance

        tts = TTS(pretrained="archa")
        result = tts.tts("สวัสดี")

        mock_instance.assert_called_once()
        call_args = mock_instance.call_args
        self.assertIsNone(call_args.kwargs['filename'])


class TestArchaTTSUnit(unittest.TestCase):
    """Unit tests for ArchaTTS class methods"""

    def _make_archa(self):
        """Create an ArchaTTS instance with mocked dependencies (no real torch/snac needed)."""
        from pythaitts.pretrained.archa_tts import ArchaTTS

        archa = ArchaTTS.__new__(ArchaTTS)
        archa.device = "cpu"
        archa.tokenizer = MagicMock()
        archa.model = MagicMock()
        archa.snac_model = MagicMock()
        return archa

    def test_decode_tokens_empty(self):
        """Test _decode_tokens returns empty array for short token list."""
        from pythaitts.pretrained.archa_tts import ArchaTTS, AUDIO_TOKENS_START

        archa = self._make_archa()
        # fewer than 7 tokens → empty result
        result = archa._decode_tokens([AUDIO_TOKENS_START] * 3)
        self.assertEqual(len(result), 0)

    def test_denoise_fallback_without_noisereduce(self):
        """Test that _denoise returns audio unchanged when noisereduce is not installed."""
        from pythaitts.pretrained.archa_tts import ArchaTTS

        archa = self._make_archa()
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        with patch.dict('sys.modules', {'noisereduce': None}):
            result = archa._denoise(audio)

        np.testing.assert_array_almost_equal(result, audio)


if __name__ == '__main__':
    unittest.main()
