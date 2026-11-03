# -*- coding: utf-8 -*-
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pythaitts import TTS


class TestMakHamTTSIntegration(unittest.TestCase):
    @patch("pythaitts.pretrained.makham_tts.MakHamTTS")
    def test_makhamtts_model_initialization(self, mock_makham_cls):
        mock_makham_cls.return_value = Mock()
        tts = TTS(pretrained="makhamtts")
        self.assertIsNotNone(tts.model)
        self.assertEqual(tts.pretrained, "makhamtts")

    @patch("pythaitts.pretrained.makham_tts.MakHamTTS")
    def test_makhamtts_tts_call(self, mock_makham_cls):
        mock_instance = Mock()
        mock_instance.return_value = "/tmp/output.wav"
        mock_makham_cls.return_value = mock_instance

        tts = TTS(pretrained="makhamtts")
        result = tts.tts("สวัสดีค่ะ", speaker_idx="paradee_thai", filename="/tmp/test.wav")

        self.assertEqual(result, "/tmp/output.wav")
        call_args = mock_instance.call_args
        self.assertEqual(call_args.kwargs["text"], "สวัสดีค่ะ")
        self.assertEqual(call_args.kwargs["speaker_idx"], "paradee_thai")
        self.assertEqual(call_args.kwargs["filename"], "/tmp/test.wav")
        self.assertEqual(call_args.kwargs["return_type"], "file")

    @patch("pythaitts.pretrained.makham_tts.ipa_to_kokoro")
    @patch("pythaitts.pretrained.makham_tts.G2P")
    @patch("pythaitts.pretrained.makham_tts.ort.InferenceSession")
    def test_makhamtts_waveform_return(
        self, mock_session_cls, mock_g2p_cls, mock_ipa_to_kokoro
    ):
        from pythaitts.pretrained.makham_tts import MakHamTTS

        mock_session = Mock()
        mock_session.run.return_value = [np.array([[0.1, 0.2]], dtype=np.float32)]
        mock_session_cls.return_value = mock_session
        mock_g2p = Mock()
        mock_g2p.convert.return_value = "/dummy/"
        mock_g2p_cls.return_value = mock_g2p
        mock_ipa_to_kokoro.return_value = "abc"

        model = MakHamTTS()
        wav = model("สวัสดีค่ะ", return_type="waveform")

        self.assertTrue(np.array_equal(wav, np.array([0.1, 0.2], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
