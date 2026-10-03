# -*- coding: utf-8 -*-
"""
Unit tests for realtime TTS integration with FastThaiG2P
"""
import os
import unittest
from unittest.mock import Mock, patch
import numpy as np

from pythaitts import TTS, RealtimeTTS, FastThaiG2PEngine
from pythaitts.realtime import (
    chunk_text,
    stream_text_to_chunks,
    FastThaiG2PVoice,
    _AudioPlayer,
)


class TestRealtimeChunking(unittest.TestCase):
    """Test text chunking and streaming utilities for realtime synthesis."""

    def test_chunk_text_empty(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])
        self.assertEqual(chunk_text(None), [])

    def test_chunk_text_basic(self):
        chunks = chunk_text("สวัสดีครับ ยินดีต้อนรับ")
        self.assertTrue(len(chunks) >= 1)
        self.assertIn("สวัสดีครับ", chunks[0])

    def test_chunk_text_multiple_sentences(self):
        text = "สวัสดีครับ วันนี้อากาศดีมาก เราไปเที่ยวกันเถอะ"
        chunks = chunk_text(text)
        self.assertTrue(len(chunks) >= 2)

    def test_chunk_text_preprocessing(self):
        text = "มี 5 คนๆ"
        chunks_preprocessed = chunk_text(text, preprocess=True)
        self.assertTrue(len(chunks_preprocessed) > 0)
        combined = " ".join(chunks_preprocessed)
        self.assertIn("ห้า", combined)
        self.assertIn("คนคน", combined)
        self.assertNotIn("5", combined)
        self.assertNotIn("ๆ", combined)

        chunks_no_preprocess = chunk_text(text, preprocess=False)
        self.assertTrue(len(chunks_no_preprocess) > 0)
        combined_raw = " ".join(chunks_no_preprocess)
        self.assertIn("5", combined_raw)
        self.assertIn("ๆ", combined_raw)

    def test_chunk_text_long_sentence(self):
        # Long Thai sentence without spaces
        long_sentence = "นี่คือตัวอย่างของข้อความภาษาไทยที่เขียนติดต่อกันยาวมากโดยไม่มีการเว้นวรรคเลยแม้แต่น้อยเพื่อทดสอบการแบ่งส่วน" * 4
        chunks = chunk_text(long_sentence, max_phonemes=300)
        self.assertTrue(len(chunks) > 1)
        from pythaitts.pretrained.fastthaig2p import G2P, ipa_to_kokoro
        g2p = G2P()
        for c in chunks:
            phonemes = ipa_to_kokoro(g2p.convert(c))
            self.assertLessEqual(len(phonemes), 300)

    def test_stream_text_to_chunks_from_string(self):
        text = "สวัสดีครับ วันนี้อากาศดี"
        chunks = list(stream_text_to_chunks(text))
        self.assertTrue(len(chunks) >= 1)

    def test_stream_text_to_chunks_from_tokens(self):
        tokens = ["สวัสดี", "ครับ", " ", "ยินดี", "ต้อนรับ", "สู่", "พายไทย", "ทีทีเอส"]
        chunks = list(stream_text_to_chunks(tokens))
        self.assertTrue(len(chunks) >= 1)
        combined = "".join(chunks)
        self.assertIn("สวัสดี", combined)
        self.assertIn("พายไทย", combined)

    def test_stream_text_to_chunks_with_newlines(self):
        tokens = ["ข้อความที่หนึ่ง\n", "ข้อความที่สอง\n"]
        chunks = list(stream_text_to_chunks(tokens))
        self.assertTrue(len(chunks) >= 2)


class TestFastThaiG2PStream(unittest.TestCase):
    """Test FastThaiG2P streaming synthesis methods."""

    @patch('onnxruntime.InferenceSession')
    @patch('pythaitts.pretrained.fastthaig2p.tts._default_asset')
    def setUp(self, mock_asset, mock_session):
        import tempfile, json
        # Dummy assets
        self.temp_files = []
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
            json.dump({"vocab": {"a": 1, "b": 2}}, fp)
            self.dummy_config = fp.name
            self.temp_files.append(fp.name)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as fp:
            np.save(fp.name, np.zeros((510, 1, 256), dtype=np.float32))
            self.dummy_voice = fp.name
            self.temp_files.append(fp.name)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as fp:
            self.dummy_onnx = fp.name
            self.temp_files.append(fp.name)

        from pythaitts.pretrained.fastthaig2p import FastThaiG2P
        self.model = FastThaiG2P(
            model_path=self.dummy_onnx,
            voicepack_path=self.dummy_voice,
            config_path=self.dummy_config,
        )

    def tearDown(self):
        for f in self.temp_files:
            if os.path.exists(f):
                os.unlink(f)

    def test_stream_waveform_return(self):
        dummy_audio = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        self.model.generate = Mock(return_value=dummy_audio)

        chunks = list(self.model.stream("สวัสดีครับ วันนี้อากาศดี", return_type="waveform"))
        self.assertTrue(len(chunks) >= 1)
        for chunk in chunks:
            self.assertIsInstance(chunk, np.ndarray)
            self.assertTrue(np.array_equal(chunk, dummy_audio))

    def test_stream_bytes_return(self):
        dummy_audio = np.array([0.5, -0.5], dtype=np.float32)
        self.model.generate = Mock(return_value=dummy_audio)

        chunks = list(self.model.stream("สวัสดีครับ", return_type="bytes"))
        self.assertTrue(len(chunks) >= 1)
        for chunk in chunks:
            self.assertIsInstance(chunk, bytes)
            self.assertEqual(len(chunk), 4)  # 2 samples * 2 bytes

    def test_stream_file_return(self):
        dummy_audio = np.zeros(2400, dtype=np.float32)
        self.model.generate = Mock(return_value=dummy_audio)

        chunks = list(self.model.stream("สวัสดีครับ", return_type="file"))
        self.assertTrue(len(chunks) >= 1)
        for fpath in chunks:
            self.assertTrue(os.path.exists(fpath))
            self.assertTrue(fpath.endswith(".wav"))
            os.unlink(fpath)

    def test_stream_from_generator(self):
        dummy_audio = np.array([0.1], dtype=np.float32)
        self.model.generate = Mock(return_value=dummy_audio)

        def token_gen():
            yield "สวัสดี"
            yield "ครับ"

        chunks = list(self.model.stream(token_gen(), return_type="waveform"))
        self.assertTrue(len(chunks) >= 1)

    def test_stream_invalid_speaker(self):
        with self.assertRaises(ValueError):
            list(self.model.stream("สวัสดี", speaker_idx="unknown_voice"))

    def test_stream_speaker_mapping(self):
        self.model.generate = Mock(return_value=np.zeros(10, dtype=np.float32))
        list(self.model.stream("สวัสดี", speaker_idx="Linda"))
        list(self.model.stream("สวัสดี", speaker_idx=None))

    def test_stream_invalid_return_type(self):
        self.model.generate = Mock(return_value=np.zeros(10, dtype=np.float32))
        with self.assertRaises(ValueError):
            list(self.model.stream("สวัสดี", return_type="invalid_type"))


class TestTTSIntegration(unittest.TestCase):
    """Test TTS.stream integration in pythaitts/__init__.py"""

    @patch('pythaitts.pretrained.fastthaig2p.FastThaiG2P')
    def test_tts_stream_delegates_to_fastthaig2p(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_instance.stream.return_value = iter([np.zeros(10, dtype=np.float32)])
        mock_fastthaig2p_cls.return_value = mock_instance

        tts = TTS(pretrained="fastthaig2p")
        chunks = list(tts.stream("สวัสดีครับ"))
        mock_instance.stream.assert_called_once()
        self.assertEqual(len(chunks), 1)

    @patch('pythaitts.pretrained.fastthaig2p.FastThaiG2P')
    def test_tts_stream_alias(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_instance.stream.return_value = iter([np.zeros(10, dtype=np.float32)])
        mock_fastthaig2p_cls.return_value = mock_instance

        tts = TTS(pretrained="fastthaig2p")
        self.assertEqual(tts.stream, tts.tts_stream)
        chunks = list(tts.tts_stream("สวัสดีครับ"))
        mock_instance.stream.assert_called_once()


class TestRealtimeTTSClass(unittest.TestCase):
    """Test RealtimeTTS class."""

    @patch('pythaitts.TTS')
    def test_realtimetts_init_and_stream(self, mock_tts_cls):
        mock_tts_instance = Mock()
        mock_tts_instance.stream.return_value = iter([np.zeros(20, dtype=np.float32)])
        mock_tts_cls.return_value = mock_tts_instance

        rt = RealtimeTTS(pretrained="fastthaig2p", speed=1.2)
        chunks = list(rt.stream("สวัสดีครับ"))
        self.assertEqual(len(chunks), 1)
        mock_tts_instance.stream.assert_called_once()
        call_kwargs = mock_tts_instance.stream.call_args.kwargs
        self.assertEqual(call_kwargs['speed'], 1.2)
        self.assertEqual(call_kwargs['speaker_idx'], "thai_som")

    @patch('pythaitts.TTS')
    def test_realtimetts_synthesize(self, mock_tts_cls):
        mock_tts_instance = Mock()
        mock_tts_instance.tts.return_value = "output.wav"
        mock_tts_cls.return_value = mock_tts_instance

        rt = RealtimeTTS()
        out = rt.synthesize("สวัสดีครับ", filename="test.wav")
        self.assertEqual(out, "output.wav")
        mock_tts_instance.tts.assert_called_once_with("สวัสดีครับ", speaker_idx="thai_som", filename="test.wav")


class TestFastThaiG2PEngine(unittest.TestCase):
    """Test FastThaiG2PEngine for RealtimeTTS library compatibility."""

    @patch('pythaitts.pretrained.fastthaig2p.tts.FastThaiG2P')
    def test_engine_init_and_stream_info(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_instance.sample_rate = 24000
        mock_instance.SUPPORTED_VOICES = ["thai_som"]
        mock_fastthaig2p_cls.return_value = mock_instance

        engine = FastThaiG2PEngine()
        self.assertEqual(engine.engine_name, "fastthaig2p")
        fmt, channels, rate = engine.get_stream_info()
        self.assertEqual(channels, 1)
        self.assertEqual(rate, 24000)

    @patch('pythaitts.pretrained.fastthaig2p.tts.FastThaiG2P')
    def test_engine_voices(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_instance.sample_rate = 24000
        mock_instance.SUPPORTED_VOICES = ["thai_som"]
        mock_fastthaig2p_cls.return_value = mock_instance

        engine = FastThaiG2PEngine()
        voices = engine.get_voices()
        self.assertEqual(len(voices), 1)
        self.assertEqual(str(voices[0]), "thai_som")

        engine.set_voice("thai_som")
        self.assertEqual(engine.current_voice, "thai_som")

        with self.assertRaises(ValueError):
            engine.set_voice("unsupported_voice")

    @patch('pythaitts.pretrained.fastthaig2p.tts.FastThaiG2P')
    def test_engine_synthesize_puts_to_queue(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_instance.sample_rate = 24000
        mock_instance.SUPPORTED_VOICES = ["thai_som"]
        mock_instance._text_to_phonemes.return_value = "sa.wat.di"
        # 100 samples float32
        dummy_audio = np.ones(100, dtype=np.float32) * 0.5
        mock_instance.generate.return_value = dummy_audio
        mock_fastthaig2p_cls.return_value = mock_instance

        engine = FastThaiG2PEngine()
        success = engine.synthesize("สวัสดีครับ")
        self.assertTrue(success)
        self.assertFalse(engine.queue.empty())
        pcm_chunk = engine.queue.get()
        self.assertEqual(len(pcm_chunk), 200)  # 100 samples * 2 bytes

    @patch('pythaitts.pretrained.fastthaig2p.tts.FastThaiG2P')
    def test_engine_stop(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_instance.sample_rate = 24000
        mock_instance.SUPPORTED_VOICES = ["thai_som"]
        mock_fastthaig2p_cls.return_value = mock_instance

        engine = FastThaiG2PEngine()
        self.assertFalse(engine.stop_synthesis_event.is_set())
        engine.stop()
        self.assertTrue(engine.stop_synthesis_event.is_set())


if __name__ == '__main__':
    unittest.main()
