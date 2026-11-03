# -*- coding: utf-8 -*-
"""
Unit tests for Custom Word Dictionary support for Thai pronunciation
"""
import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from pythaitts import TTS, preprocess_text, load_custom_dict
from pythaitts.pretrained.fastthaig2p import G2P, FastThaiG2P, ipa_to_kokoro


class TestCustomWordDictLoading(unittest.TestCase):
    """Test load_custom_dict function with various input formats"""

    def test_load_from_dict(self):
        d = {"เพชรบุรี": "เพ็ด-ชะ-บุ-รี", "สถิติ": "สะ-ถิ-ติ"}
        loaded = load_custom_dict(d)
        self.assertEqual(loaded, d)

    def test_load_from_json_file(self):
        data = {"เพชรบุรี": "เพ็ด-ชะ-บุ-รี", "ChatGPT": "แชตจีพีที"}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            filename = f.name
        try:
            loaded = load_custom_dict(filename)
            self.assertEqual(loaded, data)
        finally:
            os.unlink(filename)

    def test_load_from_tsv_file(self):
        content = "# Comment line\nเพชรบุรี\tเพ็ด-ชะ-บุ-รี\nสถิติ\tสะ-ถิ-ติ\n\n"
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write(content)
            filename = f.name
        try:
            loaded = load_custom_dict(filename)
            self.assertEqual(loaded["เพชรบุรี"], "เพ็ด-ชะ-บุ-รี")
            self.assertEqual(loaded["สถิติ"], "สะ-ถิ-ติ")
            self.assertEqual(len(loaded), 2)
        finally:
            os.unlink(filename)

    def test_load_from_csv_file(self):
        content = "เพชรบุรี,เพ็ด-ชะ-บุ-รี\nสถิติ,สะ-ถิ-ติ\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(content)
            filename = f.name
        try:
            loaded = load_custom_dict(filename)
            self.assertEqual(loaded["เพชรบุรี"], "เพ็ด-ชะ-บุ-รี")
            self.assertEqual(loaded["สถิติ"], "สะ-ถิ-ติ")
        finally:
            os.unlink(filename)

    def test_load_from_colon_file(self):
        content = "เพชรบุรี: เพ็ด-ชะ-บุ-รี\nสถิติ: สะ-ถิ-ติ\n"
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            filename = f.name
        try:
            loaded = load_custom_dict(filename)
            self.assertEqual(loaded["เพชรบุรี"], "เพ็ด-ชะ-บุ-รี")
            self.assertEqual(loaded["สถิติ"], "สะ-ถิ-ติ")
        finally:
            os.unlink(filename)

    def test_load_from_tuples(self):
        tuples = [("เพชรบุรี", "เพ็ด-ชะ-บุ-รี"), ("สถิติ", "สะ-ถิ-ติ")]
        loaded = load_custom_dict(tuples)
        self.assertEqual(loaded["เพชรบุรี"], "เพ็ด-ชะ-บุ-รี")
        self.assertEqual(loaded["สถิติ"], "สะ-ถิ-ติ")

    def test_load_none(self):
        self.assertEqual(load_custom_dict(None), {})

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_custom_dict("/nonexistent/path/dict.json")

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            load_custom_dict(12345)


class TestPreprocessWithCustomDict(unittest.TestCase):
    """Test text preprocessing with custom word dictionary"""

    def test_preprocess_text_word_substitution(self):
        text = "ฉันจะไปเพชรบุรีพรุ่งนี้"
        custom_dict = {"เพชรบุรี": "เพ็ดชะบุรี"}
        result = preprocess_text(text, custom_dict=custom_dict)
        self.assertEqual(result, "ฉันจะไปเพ็ดชะบุรีพรุ่งนี้")

    def test_preprocess_text_with_numbers(self):
        text = "ไปเพชรบุรี 2 คน"
        custom_dict = {"เพชรบุรี": "เพ็ดชะบุรี"}
        result = preprocess_text(text, custom_dict=custom_dict)
        self.assertEqual(result, "ไปเพ็ดชะบุรี สอง คน")

    def test_preprocess_text_ignores_ipa(self):
        text = "ฉันจะไปเพชรบุรี"
        custom_dict = {"เพชรบุรี": "/pʰet̚˦˥.t͡ɕʰaʔ˦˥.bu˨˩.riː˧/"}
        # Preprocess text should not replace with IPA string into raw Thai text
        result = preprocess_text(text, custom_dict=custom_dict)
        self.assertEqual(result, "ฉันจะไปเพชรบุรี")


class TestG2PCustomDict(unittest.TestCase):
    """Test G2P phoneme conversion with custom word dictionary"""

    def test_g2p_custom_thai_pronunciation(self):
        g2p = G2P(custom_dict={"เพชรบุรี": "เพ็ด-ชะ-บุ-รี"})
        result = g2p.convert("ผมไปเพชรบุรี")
        # Ensure the syllables for เพ็ด, ชะ, บุ, รี are present in the IPA
        self.assertIn("pʰet", result)
        self.assertIn("t͡ɕʰa", result)
        self.assertIn("riː", result)

    def test_g2p_custom_ipa_pronunciation(self):
        custom_ipa = "/pʰet̚˨˩.t͡ɕʰa˦˥.bu˨˩.riː˧/"
        g2p = G2P(custom_dict={"เพชรบุรี": custom_ipa})
        result = g2p.convert("ผมไปเพชรบุรี")
        self.assertIn(custom_ipa, result)

    def test_g2p_temporary_custom_dict_in_convert(self):
        g2p = G2P()
        # Default conversion
        default_result = g2p.convert("เพชรบุรี")
        # With custom_dict in convert call
        custom_ipa = "/pʰet̚˨˩.t͡ɕʰa˦˥.bu˨˩.riː˧/"
        custom_result = g2p.convert("เพชรบุรี", custom_dict={"เพชรบุรี": custom_ipa})
        self.assertEqual(custom_result, custom_ipa)
        # Calling convert again without custom_dict returns default
        self.assertEqual(g2p.convert("เพชรบุรี"), default_result)

    def test_g2p_normalizer_protection_for_abbrev(self):
        # Without custom dict, "ก.พ." is expanded to "กุมภาพันธ์"
        g2p_default = G2P()
        default_res = g2p_default.convert("นาย ก.พ. ทำงาน")
        self.assertIn("kum", default_res)  # from กุมภาพันธ์

        # With custom dict, "ก.พ." is pronounced as "กอ-พอ"
        g2p_custom = G2P(custom_dict={"ก.พ.": "กอ-พอ"})
        custom_res = g2p_custom.convert("นาย ก.พ. ทำงาน")
        self.assertNotIn("kum", custom_res)
        self.assertIn("kɔː", custom_res)
        self.assertIn("pʰɔː", custom_res)

    def test_g2p_add_and_set_and_clear_custom_dict(self):
        g2p = G2P()
        custom_ipa = "/kʰam˧.pʰiː˧.seːt̚˨˩/"
        g2p.add_custom_dict({"คำพิเศษ": custom_ipa})
        self.assertIn(custom_ipa, g2p.convert("นี่คือคำพิเศษ"))

        # remove custom word
        g2p.remove_custom_word("คำพิเศษ")
        self.assertNotIn(custom_ipa, g2p.convert("นี่คือคำพิเศษ"))

        # set custom dict replaces everything
        g2p.set_custom_dict({"อีกคำ": "/ʔiːk̚˨˩.kʰam˧/"})
        self.assertIn("/ʔiːk̚˨˩.kʰam˧/", g2p.convert("อีกคำ"))

        # clear custom dict
        g2p.clear_custom_dict()
        self.assertEqual(len(g2p._custom_dict), 0)
        self.assertEqual(len(g2p._custom_ipa), 0)


class TestTTSCustomDictIntegration(unittest.TestCase):
    """Test high-level TTS class custom_dict integration"""

    @patch('pythaitts.pretrained.fastthaig2p.FastThaiG2P')
    def test_tts_init_with_custom_dict(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_fastthaig2p_cls.return_value = mock_instance

        custom_dict = {"เพชรบุรี": "เพ็ด-ชะ-บุ-รี"}
        tts = TTS(pretrained="fastthaig2p", custom_dict=custom_dict)
        mock_fastthaig2p_cls.assert_called_once()
        self.assertEqual(tts.custom_dict, custom_dict)

    @patch('pythaitts.pretrained.fastthaig2p.FastThaiG2P')
    def test_tts_call_with_custom_dict(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_fastthaig2p_cls.return_value = mock_instance

        tts = TTS(pretrained="fastthaig2p")
        custom_dict = {"เพชรบุรี": "เพ็ด-ชะ-บุ-รี"}
        tts.tts("ไปเพชรบุรี", custom_dict=custom_dict)

        mock_instance.assert_called_once()
        _, kwargs = mock_instance.call_args
        self.assertEqual(kwargs.get("custom_dict"), custom_dict)

    @patch('pythaitts.pretrained.fastthaig2p.FastThaiG2P')
    def test_tts_add_and_set_custom_dict(self, mock_fastthaig2p_cls):
        mock_instance = Mock()
        mock_fastthaig2p_cls.return_value = mock_instance

        tts = TTS(pretrained="fastthaig2p")
        tts.add_custom_dict({"คำหนึ่ง": "คำ-หนึ่ง"})
        mock_instance.add_custom_dict.assert_called_once()

        tts.set_custom_dict({"คำสอง": "คำ-สอง"})
        mock_instance.set_custom_dict.assert_called_once()


if __name__ == "__main__":
    unittest.main()

