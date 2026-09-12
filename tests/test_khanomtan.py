# -*- coding: utf-8 -*-
"""
Unit tests for KhanomTan TTS integration
"""
import sys
import unittest
from unittest.mock import patch, MagicMock


class TestKhanomTanImportError(unittest.TestCase):
    """Test that a helpful ImportError is raised when coqui-tts is not installed"""

    def test_import_error_when_tts_not_installed(self):
        """Test that ImportError with helpful message is raised when TTS package is missing"""
        with patch.dict(sys.modules, {"TTS": None, "TTS.utils": None, "TTS.utils.synthesizer": None}):
            # Remove cached module if present
            for key in list(sys.modules.keys()):
                if key.startswith("pythaitts.pretrained.khanomtan"):
                    del sys.modules[key]

            from pythaitts.pretrained.khanomtan_tts import KhanomTan

            instance = KhanomTan.__new__(KhanomTan)
            instance.version = "1.0"
            instance.best_model_path_name = "best_model.pth"
            instance.last_checkpoint_model_path_name = "checkpoint_440000.pth"
            instance.config_path = "config.json"
            instance.speakers_path = "speakers.pth"
            instance.languages_path = "language_ids.json"
            instance.speaker_encoder_model_path = "model_se.pth"
            instance.speaker_encoder_config_path = "config_se.json"
            instance.synthesizer = None

            with self.assertRaises(ImportError) as ctx:
                instance.load_synthesizer("last_checkpoint")

            self.assertIn("coqui-tts", str(ctx.exception))
            self.assertIn("pip install", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
