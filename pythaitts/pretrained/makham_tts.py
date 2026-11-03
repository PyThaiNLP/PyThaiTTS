# -*- coding: utf-8 -*-
import json
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from pythaitts.pretrained.fastthaig2p import G2P, ipa_to_kokoro


_ASSET_DIR = Path(__file__).resolve().parent / "makhamtts"
_DEFAULT_MODEL_PATH = _ASSET_DIR / "paradee_thai_int8.onnx"
_DEFAULT_CONFIG_PATH = _ASSET_DIR / "config.json"


class MakHamTTS:
    SUPPORTED_VOICES = ["paradee_thai"]

    def __init__(self, model_path=None, config_path=None) -> None:
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self.config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

        if not self.model_path.exists():
            raise FileNotFoundError(f"MakHamTTS model file not found: {self.model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"MakHamTTS config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.vocab = cfg["vocab"]
        self.sample_rate = cfg.get("sample_rate", 24000)
        self.g2p = G2P()
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )

    def tts(self, text: str, speed: float = 1.0):
        phonemes = ipa_to_kokoro(self.g2p.convert(text))
        ids = [0] + [self.vocab[c] for c in phonemes if c in self.vocab] + [0]
        if len(ids) <= 2:
            return np.array([], dtype=np.float32)
        audio = self.session.run(
            None,
            {
                "input_ids": np.array([ids], dtype=np.int64),
                "speed": np.array([float(speed)], dtype=np.float32),
            },
        )[0][0]
        return audio

    def __call__(
        self,
        text: str,
        speaker_idx: str = "paradee_thai",
        return_type: str = "file",
        filename: str = None,
        **kwargs,
    ):
        if speaker_idx in ("thai_som", "Linda", None):
            speaker_idx = "paradee_thai"
        if speaker_idx not in self.SUPPORTED_VOICES:
            raise ValueError(
                f"Unsupported voice '{speaker_idx}'. Supported voices are: {', '.join(self.SUPPORTED_VOICES)}"
            )

        wav = self.tts(text, speed=kwargs.get("speed", 1.0))

        if return_type == "waveform":
            return wav
        if filename is not None:
            sf.write(filename, wav, self.sample_rate)
            return filename
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            sf.write(fp.name, wav, self.sample_rate)
            return fp.name
