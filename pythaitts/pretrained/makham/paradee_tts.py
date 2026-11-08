# -*- coding: utf-8 -*-
"""
MakHamTTS Paradee model
"""
from typing import Optional, Union, Iterable
from pathlib import Path
import os
import json
import wave
import numpy as np
import onnxruntime as ort

from pythaitts.pretrained.fastthaig2p.g2p import G2P
from pythaitts.pretrained.fastthaig2p.kokoro import ipa_to_kokoro

_PARADEE_DIR = Path(__file__).parent / "data"
_DEFAULT_MODEL_PATH = _PARADEE_DIR / "paradee_thai_int8.onnx"
_DEFAULT_CONFIG_PATH = _PARADEE_DIR / "config.json"


class MakHamTTSParadee:
    """MakHamTTS Paradee: Standalone, single-voice Thai ONNX TTS model (female voice).

    This model runs locally using the included quantized INT8 ONNX file without download.
    """

    MAX_PHONEMES = 510

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        speed: float = 1.0,
        intra_op_threads: int = 0,
        custom_dict: Optional[Union[dict, str, Path, Iterable]] = None,
        **kwargs,
    ):
        model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

        if not model_path.exists():
            raise FileNotFoundError(f"Paradee ONNX model not found: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Paradee config not found: {config_path}")

        options = ort.SessionOptions()
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )

        with open(config_path, encoding="utf-8") as f:
            self._vocab = json.load(f)["vocab"]

        self._g2p = G2P(custom_dict=custom_dict)
        self.sample_rate = 24000
        self.speed = speed

    def add_custom_dict(
        self, custom_dict: Union[dict, str, Path, Iterable]
    ) -> None:
        """Add custom word pronunciations to the underlying G2P engine."""
        self._g2p.add_custom_dict(custom_dict)

    def set_custom_dict(
        self, custom_dict: Union[dict, str, Path, Iterable]
    ) -> None:
        """Replace all custom word pronunciations in the underlying G2P engine."""
        self._g2p.set_custom_dict(custom_dict)

    def _text_to_phonemes(
        self,
        text: str,
        custom_dict: Optional[Union[dict, str, Path, Iterable]] = None,
    ) -> str:
        return ipa_to_kokoro(self._g2p.convert(text, custom_dict=custom_dict))

    def generate(
        self,
        text: str,
        custom_dict: Optional[Union[dict, str, Path, Iterable]] = None,
    ) -> np.ndarray:
        """Convert Thai text to float32 audio array (24 kHz mono)."""
        phonemes = self._text_to_phonemes(text, custom_dict=custom_dict)
        if not phonemes:
            return np.array([], dtype=np.float32)
        if len(phonemes) > self.MAX_PHONEMES:
            raise ValueError(
                f"Text produces {len(phonemes)} phonemes; max {self.MAX_PHONEMES}. "
                "Split the input into shorter chunks."
            )

        ids = [0] + [self._vocab[c] for c in phonemes if c in self._vocab] + [0]
        audio = self._session.run(
            None,
            {
                "input_ids": np.array([ids], dtype=np.int64),
                "speed": np.array([self.speed], dtype=np.float32),
            },
        )[0][0]
        return audio

    def synthesize(
        self,
        text: str,
        output_path: Union[str, Path],
        custom_dict: Optional[Union[dict, str, Path, Iterable]] = None,
    ) -> Path:
        """Convert Thai text to WAV file."""
        output_path = Path(output_path)
        audio = self.generate(text, custom_dict=custom_dict)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return output_path

    def __call__(
        self,
        text: str,
        speaker_idx: Optional[str] = None,
        return_type: str = "file",
        filename: Optional[str] = None,
        custom_dict: Optional[Union[dict, str, Path, Iterable]] = None,
        **kwargs,
    ):
        """Generate speech from text using MakHamTTS Paradee.

        :param str text: Input text to synthesize
        :param str speaker_idx: Unused (single voice)
        :param str return_type: Return type ('file' or 'waveform')
        :param str filename: Output filename for the generated audio
        :param custom_dict: Optional custom word dictionary
        :param kwargs: Additional parameters (e.g. speed)
        :return: File path if return_type is 'file', otherwise audio waveform (np.ndarray)
        """
        if "speed" in kwargs:
            self.speed = kwargs["speed"]

        if return_type == "waveform":
            return self.generate(text, custom_dict=custom_dict)
        else:
            import tempfile

            if filename is None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                    filename = fp.name

            self.synthesize(text, filename, custom_dict=custom_dict)
            return str(filename)

    def stream(
        self,
        text: Union[str, Iterable[str]],
        speaker_idx: Optional[str] = None,
        return_type: str = "waveform",
        play: bool = False,
        preprocess: bool = True,
        max_phonemes: int = 400,
        custom_dict: Optional[Union[dict, str, Path, Iterable]] = None,
        **kwargs,
    ):
        """Stream Thai speech generation chunk by chunk in real-time."""
        orig_speed = self.speed
        if "speed" in kwargs:
            self.speed = kwargs["speed"]

        from pythaitts.realtime import _AudioPlayer, stream_text_to_chunks

        player = _AudioPlayer(sample_rate=self.sample_rate) if play else None

        try:
            chunks = stream_text_to_chunks(
                text,
                max_phonemes=max_phonemes,
                preprocess=preprocess,
                g2p_converter=lambda t: self._text_to_phonemes(t, custom_dict=custom_dict),
            )
            for chunk in chunks:
                audio = self.generate(chunk, custom_dict=custom_dict)
                if len(audio) == 0:
                    continue

                if player is not None:
                    player.play(audio)

                if return_type == "waveform":
                    yield audio
                elif return_type in ("bytes", "raw"):
                    yield (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                elif return_type == "file":
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                        temp_name = fp.name
                    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                    with wave.open(temp_name, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(audio_int16.tobytes())
                    yield temp_name
                else:
                    raise ValueError(
                        f"Unsupported return_type '{return_type}'. Choose 'waveform', 'bytes', or 'file'."
                    )
        finally:
            self.speed = orig_speed
            if player is not None:
                player.close()

    tts_stream = stream
