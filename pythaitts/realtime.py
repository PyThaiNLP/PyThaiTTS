# -*- coding: utf-8 -*-
"""
Real-time Text-to-Speech (TTS) support for PyThaiTTS.

Provides:
- RealtimeTTS: High-level streaming TTS class.
- FastThaiG2PEngine: Engine for integration with the RealtimeTTS library (KoljaB/RealtimeTTS).
- chunk_text / stream_text_to_chunks: Text chunking utilities for streaming Thai synthesis.
- _AudioPlayer: Live speaker playback for audio chunks.
"""

from __future__ import annotations

import os
import queue
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Union

import numpy as np

# Try importing BaseEngine from RealtimeTTS library
try:
    from RealtimeTTS.engines.base_engine import BaseEngine, TimingInfo
    _HAS_REALTIMETTS = True
except ImportError:
    _HAS_REALTIMETTS = False

    class TimingInfo:
        def __init__(self, start_time: float, end_time: float, word: str):
            self.start_time = start_time
            self.end_time = end_time
            self.word = word

        def __str__(self):
            return f"Word: {self.word}, Start Time: {self.start_time}, End Time: {self.end_time}"

    class BaseEngine:
        """Fallback BaseEngine compatible with RealtimeTTS's BaseEngine."""

        def __init__(self):
            import multiprocessing as mp

            self.engine_name = "unknown"
            self.can_consume_generators = False
            self.preload_sentence_tokenizer = False
            self.queue = queue.Queue()
            self.timings = queue.Queue()
            self.provides_word_timings = False
            self.on_audio_chunk = None
            self.on_playback_start = None
            self.stop_synthesis_event = mp.Event()
            self._trim_silence_start_pending = None
            self.audio_duration = 0

        def post_init(self):
            pass

        def reset_audio_duration(self):
            self.audio_duration = 0

        def verify_sample_rate(self, sample_rate: int) -> int:
            return sample_rate if sample_rate > 0 else 24000

        def get_stream_info(self):
            raise NotImplementedError

        def synthesize(self, text: str, sentence_count: int = 0) -> bool:
            self.stop_synthesis_event.clear()
            self._trim_silence_start_pending = True
            return True

        def get_voices(self):
            raise NotImplementedError

        def set_voice(self, voice: Union[str, object]):
            raise NotImplementedError

        def set_voice_parameters(self, **voice_parameters):
            pass

        def stop(self):
            self.stop_synthesis_event.set()

        def shutdown(self):
            self.stop()


class FastThaiG2PVoice:
    """Voice descriptor for FastThaiG2P in RealtimeTTS."""

    def __init__(self, name: str = "thai_som"):
        self.name = name

    def __repr__(self):
        return f"FastThaiG2PVoice(name={self.name})"

    def __str__(self):
        return self.name


def chunk_text(
    text: str,
    max_phonemes: int = 400,
    preprocess: bool = True,
    g2p_converter: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """Split Thai text into chunks suitable for FastThaiG2P synthesis.

    Ensures each chunk does not exceed max_phonemes to prevent Kokoro context
    overflow (limit 510).

    :param str text: Thai text to split
    :param int max_phonemes: Maximum phonemes per chunk (default: 400)
    :param bool preprocess: Whether to preprocess text (numbers to words, ๆ)
    :param Callable g2p_converter: Optional custom function converting text to phonemes
    :return: List of text chunks
    """
    if not text or not str(text).strip():
        return []

    if preprocess:
        from pythaitts.preprocess import preprocess_text

        text = preprocess_text(str(text))

    try:
        from pythainlp.tokenize import sent_tokenize

        sentences = sent_tokenize(text, engine="crfcut")
    except Exception:
        sentences = [s.strip() for s in text.splitlines() if s.strip()]
        if not sentences:
            sentences = [text]

    if g2p_converter is None:
        from pythaitts.pretrained.fastthaig2p import G2P, ipa_to_kokoro

        _g2p = G2P()

        def _to_phonemes(t: str) -> str:
            return ipa_to_kokoro(_g2p.convert(t))

        g2p_converter = _to_phonemes

    chunks: List[str] = []
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        try:
            phonemes = g2p_converter(s)
            p_len = len(phonemes)
        except Exception:
            p_len = len(s) * 2  # conservative estimate

        if p_len <= max_phonemes:
            chunks.append(s)
        else:
            # Sub-divide by words using PyThaiNLP
            try:
                from pythainlp.tokenize import word_tokenize

                words = word_tokenize(s, engine="newmm")
            except Exception:
                words = s.split(" ")

            cur_words: List[str] = []
            for w in words:
                cur_words.append(w)
                candidate = "".join(cur_words)
                try:
                    cand_phonemes = g2p_converter(candidate)
                    cand_len = len(cand_phonemes)
                except Exception:
                    cand_len = len(candidate) * 2

                if cand_len > max_phonemes:
                    if len(cur_words) > 1:
                        chunks.append("".join(cur_words[:-1]))
                        cur_words = [w]
                    else:
                        chunks.append(w)
                        cur_words = []
            if cur_words:
                chunks.append("".join(cur_words))

    return chunks


def stream_text_to_chunks(
    text_or_stream: Union[str, Iterable[str]],
    max_phonemes: int = 400,
    max_buffer_chars: int = 120,
    preprocess: bool = True,
    g2p_converter: Optional[Callable[[str], str]] = None,
) -> Iterator[str]:
    """Yield text chunks from either a full string or a streaming iterable of text tokens.

    Buffers incoming tokens from LLM streams and emits complete sentences/clauses
    as soon as they are ready.

    :param Union[str, Iterable[str]] text_or_stream: String or generator of text tokens
    :param int max_phonemes: Maximum phonemes per chunk
    :param int max_buffer_chars: Maximum characters to buffer before breaking on word boundary
    :param bool preprocess: Whether to preprocess text
    :param Callable g2p_converter: Optional phoneme conversion function
    :yield: Text chunks ready for synthesis
    """
    if isinstance(text_or_stream, str):
        for c in chunk_text(
            text_or_stream,
            max_phonemes=max_phonemes,
            preprocess=preprocess,
            g2p_converter=g2p_converter,
        ):
            yield c
        return

    buffer = ""
    try:
        from pythainlp.tokenize import sent_tokenize, word_tokenize
    except ImportError:
        sent_tokenize = None
        word_tokenize = None

    for token in text_or_stream:
        if not token:
            continue
        buffer += str(token)

        # Handle explicit newlines as immediate sentence breaks
        if "\n" in buffer:
            parts = buffer.split("\n")
            for part in parts[:-1]:
                if part.strip():
                    for c in chunk_text(
                        part,
                        max_phonemes=max_phonemes,
                        preprocess=preprocess,
                        g2p_converter=g2p_converter,
                    ):
                        yield c
            buffer = parts[-1]

        # Check for sentence boundaries
        sents = []
        if sent_tokenize is not None and len(buffer) > 20:
            try:
                sents = sent_tokenize(buffer, engine="crfcut")
            except Exception:
                sents = [buffer]

        if len(sents) > 1:
            for s in sents[:-1]:
                if s.strip():
                    for c in chunk_text(
                        s,
                        max_phonemes=max_phonemes,
                        preprocess=preprocess,
                        g2p_converter=g2p_converter,
                    ):
                        yield c
            buffer = sents[-1]
        elif len(buffer) >= max_buffer_chars:
            # Buffer is long, break at word boundary
            if word_tokenize is not None:
                try:
                    words = word_tokenize(buffer, engine="newmm")
                except Exception:
                    words = buffer.split(" ")
            else:
                words = buffer.split(" ")

            if len(words) > 1:
                cutoff = max(1, int(len(words) * 0.75))
                to_emit = "".join(words[:cutoff])
                if to_emit.strip():
                    for c in chunk_text(
                        to_emit,
                        max_phonemes=max_phonemes,
                        preprocess=preprocess,
                        g2p_converter=g2p_converter,
                    ):
                        yield c
                buffer = "".join(words[cutoff:])

    # Flush remainder of buffer
    if buffer.strip():
        for c in chunk_text(
            buffer,
            max_phonemes=max_phonemes,
            preprocess=preprocess,
            g2p_converter=g2p_converter,
        ):
            yield c


class _AudioPlayer:
    """Helper for streaming audio playback through speaker devices."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.pa_instance = None
        self.pa_stream = None
        self.sd = None

        # Try pyaudio first (allows continuous stream writes)
        try:
            import pyaudio

            self.pa_instance = pyaudio.PyAudio()
            self.pa_stream = self.pa_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
            )
            return
        except Exception:
            pass

        # Fallback to sounddevice
        try:
            import sounddevice as sd

            self.sd = sd
            return
        except ImportError:
            pass

        raise ImportError(
            "Live audio playback requires 'pyaudio' or 'sounddevice'. "
            "Please install one via: pip install pyaudio or pip install sounddevice"
        )

    def play(self, audio: np.ndarray):
        if self.pa_stream is not None:
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            self.pa_stream.write(pcm)
        elif self.sd is not None:
            self.sd.play(audio, samplerate=self.sample_rate)
            self.sd.wait()

    def close(self):
        if self.pa_stream is not None:
            try:
                self.pa_stream.stop_stream()
                self.pa_stream.close()
            except Exception:
                pass
            self.pa_stream = None
        if self.pa_instance is not None:
            try:
                self.pa_instance.terminate()
            except Exception:
                pass
            self.pa_instance = None


class FastThaiG2PEngine(BaseEngine):
    """RealtimeTTS engine implementation using FastThaiG2P + Kokoro-82M.

    Seamlessly integrates with KoljaB/RealtimeTTS:
    ```python
    from RealtimeTTS import TextToAudioStream
    from pythaitts.realtime import FastThaiG2PEngine

    engine = FastThaiG2PEngine()
    stream = TextToAudioStream(engine)
    stream.feed("สวัสดีครับ วันนี้อากาศดีมาก")
    stream.play()
    ```
    """

    SUPPORTED_VOICES = ["thai_som"]

    def __init__(
        self,
        voice: Union[str, FastThaiG2PVoice] = "thai_som",
        speed: float = 1.0,
        model_path: Optional[str | Path] = None,
        voicepack_path: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        preprocess: bool = True,
        max_phonemes: int = 400,
        debug: bool = False,
    ):
        super().__init__()
        self.engine_name = "fastthaig2p"
        self.debug = debug
        self.preprocess = preprocess
        self.max_phonemes = max_phonemes
        self.speed = speed

        from pythaitts.pretrained.fastthaig2p.tts import FastThaiG2P

        self.model = FastThaiG2P(
            model_path=model_path,
            voicepack_path=voicepack_path,
            config_path=config_path,
            device=device,
            speed=speed,
        )
        self.sample_rate = self.model.sample_rate
        self.set_voice(voice)

    def post_init(self):
        self.engine_name = "fastthaig2p"

    def get_stream_info(self):
        """Returns audio stream format, channels, and sample rate."""
        try:
            import pyaudio

            pa_int16 = pyaudio.paInt16
        except ImportError:
            pa_int16 = 1  # Standard paInt16 enum value

        return pa_int16, 1, self.sample_rate

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        """Synthesizes text and pushes 16-bit PCM chunks into self.queue."""
        if hasattr(super(), "synthesize"):
            super().synthesize(text, sentence_count)
        self.stop_synthesis_event.clear()

        if not text or not str(text).strip():
            return True

        chunks = chunk_text(
            str(text),
            max_phonemes=self.max_phonemes,
            preprocess=self.preprocess,
            g2p_converter=self.model._text_to_phonemes,
        )

        for chunk in chunks:
            if self.stop_synthesis_event.is_set():
                if self.debug:
                    print("[FastThaiG2PEngine] Synthesis stopped by event.")
                return False

            try:
                audio = self.model.generate(chunk)
                if len(audio) == 0:
                    continue

                pcm_bytes = (
                    (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                )
                self.queue.put(pcm_bytes)
                self.audio_duration += len(audio) / float(self.sample_rate)

                if self.on_audio_chunk is not None:
                    try:
                        self.on_audio_chunk(pcm_bytes)
                    except Exception:
                        pass
            except Exception as e:
                if self.debug:
                    print(f"[FastThaiG2PEngine] Error synthesizing chunk: {e}")
                return False

        return True

    def get_voices(self) -> List[FastThaiG2PVoice]:
        """Returns list of supported voices."""
        return [FastThaiG2PVoice(v) for v in self.SUPPORTED_VOICES]

    def set_voice(self, voice: Union[str, FastThaiG2PVoice]):
        """Sets the active voice."""
        voice_str = voice.name if isinstance(voice, FastThaiG2PVoice) else str(voice)
        if voice_str in ("Linda", None, ""):
            voice_str = "thai_som"

        if voice_str not in self.SUPPORTED_VOICES and not os.path.exists(voice_str):
            raise ValueError(
                f"Unsupported voice '{voice_str}'. Supported voices are: {', '.join(self.SUPPORTED_VOICES)}"
            )
        self.current_voice = voice_str

    def set_voice_parameters(self, **voice_parameters):
        """Sets optional voice parameters (e.g. speed)."""
        if "speed" in voice_parameters:
            self.speed = float(voice_parameters["speed"])
            self.model.speed = self.speed

    def stop(self):
        """Stops ongoing synthesis."""
        self.stop_synthesis_event.set()

    def shutdown(self):
        """Shuts down the engine."""
        self.stop()


# Aliases for convenience
PyThaiTTSEngine = FastThaiG2PEngine
RealtimeTTSEngine = FastThaiG2PEngine


class RealtimeTTS:
    """High-level Real-time Text-to-Speech manager for PyThaiTTS.

    Supports streaming generation from text or token streams with low latency.
    """

    def __init__(
        self,
        pretrained: str = "fastthaig2p",
        speaker_idx: str = "thai_som",
        speed: float = 1.0,
        device: str = "cpu",
        **kwargs,
    ):
        from pythaitts import TTS

        self.tts = TTS(pretrained=pretrained, device=device, **kwargs)
        self.speaker_idx = speaker_idx
        self.speed = speed

    def stream(
        self,
        text: Union[str, Iterable[str]],
        speaker_idx: Optional[str] = None,
        return_type: str = "waveform",
        play: bool = False,
        preprocess: bool = True,
        max_phonemes: int = 400,
        **kwargs,
    ) -> Iterator[np.ndarray | bytes | str]:
        """Stream speech synthesis in real-time.

        :param Union[str, Iterable[str]] text: Input text or token generator
        :param str speaker_idx: Voice to use (default: initialized voice)
        :param str return_type: Return format ("waveform", "bytes", "raw", "file")
        :param bool play: Whether to play audio chunks in real-time
        :param bool preprocess: Whether to preprocess text
        :param int max_phonemes: Maximum phonemes per synthesized chunk
        :param kwargs: Additional parameters passed to the model
        :yield: Audio chunk (numpy array, PCM bytes, or wav file path)
        """
        voice = speaker_idx or self.speaker_idx
        speed = kwargs.get("speed", self.speed)
        return self.tts.stream(
            text=text,
            speaker_idx=voice,
            return_type=return_type,
            play=play,
            preprocess=preprocess,
            max_phonemes=max_phonemes,
            speed=speed,
            **kwargs,
        )

    def synthesize(self, text: str, **kwargs):
        """Standard batch synthesis."""
        voice = kwargs.pop("speaker_idx", self.speaker_idx)
        return self.tts.tts(text, speaker_idx=voice, **kwargs)
