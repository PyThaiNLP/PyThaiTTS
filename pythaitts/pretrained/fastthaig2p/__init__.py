# -*- coding: utf-8 -*-
from __future__ import annotations

from .g2p import G2P
from .kokoro import ipa_to_kokoro
from .normalizer import normalize
from .tokenizer import Tokenizer

__all__ = ["G2P", "Tokenizer", "normalize", "ipa_to_kokoro", "TTS", "FastThaiG2P"]


def __getattr__(name):
    if name in ("TTS", "FastThaiG2P"):
        from .tts import TTS, FastThaiG2P

        return FastThaiG2P if name == "FastThaiG2P" else TTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

