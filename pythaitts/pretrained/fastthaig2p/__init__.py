# -*- coding: utf-8 -*-
"""
## License

Copyright 2026 Charin Polpanumas and Amazon Web Services

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
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

