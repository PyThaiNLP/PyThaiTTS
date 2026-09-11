from __future__ import annotations

import json
from pathlib import Path

from ._data import data_path
from .fallback import fallback_g2p
from .normalizer import normalize
from .tokenizer import Tokenizer

DEFAULT_DICT_PATH = data_path("dict.txt")
DEFAULT_IPA_PATH = data_path("ipa.json")


class G2P:
    def __init__(
        self,
        dict_path: Path | str = DEFAULT_DICT_PATH,
        ipa_path: Path | str = DEFAULT_IPA_PATH,
    ):
        self._tokenizer = Tokenizer(dict_path)
        ipa_path = Path(ipa_path)
        self._ipa = json.loads(ipa_path.read_text(encoding="utf-8"))

    def convert(self, text: str) -> str:
        """Convert Thai text to IPA phonemes.

        Pipeline: normalize → tokenize → dict lookup → fallback G2P → join.
        """
        normalized = normalize(text)
        tokens = self._tokenizer.tokenize(normalized)
        phonemes = []
        for token in tokens:
            if token.strip() == "":
                continue
            ipa = self._ipa.get(token)
            if ipa:
                phonemes.append(ipa)
            else:
                phonemes.append(fallback_g2p(token))
        return " ".join(phonemes)
