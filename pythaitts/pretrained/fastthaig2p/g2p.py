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

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from pythaitts.preprocess import _is_ipa, load_custom_dict
from ._data import data_path
from .en_g2p import en_g2p
from .fallback import fallback_g2p
from .normalizer import normalize
from .tokenizer import Tokenizer

DEFAULT_DICT_PATH = data_path("dict.txt")
DEFAULT_IPA_PATH = data_path("ipa.json")

_EN_TOKEN_RE = re.compile(r"^[A-Za-z']+$")


class G2P:
    def __init__(
        self,
        dict_path: Path | str = DEFAULT_DICT_PATH,
        ipa_path: Path | str = DEFAULT_IPA_PATH,
        use_espeak: bool = False,
        custom_dict: Optional[Union[Dict[str, str], str, Path, Iterable[Any]]] = None,
    ):
        self._tokenizer = Tokenizer(dict_path)
        ipa_path = Path(ipa_path)
        self._ipa = json.loads(ipa_path.read_text(encoding="utf-8"))
        self.use_espeak = use_espeak
        self._custom_dict: Dict[str, str] = {}
        self._custom_ipa: Dict[str, str] = {}
        if custom_dict is not None:
            self.add_custom_dict(custom_dict)

    def add_custom_dict(
        self, custom_dict: Union[Dict[str, str], str, Path, Iterable[Any]]
    ) -> None:
        """Add custom word pronunciations to the G2P engine."""
        loaded = load_custom_dict(custom_dict)
        for word, pron in loaded.items():
            self._custom_dict[word] = pron
            self._custom_ipa[word] = self._pronunciation_to_ipa(pron)
        self._tokenizer.add_words(loaded.keys())

    def set_custom_dict(
        self, custom_dict: Union[Dict[str, str], str, Path, Iterable[Any]]
    ) -> None:
        """Replace all custom word pronunciations with a new dictionary."""
        self.clear_custom_dict()
        if custom_dict is not None:
            self.add_custom_dict(custom_dict)

    def remove_custom_word(self, word: str) -> None:
        """Remove a word from the custom dictionary."""
        word = str(word).strip()
        if word in self._custom_dict:
            del self._custom_dict[word]
        if word in self._custom_ipa:
            del self._custom_ipa[word]
        self._tokenizer.remove_words(word)

    def clear_custom_dict(self) -> None:
        """Clear all custom word pronunciations."""
        keys_to_remove = list(self._custom_dict.keys())
        self._custom_dict.clear()
        self._custom_ipa.clear()
        self._tokenizer.remove_words(keys_to_remove)

    def _pronunciation_to_ipa(self, pron: str) -> str:
        """Convert a custom pronunciation string (Thai text or IPA) to IPA phonemes."""
        pron = pron.strip()
        if not pron:
            return ""
        if _is_ipa(pron):
            return pron

        clean_pron = pron.replace("-", " ")
        normalized = normalize(clean_pron)
        tokens = self._tokenizer.tokenize(normalized)
        phonemes = []
        for token in tokens:
            if token.strip() == "":
                continue
            ipa = self._ipa.get(token)
            if ipa:
                phonemes.append(ipa)
            elif _EN_TOKEN_RE.match(token):
                phonemes.append(en_g2p(token, use_espeak=self.use_espeak))
            else:
                phonemes.append(fallback_g2p(token))
        return " ".join(phonemes)

    def convert(
        self,
        text: str,
        custom_dict: Optional[Union[Dict[str, str], str, Path, Iterable[Any]]] = None,
    ) -> str:
        """Convert Thai (and code-switched English) text to IPA phonemes.

        Pipeline: normalize → tokenize → dict lookup → English G2P / fallback G2P → join.
        Supports custom_dict for custom Thai pronunciation in words.
        """
        temp_keys = []
        if custom_dict is not None:
            temp_loaded = load_custom_dict(custom_dict)
            effective_custom_dict = {**self._custom_dict, **temp_loaded}
            effective_custom_ipa = {
                **self._custom_ipa,
                **{k: self._pronunciation_to_ipa(v) for k, v in temp_loaded.items()},
            }
            temp_keys = [k for k in temp_loaded.keys() if k not in self._custom_dict]
            if temp_keys:
                self._tokenizer.add_words(temp_keys)
        else:
            effective_custom_dict = self._custom_dict
            effective_custom_ipa = self._custom_ipa

        try:
            normalized = normalize(text, custom_dict=effective_custom_dict)
            tokens = self._tokenizer.tokenize(normalized)
            phonemes = []
            for token in tokens:
                if token.strip() == "":
                    continue
                if token in effective_custom_ipa:
                    phonemes.append(effective_custom_ipa[token])
                else:
                    ipa = self._ipa.get(token)
                    if ipa:
                        phonemes.append(ipa)
                    elif _EN_TOKEN_RE.match(token):
                        phonemes.append(en_g2p(token, use_espeak=self.use_espeak))
                    else:
                        phonemes.append(fallback_g2p(token))
            return " ".join(phonemes)
        finally:
            if temp_keys:
                self._tokenizer.remove_words(temp_keys)
