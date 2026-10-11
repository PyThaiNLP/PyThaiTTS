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
    ):
        self._tokenizer = Tokenizer(dict_path)
        ipa_path = Path(ipa_path)
        self._ipa = json.loads(ipa_path.read_text(encoding="utf-8"))
        self.use_espeak = use_espeak

    def convert(self, text: str) -> str:
        """Convert Thai (and code-switched English) text to IPA phonemes.

        Pipeline: normalize → tokenize → dict lookup → English G2P / fallback G2P → join.
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
            elif _EN_TOKEN_RE.match(token):
                phonemes.append(en_g2p(token, use_espeak=self.use_espeak))
            else:
                phonemes.append(fallback_g2p(token))
        return " ".join(phonemes)
