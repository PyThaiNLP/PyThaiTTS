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

from pathlib import Path

from pythainlp.tokenize import Tokenizer as _PyThaiTokenizer

from ._data import data_path

DEFAULT_DICT_PATH = data_path("dict.txt")


class Tokenizer:
    def __init__(self, dict_path: Path | str = DEFAULT_DICT_PATH):
        dict_path = Path(dict_path)
        words = set(dict_path.read_text(encoding="utf-8").splitlines())
        words.discard("")
        self._dict = words
        self._engine = _PyThaiTokenizer(custom_dict=words, engine="newmm")

    @property
    def vocab(self) -> set[str]:
        return self._dict

    def tokenize(self, text: str) -> list[str]:
        return self._engine.word_tokenize(text)
