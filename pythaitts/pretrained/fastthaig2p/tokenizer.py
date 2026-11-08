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

from typing import Iterable, Optional, Union
from pathlib import Path

from pythainlp.tokenize import Tokenizer as _PyThaiTokenizer

from ._data import data_path

DEFAULT_DICT_PATH = data_path("dict.txt")


class Tokenizer:
    def __init__(
        self,
        dict_path: Path | str = DEFAULT_DICT_PATH,
        custom_dict: Optional[Union[Iterable[str], dict, str, Path]] = None,
    ):
        dict_path = Path(dict_path)
        words = set(dict_path.read_text(encoding="utf-8").splitlines())
        words.discard("")
        self._base_dict = words
        self._custom_words: set[str] = set()
        self._dict = set(self._base_dict)
        if custom_dict:
            self.add_words(custom_dict)
        else:
            self._engine = _PyThaiTokenizer(custom_dict=self._dict, engine="newmm")

    @property
    def vocab(self) -> set[str]:
        return self._dict

    def add_words(self, words: Union[Iterable[str], dict, str, Path]) -> None:
        """Add custom words to the tokenizer vocabulary."""
        if isinstance(words, (str, Path)):
            p = Path(words)
            if p.exists():
                if p.suffix.lower() == ".json":
                    import json

                    loaded = json.loads(p.read_text(encoding="utf-8"))
                    words = loaded.keys() if isinstance(loaded, dict) else loaded
                else:
                    words = [
                        line.split("\t")[0].split(",")[0].split(":")[0].strip()
                        for line in p.read_text(encoding="utf-8").splitlines()
                        if line.strip() and not line.strip().startswith("#")
                    ]
            else:
                words = [str(words)]
        elif isinstance(words, dict):
            words = words.keys()

        for w in words:
            w_str = str(w).strip()
            if w_str:
                self._custom_words.add(w_str)
        self._dict = self._base_dict | self._custom_words
        self._engine = _PyThaiTokenizer(custom_dict=self._dict, engine="newmm")

    def remove_words(self, words: Union[Iterable[str], dict, str]) -> None:
        """Remove custom words from the tokenizer vocabulary."""
        if isinstance(words, str):
            words = [words]
        elif isinstance(words, dict):
            words = words.keys()

        for w in words:
            w_str = str(w).strip()
            self._custom_words.discard(w_str)
        self._dict = self._base_dict | self._custom_words
        self._engine = _PyThaiTokenizer(custom_dict=self._dict, engine="newmm")

    def tokenize(self, text: str) -> list[str]:
        return self._engine.word_tokenize(text)
