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
