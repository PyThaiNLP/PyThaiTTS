"""Rule-based fallback G2P for OOV words.

Uses a vendored version of tltk's th2ipa module (pure stdlib Python)
with data files stored in data/. Zero external dependencies.

Converts tltk's romanized phoneme output to our Wiktionary IPA convention.
"""

from __future__ import annotations

import re
from functools import lru_cache

from . import _th2ipa

_TLTK_PHON_RE = re.compile(r"<tr/>(.+?)\|<s/>")

_TONE_MAP = {
    "0": "˧",    # mid
    "1": "˨˩",   # low
    "2": "˥˩",   # falling
    "3": "˦˥",   # high
    "4": "˩˩˦",  # rising
}

_CONSONANT_MAP = [
    ("kh", "kʰ"), ("ph", "pʰ"), ("th", "tʰ"), ("ch", "t͡ɕʰ"),
    ("c", "t͡ɕ"), ("N", "ŋ"), ("?", "ʔ"),
]

_VOWEL_MAP = [
    ("UUa", "ɯa̯"), ("Ua", "ɯa̯"), ("iia", "ia̯"), ("ia", "ia̯"),
    ("uua", "ua̯"), ("ua", "ua̯"),
    ("aa", "aː"), ("ii", "iː"), ("uu", "uː"), ("xx", "ɛː"),
    ("ee", "eː"), ("oo", "oː"), ("OO", "ɔː"), ("@@", "ɤː"), ("UU", "ɯː"),
    ("a", "a"), ("i", "i"), ("u", "u"), ("x", "ɛ"), ("e", "e"),
    ("o", "o"), ("O", "ɔ"), ("@", "ɤ"), ("U", "ɯ"),
]


def _tltk_syllable_to_ipa(syl: str) -> str:
    """Convert a single tltk syllable like 'khiian4' to IPA."""
    tone_digit = syl[-1] if syl and syl[-1] in "01234" else "0"
    if syl and syl[-1] in "01234":
        syl = syl[:-1]
    tone = _TONE_MAP[tone_digit]

    result = syl
    for old, new in _CONSONANT_MAP:
        result = result.replace(old, new)
    for old, new in _VOWEL_MAP:
        result = result.replace(old, new)

    # Add unreleased mark to final stops
    if result and result[-1] in "ktp":
        vowel_chars = set("aeiouɔɛɯɤː̯")
        if any(c in vowel_chars for c in result[:-1]):
            result = result[:-1] + result[-1] + "̚"

    return result + tone


def fallback_g2p(word: str) -> str:
    """Generate IPA for a word using rule-based G2P fallback.

    Returns IPA in Wiktionary convention: /syl.la.ble˧/
    """
    result = _th2ipa.g2p(word)
    match = _TLTK_PHON_RE.search(result)
    if not match:
        return word

    tltk_phon = match.group(1)

    # Split syllables on ~ | ^ and '
    syllables = []
    for part in re.split(r"[|~^]", tltk_phon):
        for sp in part.split("'"):
            sp = sp.strip()
            if sp:
                syllables.append(_tltk_syllable_to_ipa(sp))

    if syllables:
        return "/" + ".".join(syllables) + "/"
    return word
