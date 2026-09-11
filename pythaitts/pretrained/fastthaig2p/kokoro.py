"""IPA-to-Kokoro phoneme mapping for Thai.

Maps FastThaiG2P's Wiktionary IPA output to Kokoro-82M's phoneme vocabulary.
Kokoro doesn't natively support Thai, but its phoneme set covers most Thai
segmental phonemes. Tones are approximated via Kokoro's intonation marks.
"""

from __future__ import annotations

import re


# Thai tone → Kokoro intonation mark (one symbol per tone, no mergers)
# ↑ is an otherwise-unused Kokoro token (ID 170) repurposed as the high tone
# so that high (˦˥) and rising (˩˩˦) stay phonemically distinct.
_TONE_MAP = [
    ("˩˩˦", "↗"),  # rising
    ("˥˩", "↘"),   # falling
    ("˦˥", "↑"),   # high
    ("˨˩", "↓"),   # low
    ("˧", "→"),    # mid
]

# Thai affricates → Kokoro precomposed symbols
_AFFRICATE_MAP = [
    ("t͡ɕʰ", "ʨʰ"),
    ("t͡ɕ", "ʨ"),
]

# Characters to strip (no Kokoro equivalent)
_STRIP_CHARS = ["̚", "̯", "͡"]

# Character replacements
_CHAR_MAP = {
    "g": "ɡ",  # ASCII g → IPA g
}


def ipa_to_kokoro(ipa: str) -> str:
    """Convert FastThaiG2P IPA output to Kokoro-compatible phoneme string.

    Strips word-boundary slashes and maps tones, affricates, and unsupported
    characters to Kokoro equivalents.

    Args:
        ipa: Raw output from G2P.convert() e.g. '/sa˨˩.wat̚˨˩.diː˧/ /kʰrap̚˦˥/'

    Returns:
        Kokoro phoneme string e.g. 'sa↓.wat↓.diː→ kʰrap↗'
    """
    # Strip slashes, join words with space
    phonemes = " ".join(p.strip("/") for p in ipa.split() if p.startswith("/"))
    if not phonemes:
        phonemes = ipa.replace("/", "").strip()

    # Affricates (before stripping tie bar)
    for old, new in _AFFRICATE_MAP:
        phonemes = phonemes.replace(old, new)

    # Tones (longest match first)
    for old, new in _TONE_MAP:
        phonemes = phonemes.replace(old, new)

    # Strip unsupported combining characters
    for char in _STRIP_CHARS:
        phonemes = phonemes.replace(char, "")

    # Character replacements
    for old, new in _CHAR_MAP.items():
        phonemes = phonemes.replace(old, new)

    return phonemes
