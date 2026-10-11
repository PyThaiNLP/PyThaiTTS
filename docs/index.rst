.. currentmodule:: pythaitts
PyThaiTTS
=========

Open Source Thai Text-to-speech library in Python

.. autoclass:: TTS
   :members:

Real-time / Streaming TTS
-------------------------

PyThaiTTS provides real-time streaming speech synthesis and integration with `RealtimeTTS <https://github.com/KoljaB/RealtimeTTS>`_.

.. autoclass:: RealtimeTTS
   :members:

.. autoclass:: FastThaiG2PEngine
   :members:

.. autoclass:: pythaitts.realtime.FastThaiG2PVoice
   :members:

.. autofunction:: chunk_text

.. autofunction:: stream_text_to_chunks

English-to-Thai-IPA G2P (Code-Switching)
----------------------------------------

PyThaiTTS provides English-to-Thai-IPA Grapheme-to-Phoneme (G2P) conversion for Thai-English code-switching synthesis.

.. autoclass:: EnglishG2P
   :members:

.. autofunction:: en_g2p

.. autofunction:: en_to_ipa

.. autofunction:: en_ipa_to_thai_ipa

FastThaiG2P Components
----------------------

Low-level components from :mod:`pythaitts.pretrained.fastthaig2p`.

.. autoclass:: pythaitts.pretrained.fastthaig2p.tts.TTS
   :members:

.. autoclass:: pythaitts.pretrained.fastthaig2p.G2P
   :members:

.. autoclass:: pythaitts.pretrained.fastthaig2p.Tokenizer
   :members:

.. autofunction:: pythaitts.pretrained.fastthaig2p.normalize

.. autofunction:: pythaitts.pretrained.fastthaig2p.ipa_to_kokoro

Text Preprocessing
------------------

PyThaiTTS provides text preprocessing functions to improve TTS quality.

.. autofunction:: preprocess_text

.. autofunction:: num_to_thai

.. autofunction:: expand_maiyamok