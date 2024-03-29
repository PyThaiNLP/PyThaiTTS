# PyThaiTTS
Open Source Thai Text-to-speech library in Python

[Google Colab](https://colab.research.google.com/github/PyThaiNLP/PyThaiTTS/blob/dev/notebook/use_lunarlist_model.ipynb) | [Docs](https://pythainlp.github.io/PyThaiTTS/) | [Notebooks](https://github.com/PyThaiNLP/PyThaiTTS/tree/dev/notebook)
<a href="https://pepy.tech/project/pythaitts"><img alt="Download" src="https://pepy.tech/badge/pythaitts/month"/></a>

License: [Apache-2.0 License](https://github.com/PyThaiNLP/pythaitts/blob/main/LICENSE)

## Install

Install by pip:

> pip install pythaitts

## Usage

```python
from pythaitts import TTS

tts = TTS()
file = tts.tts("ภาษาไทย ง่าย มาก มาก", filename="cat.wav") # It will get wav file path.
wave = tts.tts("ภาษาไทย ง่าย มาก มาก",return_type="waveform") # It will get waveform.
```

You can see more at [https://pythainlp.github.io/PyThaiTTS/](https://pythainlp.github.io/PyThaiTTS/).
