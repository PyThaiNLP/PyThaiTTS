# PyThaiTTS
Open Source Thai Text-to-speech library in Python

License: [Apache-2.0 License](https://github.com/PyThaiNLP/pythaitts/blob/main/LICENSE)

## Install

Install by pip:

> pip install pythaitts

## Usage

```python
from pythaitts import TTS

tts = TTS()
file = tts.tts("ภาษาไทย ง่าย มาก มาก") # It will get temp file path.
wave = tts.tts("ภาษาไทย ง่าย มาก มาก",return_type="waveform") # It will get waveform.
```

You can see more at [https://pythainlp.github.io/PyThaiTTS/](https://pythainlp.github.io/PyThaiTTS/).