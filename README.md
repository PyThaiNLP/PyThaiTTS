# PyThaiTTS
Open Source Thai Text-to-speech library in Python

[Google Colab](https://colab.research.google.com/github/PyThaiNLP/PyThaiTTS/blob/main/notebook/use_fastthaig2p.ipynb) | [Docs](https://pythainlp.github.io/PyThaiTTS/) | [Notebooks](https://github.com/PyThaiNLP/PyThaiTTS/tree/main/notebook)
<a href="https://pepy.tech/project/pythaitts"><img alt="Download" src="https://pepy.tech/badge/pythaitts/month"/></a>

License: [Apache-2.0 License](https://github.com/PyThaiNLP/pythaitts/blob/main/LICENSE)

## Install

Install by pip:

```sh
pip install pythaitts
```

## Usage

### Basic Usage

```python
from pythaitts import TTS

tts = TTS()
file = tts.tts("ภาษาไทย ง่าย มาก มาก", filename="cat.wav") # It will get wav file path.
wave = tts.tts("ภาษาไทย ง่าย มาก มาก",return_type="waveform") # It will get waveform.
```

### Using Different TTS Models

PyThaiTTS supports multiple TTS models. You can specify which model to use:

```python
from pythaitts import TTS

# Use FastThaiG2P (default) (default voice: thai_som)
# FastThaiG2P Sample Rate is 24000 Hz
tts = TTS(pretrained="fastthaig2p")
file = tts.tts("สวัสดีครับ", speaker_idx="thai_som", filename="output.wav")

# Use Lunarlist ONNX
# Sample Rate is 22050 Hz
tts = TTS(pretrained="lunarlist_onnx")
file = tts.tts("ภาษาไทย ง่าย มาก", filename="output.wav")

# Use VachanaTTS (default voices: th_f_1, th_m_1, th_f_2, th_m_2)
# Sample Rate is 22050 Hz
tts = TTS(pretrained="vachana")
file = tts.tts("สวัสดีครับ", speaker_idx="th_f_1", filename="output.wav")

# Use KhanomTan
# Sample Rate is 16000 Hz
tts = TTS(pretrained="khanomtan")
file = tts.tts("ภาษาไทย", speaker_idx="Linda", filename="output.wav")
```

### Real-time / Streaming TTS (FastThaiG2P)

PyThaiTTS supports low-latency, real-time streaming speech synthesis with FastThaiG2P, making it ideal for conversational voice agents and LLM streaming:

#### 1. Streaming Audio from Text

Synthesize chunk-by-chunk in real time:

```python
from pythaitts import TTS

tts = TTS(pretrained="fastthaig2p")

# Stream audio chunks as 24kHz float32 NumPy arrays
for audio_chunk in tts.stream("สวัสดีครับ ยินดีต้อนรับสู่ระบบเรียลไทม์ทีทีเอส"):
    print(f"Audio chunk shape: {audio_chunk.shape}")

# Stream raw 16-bit PCM bytes (for WebSockets or PyAudio)
for pcm_bytes in tts.stream("สวัสดีครับ", return_type="bytes"):
    # send over websocket or write to audio stream
    pass
```

#### 2. Streaming from an LLM Token Stream

Feed tokens directly from an LLM or generator into `tts.stream()`:

```python
from pythaitts import TTS

tts = TTS()

def token_stream():
    tokens = ["สวัสดี", "ครับ", " ", "นี่", "คือ", "การ", "สตรีม", "มิ่ง"]
    for tok in tokens:
        yield tok

for audio_chunk in tts.stream(token_stream()):
    # Process or play chunk with low latency
    pass
```

#### 3. Integration with the RealtimeTTS Library

You can use FastThaiG2P as an engine with [KoljaB/RealtimeTTS](https://github.com/KoljaB/RealtimeTTS):

```sh
pip install pythaitts[realtime]
```

```python
from RealtimeTTS import TextToAudioStream
from pythaitts.realtime import FastThaiG2PEngine

engine = FastThaiG2PEngine()
stream = TextToAudioStream(engine)
stream.feed("สวัสดีครับ วันนี้อากาศดีมาก")
stream.play()
```


### Text Preprocessing

PyThaiTTS includes automatic text preprocessing to improve TTS quality:
- **Number to Thai text conversion**: Converts digits (e.g., "123") to Thai text (e.g., "หนึ่งร้อยยี่สิบสาม")
- **Mai yamok (ๆ) expansion**: Expands the Thai repetition character (e.g., "ดีๆ" becomes "ดีดี")

Preprocessing is enabled by default:

```python
from pythaitts import TTS

tts = TTS()
# Automatic preprocessing: "มี 5 คนๆ" becomes "มี ห้า คนคน"
file = tts.tts("มี 5 คนๆ", filename="output.wav")
```

You can disable preprocessing if needed:

```python
file = tts.tts("มี 5 คนๆ", preprocess=False, filename="output.wav")
```

You can also use preprocessing functions directly:

```python
from pythaitts import num_to_thai, expand_maiyamok, preprocess_text

# Convert numbers to Thai text
print(num_to_thai("123"))  # Output: หนึ่งร้อยยี่สิบสาม

# Expand mai yamok
print(expand_maiyamok("ดีๆ"))  # Output: ดีดี

# Full preprocessing
print(preprocess_text("มี 5 คนๆ"))  # Output: มี ห้า คนคน
```

You can see more at [https://pythainlp.github.io/PyThaiTTS/](https://pythainlp.github.io/PyThaiTTS/).
