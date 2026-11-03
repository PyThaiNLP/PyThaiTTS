# Docker Usage Guide for PyThaiTTS

This guide explains how to build and run PyThaiTTS using Docker.

## Building the Docker Image

To build the Docker image, run the following command from the root directory of the repository:

```bash
docker build -t pythaitts:latest .
```

This will create a Docker image named `pythaitts:latest` with all dependencies installed.

## Running the Demo

To run the demo script that demonstrates Thai text-to-speech synthesis:

```bash
docker run --rm pythaitts:latest
```

The demo will:
1. Initialize the PyThaiTTS model (default: fastthaig2p)
2. Generate speech from Thai text
3. Save the output to a WAV file
4. Display the waveform information

## Custom Usage

### Interactive Shell

To start an interactive shell inside the container:

```bash
docker run --rm -it pythaitts:latest /bin/bash
```

### Run Custom Python Script

To run your own Python script:

```bash
docker run --rm -v $(pwd)/your_script.py:/app/custom.py pythaitts:latest python custom.py
```

### Save Output Files

To save generated audio files to your host machine:

```bash
docker run --rm -v $(pwd)/output:/app/output pythaitts:latest python -c "
from pythaitts import TTS
tts = TTS()
tts.tts('สวัสดีครับ', filename='output/hello.wav')
"
```

This will save the generated `hello.wav` file to the `output` directory on your host machine.

## Example Usage in Python

Inside the container, you can use PyThaiTTS as follows:

```python
from pythaitts import TTS

# Initialize TTS with default model
tts = TTS()

# Generate speech and save to file
file_path = tts.tts("ภาษาไทย ง่าย มาก มาก", filename="output.wav")
print(f"Audio saved to: {file_path}")

# Generate speech and get waveform
waveform = tts.tts("ภาษาไทย ง่าย มาก มาก", return_type="waveform")
print(f"Waveform shape: {waveform.shape}")
```

## Available TTS Models

PyThaiTTS supports multiple models:

- **fastthaig2p** (default): FastThaiG2P model (voice: thai_som)
- **lunarlist_onnx**: ONNX-optimized model, CPU-only
- **vachana**: VachanaTTS model
- **khanomtan**: KhanomTan TTS model
- **lunarlist**: Original Lunarlist model

To use a different model:

```python
from pythaitts import TTS

# Using KhanomTan model
tts = TTS(pretrained="khanomtan", version="1.0")
```

## Requirements

- Docker installed on your system
- At least 2GB of available disk space
- Internet connection for downloading models on first run

## Troubleshooting

If you encounter issues with model downloads, ensure:
1. You have a stable internet connection
2. The Hugging Face Hub is accessible from your network
3. You have sufficient disk space for model files

## Notes

- The first run will download model files from Hugging Face Hub, which may take some time depending on your internet connection
- Generated audio files are in WAV format
- The default model (fastthaig2p) runs on CPU and doesn't require GPU support
