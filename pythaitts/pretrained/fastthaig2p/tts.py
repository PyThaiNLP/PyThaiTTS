"""Text-to-speech: Thai text in, audio out.

Wraps G2P + a Thai-finetuned Kokoro-82M checkpoint (trained via
https://github.com/cstorm125/kukuru-tts). Zero-arg TTS() downloads the
released model; or pass paths to your own artifacts.

Two backends, chosen by model file extension:
  .onnx  — onnxruntime (default/recommended: no torch import, ~6s
           lighter cold start; export via scripts/export_onnx.py)
  .pth   — PyTorch KModel (needed for GPU inference)
"""

from __future__ import annotations

import json
import os
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from .g2p import G2P
from .kokoro import ipa_to_kokoro


# Default model release — downloaded on first TTS() with no arguments.
# Full 2-stage Thai fine-tune (stage 2 incl. adversarial phase).
_REPO = "awslabs/FastThaiG2P"
_RELEASE_URL = f"https://github.com/{_REPO}/releases/download/v0.3.0"
_DEFAULT_ASSETS = {
    "model": "kokoro_thai.onnx",
    "voicepack": "thai_som.npy",  # npy so the ONNX path needs no torch
    "config": "config.json",
}


def _default_asset(kind: str) -> Path:
    """Return the cached path of a default asset, downloading if missing.

    Public repos need no auth. While the repo is private, set GITHUB_TOKEN
    (repo scope) — the download then goes through the GitHub API.
    """
    import urllib.request

    cache = Path(
        os.environ.get("FASTTHAIG2P_HOME", Path.home() / ".cache" / "fastthaig2p")
    )
    cache.mkdir(parents=True, exist_ok=True)
    name = _DEFAULT_ASSETS[kind]
    dest = cache / name
    if dest.exists():
        return dest

    print(f"Downloading {name} → {dest} ...")
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        # resolve the asset id via the API, then fetch with octet-stream
        import json as _json

        tag = _RELEASE_URL.rsplit("/", 1)[-1]
        api = f"https://api.github.com/repos/{_REPO}/releases/tags/{tag}"
        req = urllib.request.Request(api, headers={"Authorization": f"token {token}"})
        release = _json.load(urllib.request.urlopen(req))
        asset = next(a for a in release["assets"] if a["name"] == name)
        req = urllib.request.Request(
            asset["url"],
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/octet-stream",
            },
        )
    else:
        req = urllib.request.Request(f"{_RELEASE_URL}/{name}")

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Could not download {name} ({e.code}). If the repo is private, "
            "set GITHUB_TOKEN, or pass explicit model/voicepack/config paths."
        ) from e
    tmp.rename(dest)
    return dest


class TTS:
    """Thai TTS powered by FastThaiG2P + Kokoro-82M.

    Usage:
        # Batteries included: downloads the default ONNX model, voicepack,
        # and config to ~/.cache/fastthaig2p on first use (~330 MB)
        tts = TTS()

        # Or bring your own artifacts
        tts = TTS("kokoro_thai.onnx", "thai_som.pt", config_path="config.json")
        tts = TTS("kokoro_thai.pth", "thai_som.pt", config_path="config.json")  # GPU

        tts.synthesize("สวัสดีครับ", "out.wav")
        audio = tts.generate("สวัสดีครับ")  # float32 numpy, 24 kHz
    """

    MAX_PHONEMES = 510  # Kokoro context limit (BOS/EOS excluded)
    SUPPORTED_VOICES = ["thai_som"]

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        voicepack_path: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        speed: float = 1.0,
        intra_op_threads: int = 0,
    ):
        if model_path is None:
            model_path = _default_asset("model")
            voicepack_path = voicepack_path or _default_asset("voicepack")
            config_path = config_path or _default_asset("config")
        if voicepack_path is None:
            raise ValueError("voicepack_path is required when model_path is given")
        model_path = Path(model_path)
        voicepack_path = Path(voicepack_path)
        for p in (model_path, voicepack_path):
            if not p.exists():
                raise FileNotFoundError(f"Not found: {p}")

        self._onnx = model_path.suffix == ".onnx"
        if self._onnx:
            if not config_path:
                raise ValueError("config_path (vocab) is required with ONNX models")
            import onnxruntime as ort

            options = ort.SessionOptions()
            if intra_op_threads:
                options.intra_op_num_threads = intra_op_threads
            self._session = ort.InferenceSession(
                str(model_path), options, providers=["CPUExecutionProvider"]
            )
            with open(config_path, encoding="utf-8") as f:
                self._vocab = json.load(f)["vocab"]
            # voicepack without torch: torch .pt zips are torch-only, so
            # load lazily via numpy when possible, else fall back to torch
            self._voice = _load_voicepack(voicepack_path)
        else:
            import torch
            from kokoro import KModel

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = KModel(
                repo_id="hexgrad/Kokoro-82M",
                config=str(config_path) if config_path else None,
                model=str(model_path),
            ).to(device).eval()
            self._voice = torch.load(
                voicepack_path, map_location="cpu", weights_only=True
            )
            self._device = device

        self._g2p = G2P()
        self.sample_rate = 24000
        self.speed = speed

    def _text_to_phonemes(self, text: str) -> str:
        return ipa_to_kokoro(self._g2p.convert(text))

    def generate(self, text: str) -> "np.ndarray":
        """Convert Thai text to float32 audio array (24 kHz mono)."""
        phonemes = self._text_to_phonemes(text)
        if not phonemes:
            return np.array([], dtype=np.float32)
        if len(phonemes) > self.MAX_PHONEMES:
            raise ValueError(
                f"Text produces {len(phonemes)} phonemes; max {self.MAX_PHONEMES}. "
                "Split the input into shorter chunks."
            )
        ref_s = self._voice[len(phonemes) - 1]

        if self._onnx:
            ids = [self._vocab[c] for c in phonemes if c in self._vocab]
            input_ids = np.array([[0, *ids, 0]], dtype=np.int64)
            ref = np.asarray(ref_s, dtype=np.float32)
            audio, durations = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "ref_s": ref,
                    "speed": np.full(1, self.speed, dtype=np.float32),
                },
            )
            return _trim_boundary_tokens(audio, durations)

        import torch

        ids = [self._model.vocab[c] for c in phonemes if c in self._model.vocab]
        input_ids = torch.LongTensor([[0, *ids, 0]]).to(self._device)
        ref_t = (
            torch.from_numpy(ref_s) if isinstance(ref_s, np.ndarray)
            else ref_s
        ).to(self._device)
        with torch.no_grad():
            audio, durations = self._model.forward_with_tokens(
                input_ids, ref_t, speed=self.speed
            )
        return _trim_boundary_tokens(
            audio.cpu().numpy(), durations.cpu().numpy()
        )

    def synthesize(self, text: str, output_path: str | Path) -> Path:
        """Convert Thai text to WAV file."""
        output_path = Path(output_path)
        audio = self.generate(text)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return output_path

    def __call__(
        self,
        text: str,
        speaker_idx: str = "thai_som",
        return_type: str = "file",
        filename: Optional[str] = None,
        **kwargs,
    ):
        """Generate speech from text using FastThaiG2P TTS.

        :param str text: Input text to synthesize
        :param str speaker_idx: Voice to use (default: "thai_som" or path to custom voicepack)
        :param str return_type: Return type ("file" or "waveform")
        :param str filename: Output filename for the generated audio
        :param kwargs: Additional parameters (e.g., speed)
        :return: File path if return_type is "file", otherwise audio waveform data (numpy.ndarray)
        """
        if speaker_idx in ("Linda", None):
            speaker_idx = "thai_som"

        if speaker_idx not in self.SUPPORTED_VOICES and not os.path.exists(str(speaker_idx)):
            raise ValueError(
                f"Unsupported voice '{speaker_idx}'. Supported voices are: {', '.join(self.SUPPORTED_VOICES)}"
            )

        if "speed" in kwargs:
            self.speed = kwargs["speed"]

        if return_type == "waveform":
            return self.generate(text)
        else:
            import tempfile

            if filename is None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                    filename = fp.name

            self.synthesize(text, filename)
            return str(filename)


FastThaiG2P = TTS


def _trim_boundary_tokens(
    audio: "np.ndarray",
    durations: "np.ndarray",
    sample_rate: int = 24000,
) -> "np.ndarray":
    """Remove the static generated for the BOS/EOS pad tokens.

    Training pads every clip with token 0 aligned to zero samples, so at
    inference the model fills the BOS token's ~400 ms with noise — often
    including a faint hallucinated "ghost" copy of the first syllable —
    before the real speech starts at the BOS boundary. A plain energy
    onset detector latches onto that ghost, so instead we anchor at the
    BOS boundary and walk backwards only while the audio stays *voiced*
    (periodic and energetic, allowing short gaps): speech that genuinely
    starts early connects to the boundary and is kept, while a
    disconnected ghost blob is trimmed along with the static.
    """
    total_frames = int(durations.sum())
    if total_frames <= 0 or len(audio) == 0:
        return audio
    samples_per_frame = len(audio) / total_frames
    bos_end = int(durations[0] * samples_per_frame)
    eos_start = len(audio) - int(durations[-1] * samples_per_frame)

    win = sample_rate // 100  # 10 ms windows
    n_win = len(audio) // win
    if n_win == 0:
        return audio
    frames = audio[: n_win * win].reshape(n_win, win)
    env = np.sqrt(np.mean(frames**2, axis=1))
    threshold = env.max() * 0.1  # -20 dB relative to peak

    # Voicing per window: normalized autocorrelation peak in the pitch
    # range (60-400 Hz). Ghost static sits ~0.2; voiced speech >0.5.
    lo, hi = sample_rate // 400, sample_rate // 60
    centered = frames - frames.mean(axis=1, keepdims=True)
    speech = np.zeros(n_win, dtype=bool)
    for i in np.flatnonzero(env >= threshold):
        x = centered[i]
        ac = np.correlate(x, x, "full")[win - 1 :]
        if ac[0] > 1e-9 and ac[lo:hi].max() / ac[0] > 0.5:
            speech[i] = True

    max_gap = 5  # bridge unvoiced stretches up to 50 ms (stops, fricatives)
    margin = sample_rate // 40  # 25 ms of natural attack/decay

    # Head: from the BOS boundary, extend backwards through voiced audio.
    anchor = min(bos_end // win, n_win - 1)
    first_speech = anchor
    gap = 0
    for i in range(anchor, -1, -1):
        if speech[i]:
            first_speech = i
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
    start = max(0, min(first_speech * win, bos_end) - margin)

    # Tail: mirror from the EOS boundary forwards through voiced audio.
    anchor = min(eos_start // win, n_win - 1)
    last_speech = anchor
    gap = 0
    for i in range(anchor, n_win):
        if speech[i]:
            last_speech = i
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
    end = min(len(audio), max(last_speech * win + win, eos_start) + margin)

    trimmed = audio[start:end].copy()
    fade = min(sample_rate // 100, len(trimmed))  # 10 ms
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=trimmed.dtype)
        trimmed[:fade] *= ramp
        trimmed[-fade:] *= ramp[::-1]
    return trimmed


def _load_voicepack(path: Path) -> "np.ndarray":
    """Load a [510, 1, 256] voicepack as numpy. .npy needs no torch;
    torch-serialized .pt files require torch (install fastthaig2p[tts-torch]
    or convert once: np.save(p.with_suffix('.npy'), torch.load(p).numpy())."""
    if path.suffix == ".npy":
        return np.load(path)
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            f"{path.name} is a torch-serialized voicepack but torch is not "
            "installed. Use a .npy voicepack (the default download is one), "
            "or pip install 'fastthaig2p[tts-torch]'."
        ) from e
    return torch.load(path, map_location="cpu", weights_only=True).numpy()
