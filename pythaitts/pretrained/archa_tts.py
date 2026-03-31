# -*- coding: utf-8 -*-
"""
Archa TTS model (YangNobody12/Archa-TTS-0.5B-th)

Archa TTS is a Thai text-to-speech model built on Qwen2.5-0.5B with LoRA fine-tuning
and SNAC 24kHz audio codec.

See more: https://github.com/YangNobody12/Archa-TTS-0.5B-th
HuggingFace model: https://huggingface.co/Pakorn2112/Archa-TTS-0.5B-th
"""
import tempfile
import os
import numpy as np

BASE_MODEL_PATH = "Pakorn2112/Archa-TTS-0.5B-th"
SNAC_MODEL_PATH = "hubertsiuzdak/snac_24khz"
SNAC_SR = 24000
TOKENISER_LENGTH = 151665
VOCAB_SIZE = 180500

# Special token IDs
END_OF_TEXT = TOKENISER_LENGTH + 2
START_OF_SPEECH = TOKENISER_LENGTH + 3
END_OF_SPEECH = TOKENISER_LENGTH + 4
START_OF_HUMAN = TOKENISER_LENGTH + 5
END_OF_HUMAN = TOKENISER_LENGTH + 6
START_OF_AI = TOKENISER_LENGTH + 7
AUDIO_TOKENS_START = TOKENISER_LENGTH + 10  # 151675

# Token generation defaults
ESTIMATED_TOKENS_PER_CHAR = 30
MIN_MAX_NEW_TOKENS = 16384


class ArchaTTS:
    def __init__(self, device: str = None) -> None:
        """
        Initialize ArchaTTS model.
        The model will be automatically downloaded from HuggingFace on first use.

        :param str device: Device to run the model on ('cpu' or 'cuda').
                           Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is not installed. Please install it with: pip install torch"
            )
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is not installed. Please install it with: pip install transformers"
            )
        try:
            from snac import SNAC
        except ImportError:
            raise ImportError(
                "snac is not installed. Please install it with: pip install snac"
            )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        torch_dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, torch_dtype=torch_dtype, device_map=self.device
        )
        self.model.resize_token_embeddings(VOCAB_SIZE)
        self.snac_model = SNAC.from_pretrained(SNAC_MODEL_PATH).eval().to(self.device)

    def _decode_tokens(self, token_list):
        """Decode a list of audio tokens into a waveform using SNAC."""
        valid_len = (len(token_list) // 7) * 7
        token_list = token_list[:valid_len]
        if valid_len < 7:
            return np.array([], dtype=np.float32)

        import torch

        l1, l2, l3 = [], [], []
        for i in range(valid_len // 7):
            b = 7 * i
            codes = [t - AUDIO_TOKENS_START for t in token_list[b : b + 7]]
            l1.append(codes[0])
            l2.append(codes[1] - 4096)
            l3.append(codes[2] - 2 * 4096)
            l3.append(codes[3] - 3 * 4096)
            l2.append(codes[4] - 4 * 4096)
            l3.append(codes[5] - 5 * 4096)
            l3.append(codes[6] - 6 * 4096)

        snac_codes = [
            torch.tensor(l1, dtype=torch.long).unsqueeze(0).to(self.device),
            torch.tensor(l2, dtype=torch.long).unsqueeze(0).to(self.device),
            torch.tensor(l3, dtype=torch.long).unsqueeze(0).to(self.device),
        ]
        with torch.no_grad():
            audio = self.snac_model.decode(snac_codes)
        return audio.squeeze().cpu().numpy()

    def _generate_audio_tokens(self, text: str) -> list:
        """Generate audio tokens from text."""
        import torch

        text_ids = self.tokenizer.encode(text, add_special_tokens=True)
        text_ids.append(END_OF_TEXT)
        prompt_ids = (
            [START_OF_HUMAN]
            + text_ids
            + [END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
        )
        input_ids = torch.tensor([prompt_ids]).to(self.device)

        estimated_tokens = len(text) * ESTIMATED_TOKENS_PER_CHAR
        max_tokens = max(MIN_MAX_NEW_TOKENS, estimated_tokens)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                use_cache=True,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=END_OF_SPEECH,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][len(prompt_ids) :].tolist()
        audio_tokens = [t for t in generated if t >= AUDIO_TOKENS_START]
        return audio_tokens

    def _denoise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio."""
        try:
            import noisereduce as nr

            return nr.reduce_noise(y=audio, sr=SNAC_SR, prop_decrease=0.8).astype(
                np.float32
            )
        except ImportError:
            return audio.astype(np.float32)

    def __call__(
        self,
        text: str,
        return_type: str = "file",
        filename: str = None,
        **kwargs,
    ):
        """
        Generate speech from text using Archa TTS.

        :param str text: Input Thai text to synthesize
        :param str return_type: Return type ("file" or "waveform"). Default is "file".
        :param str filename: Output filename for the generated audio (WAV). Used when
                             return_type is "file". A temporary file is created if None.
        :return: File path if return_type is "file", otherwise numpy waveform array.
        """
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "soundfile is not installed. Please install it with: pip install soundfile"
            )

        audio_tokens = self._generate_audio_tokens(text)
        audio = self._decode_tokens(audio_tokens)

        if len(audio) == 0:
            raise RuntimeError("Archa TTS failed to generate audio for the given text.")

        audio = self._denoise(audio)

        if return_type == "waveform":
            return audio

        # File output
        if filename is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                filename = fp.name

        sf.write(filename, audio, SNAC_SR)
        return filename
