# -*- coding: utf-8 -*-
"""
KhanomTan TTS model V1.0

KhanomTan TTS (ขนมตาล) is a open-source Thai text-to-speech model that supports multilingual speakers.
It supports Thai, English, and others.

Homepage: `https://github.com/wannaphong/KhanomTan-TTS-v1.0 <https://github.com/wannaphong/KhanomTan-TTS-v1.0>`_
"""
import tempfile
from TTS.utils.synthesizer import Synthesizer
from huggingface_hub import hf_hub_download


class KhanomTan:
    def __init__(self, mode) -> None:
        self.config_path = hf_hub_download(repo_id="wannaphong/khanomtan-tts-v1.0",filename="config.json",force_filename="config.json")
        self.speakers_path =  hf_hub_download(repo_id="wannaphong/khanomtan-tts-v1.0",filename="speakers.pth",force_filename="speakers.pth")
        self.languages_path = hf_hub_download(repo_id="wannaphong/khanomtan-tts-v1.0",filename="language_ids.json",force_filename="language_ids.json")
        self.speaker_encoder_model_path = hf_hub_download(repo_id="wannaphong/khanomtan-tts-v1.0",filename="model_se.pth",force_filename="model_se.pth")
        self.speaker_encoder_config_path = hf_hub_download(repo_id="wannaphong/khanomtan-tts-v1.0",filename="config_se.json",force_filename="config_se.json")
        self.synthesizer = None
        with open(self.config_path,"r") as f:
            _temp = f.read()
        _temp = _temp.replace("speakers.pth",self.speakers_path)
        _temp = _temp.replace("language_ids.json",self.languages_path)
        _temp = _temp.replace("config_se.json",self.speaker_encoder_config_path)
        _temp = _temp.replace("model_se.pth",self.speaker_encoder_model_path)
        with open("config-khanomtan.json","w",encoding="utf-8") as fp:
            fp.write(_temp)
        self.config_path = "config-khanomtan.json"
        self.load_synthesizer(mode)
    def load_synthesizer(self, mode):
        """
        mode: The model mode (best_mode or last_checkpoint)
        """
        if mode=="best_model":
            self.best_model_path = hf_hub_download(repo_id="wannaphong/khanomtan-tts-v1.0",filename="best_model.pth",force_filename="best_model.pth")
            self.synthesizer = Synthesizer(
                self.best_model_path,
                self.config_path,
                self.speakers_path,
                self.languages_path,
                None,
                None,
                self.speaker_encoder_model_path,
                self.speaker_encoder_config_path,
                False
            )
        else:
            self.last_checkpoint_model_path = hf_hub_download(repo_id="wannaphong/khanomtan-tts-v1.0",filename="checkpoint_440000.pth",force_filename="checkpoint_440000.pth")
            self.synthesizer = Synthesizer(
                self.last_checkpoint_model_path,
                self.config_path,
                self.speakers_path,
                self.languages_path,
                None,
                None,
                self.speaker_encoder_model_path,
                self.speaker_encoder_config_path,
                False
            )
    def __call__(self, text: str, speaker_idx: str, language_idx: str, return_type: str = "file", filename: str = None):
        wavs = self.synthesizer.tts(text, speaker_idx, language_idx)
        if return_type == "waveform":
            return wavs
        if filename != None:
            with open(filename, "wb") as fp:
                self.synthesizer.save_wav(wavs, fp)
        else:
            with tempfile.NamedTemporaryFile(suffix = ".wav", delete = False) as fp:
                self.synthesizer.save_wav(wavs, fp)
        return fp.name