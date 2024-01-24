# -*- coding: utf-8 -*-
"""
Lunarlist TTS model

You can see more about lunarlist tts at `https://link.medium.com/OpPjQis6wBb <https://link.medium.com/OpPjQis6wBb>`_
"""
import tempfile
try:
    import torch
except ImportError:
    raise ImportError("You must to install torch before use this model.")


class LunarlistModel:
    def __init__(self, mode:str="last_checkpoint", device:str="cpu") -> None:
        try:
            from nemo.collections.tts.models import UnivNetModel
        except ImportError:
            raise ImportError("You must to install nemo by pip install nemo_toolkit['tts'] before use this model.")
        self.mode = mode
        self.device = device
        self.vcoder_model = UnivNetModel.from_pretrained(model_name="tts_en_libritts_univnet").to(self.device)
        self.load_synthesizer(self.mode)
    def load_synthesizer(self, mode:str):
        from nemo.collections.tts.models import Tacotron2Model
        if mode=="last_checkpoint":
            self.model = Tacotron2Model.from_pretrained("lunarlist/tts-thai-last-step").to(self.device)
        else:
            self.model = Tacotron2Model.from_pretrained("lunarlist/tts-thai").to(self.device)
        self.dict_idx={k:i for i,k in enumerate(self.model.hparams["cfg"]['labels'])}
    def tts(self, text: str):
        parsed2=torch.Tensor([[66]+[self.dict_idx[i] for i in text if i]+[67]]).int().to(self.device)
        spectrogram2 = self.model.generate_spectrogram(tokens=parsed2)
        audio2 = self.vcoder_model.convert_spectrogram_to_audio(spec=spectrogram2)
        return audio2.to('cpu').detach().numpy()
    def __call__(self, text: str,return_type: str = "file", filename: str = None):
        wavs = self.tts(text)
        if return_type == "waveform":
            return wavs
        import soundfile as sf
        if filename != None:
            sf.write(filename, wavs[0], 22050)
            return filename
        else:
            with tempfile.NamedTemporaryFile(suffix = ".wav", delete = False) as fp:
                sf.write(fp.name, wavs[0], 22050)
            return fp.name