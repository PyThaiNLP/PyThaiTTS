# -*- coding: utf-8 -*-
"""
PyThaiTTS
"""
__version__ = "0.4.2"

from pythaitts.preprocess import preprocess_text, num_to_thai, expand_maiyamok


class TTS:
    def __init__(self, pretrained="lunarlist_onnx", mode="last_checkpoint", version="1.0", device:str="cpu") -> None:
        """
        :param str pretrained: TTS pretrained (lunarlist_onnx, khanomtan, lunarlist, vachana, archa)
        :param str mode: pretrained mode (lunarlist_onnx and vachana don't support)
        :param str version: model version (default is 1.0 or 1.1)
        :param str device: device for running model. (lunarlist_onnx and vachana support CPU only. archa supports cpu and cuda.)

        **Options for mode**
            * *last_checkpoint* (default) - last checkpoint of model
            * *best_model* - Best model (best loss)
        
        You can see more about khanomtan tts at `https://github.com/wannaphong/KhanomTan-TTS-v1.0 <https://github.com/wannaphong/KhanomTan-TTS-v1.0>`_
        and `https://github.com/wannaphong/KhanomTan-TTS-v1.1 <https://github.com/wannaphong/KhanomTan-TTS-v1.1>`_
        If you want to use khanomtan tts, you must to install coqui-tts before use the model by pip install coqui-tts.
        
        For lunarlist tts model, you must to install nemo before use the model by pip install nemo_toolkit['tts'].
        You can see more about lunarlist tts at `https://link.medium.com/OpPjQis6wBb <https://link.medium.com/OpPjQis6wBb>`_

        For lunarlist_onnx tts model, \
        You can see more about lunarlist tts at `https://github.com/PyThaiNLP/thaitts-onnx <https://github.com/PyThaiNLP/thaitts-onnx>`_

        For vachana tts model, \
        You can see more about vachana tts at `https://github.com/VYNCX/VachanaTTS2 <https://github.com/VYNCX/VachanaTTS2>`_

        For archa tts model, you must install the required packages before use: \
        pip install torch transformers snac soundfile noisereduce scipy numpy. \
        You can see more about archa tts at `https://github.com/YangNobody12/Archa-TTS-0.5B-th <https://github.com/YangNobody12/Archa-TTS-0.5B-th>`_

        
        """
        self.pretrained = pretrained
        self.mode = mode
        self.device = device
        self.load_pretrained(version=version)

    def load_pretrained(self,version):
        """
        Load pretrained
        """
        if self.pretrained == "lunarlist_onnx":
            from pythaitts.pretrained.lunarlist_onnx import LunarlistONNX
            self.model = LunarlistONNX()
        elif self.pretrained == "khanomtan":
            from pythaitts.pretrained.khanomtan_tts import KhanomTan
            self.model = KhanomTan(mode=self.mode, version=version)
        elif self.pretrained == "lunarlist":
            from pythaitts.pretrained.lunarlist_model import LunarlistModel
            self.model = LunarlistModel(mode=self.mode, device=self.device)
        elif self.pretrained == "vachana":
            from pythaitts.pretrained.vachana_tts import VachanaTTS
            self.model = VachanaTTS()
        elif self.pretrained == "archa":
            from pythaitts.pretrained.archa_tts import ArchaTTS
            self.model = ArchaTTS(device=self.device)
        else:
            raise NotImplementedError(
                "PyThaiTTS doesn't support %s pretrained." % self.pretrained
            )

    def tts(self, text: str, speaker_idx: str = "Linda", language_idx: str = "th-th", return_type: str = "file", filename: str = None, preprocess: bool = True):
        """
        speech synthesis

        :param str text: text
        :param str speaker_idx: speaker (default is Linda for khanomtan, th_f_1 for vachana)
        :param str language_idx: language (default is th-th)
        :param str return_type: return type (default is file)
        :param str filename: path filename for save wav file if return_type is file.
        :param bool preprocess: whether to preprocess text (convert numbers to Thai text and expand ๆ). Default is True.
        """
        # Preprocess text if requested
        if preprocess:
            from pythaitts.preprocess import preprocess_text
            text = preprocess_text(text)
        
        if self.pretrained == "lunarlist" or self.pretrained == "lunarlist_onnx":
            return self.model(text=text,return_type=return_type,filename=filename)
        elif self.pretrained == "vachana":
            return self.model(text=text,speaker_idx=speaker_idx,return_type=return_type,filename=filename)
        elif self.pretrained == "archa":
            return self.model(text=text,return_type=return_type,filename=filename)
        return self.model(
            text=text,
            speaker_idx=speaker_idx,
            language_idx=language_idx,
            return_type=return_type,
            filename=filename
        )
