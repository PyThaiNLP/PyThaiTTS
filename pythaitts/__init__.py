# -*- coding: utf-8 -*-
"""
PyThaiTTS
"""
__version__ = "0.1.1"


class TTS:
    def __init__(self, pretrained="khanomtan", mode="last_checkpoint", version="1.0") -> None:
        """
        :param str pretrained: TTS pretrained (khanomtan)
        :param str mode: pretrained mode
        :param str version: model version (default is 1.0 or 1.1)

        **Options for mode**
            * *last_checkpoint* (default) - last checkpoint of model
            * *best_model* - Best model (best loss)
        
        You can see more about khanomtan tts at `https://github.com/wannaphong/KhanomTan-TTS-v1.0 <https://github.com/wannaphong/KhanomTan-TTS-v1.0>`_
        and `https://github.com/wannaphong/KhanomTan-TTS-v1.1 <https://github.com/wannaphong/KhanomTan-TTS-v1.1>`_
        """
        self.pretrained = pretrained
        self.mode = mode
        self.load_pretrained(version=version)

    def load_pretrained(self,version):
        """
        Load pretrined
        """
        if self.pretrained == "khanomtan":
            from pythaitts.pretrained import KhanomTan
            self.model = KhanomTan(mode=self.mode, version=version)
        else:
            raise NotImplemented(
                "PyThaiTTS doesn't support %s pretrained." % self.pretrained
            )

    def tts(self, text: str, speaker_idx: str = "Linda", language_idx: str = "th-th", return_type: str = "file", filename: str = None):
        """
        speech synthesis

        :param str text: text
        :param str speaker_idx: speaker (default is Linda)
        :param str language_idx: language (default is th-th)
        :param str return_type: return type (default is file)
        :param str filename: path filename for save wav file if return_type is file.
        """
        return self.model(
            text=text,
            speaker_idx=speaker_idx,
            language_idx=language_idx,
            return_type=return_type,
            filename=filename
        )
