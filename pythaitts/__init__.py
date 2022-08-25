# -*- coding: utf-8 -*-
"""
PyThaiTTS
"""
__version__ = "0.1.0"


class TTS:
    def __init__(self, pretrained="khanomtan", mode="last_checkpoint") -> None:
        """
        :param str pretrained: TTS pretrained (khanomtan)
        :param str mode: pretrained mode

        **Options for mode**
            * *last_checkpoint* (default) - last checkpoint of model
            * *best_model* - Best model (best loss)
        
        You can see more about khanomtan tts at `https://github.com/wannaphong/KhanomTan-TTS-v1.0 <https://github.com/wannaphong/KhanomTan-TTS-v1.0>`_
        """
        self.pretrained = pretrained
        self.mode = mode
        self.load_pretrained()

    def load_pretrained(self):
        """
        Load pretrined
        """
        if self.pretrained == "khanomtan":
            from pythaitts.pretrained import KhanomTan
            self.model = KhanomTan(mode=self.mode)
        else:
            raise NotImplemented(
                "PyThaiTTS doesn't support %s pretrained." % self.pretrained
            )

    def tts(self, text: str, speaker_idx: str = "Tsynctwo", language_idx: str = "th-th", return_type: str = "file", filename: str = None):
        """
        speech synthesis

        :param str text: text
        :param str speaker_idx: speaker (default is Tsynctwo)
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
