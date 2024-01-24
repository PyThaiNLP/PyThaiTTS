# -*- coding: utf-8 -*-
"""
Lunarlist TTS model (ONNX)

You can see more about lunarlist tts at `https://link.medium.com/OpPjQis6wBb <https://link.medium.com/OpPjQis6wBb>`_

ONNX port: `https://github.com/PyThaiNLP/thaitts-onnx <https://github.com/PyThaiNLP/thaitts-onnx>`_
"""
import tempfile
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download


# from https://huggingface.co/lunarlist/tts-thai-last-step
index_list=['ก', 'ข', 'ค', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 'ะ', 'ั', 'า', 'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'เ', 'แ', 'โ', 'ใ', 'ไ', 'ๅ', '็', '่', '้', '๊', '๋', '์', ' ']
dict_idx={k:i for i,k in enumerate(index_list)}

def clean(text):
    seq = np.array([[66]+[dict_idx[i] for i in text if i]+[67]])
    _s=np.array([len(seq[0])])
    return seq,_s

n_mel_channels = 80
n_frames_per_step = 1
attention_rnn_dim = 1024
decoder_rnn_dim=1024
encoder_embedding_dim=512

def initialize_decoder_states(memory):
    B = memory.shape[0]
    MAX_TIME = memory.shape[1]

    attention_hidden = np.zeros((B, attention_rnn_dim), dtype=np.float32)
    attention_cell = np.zeros((B, attention_rnn_dim), dtype=np.float32)

    decoder_hidden = np.zeros((B, decoder_rnn_dim), dtype=np.float32)
    decoder_cell = np.zeros((B, decoder_rnn_dim), dtype=np.float32)

    attention_weights = np.zeros((B, MAX_TIME), dtype=np.float32)
    attention_weights_cum = np.zeros((B, MAX_TIME), dtype=np.float32)
    attention_context = np.zeros((B, encoder_embedding_dim), dtype=np.float32)

    return (
        attention_hidden,
        attention_cell,
        decoder_hidden,
        decoder_cell,
        attention_weights,
        attention_weights_cum,
        attention_context,
    )


def get_go_frame(memory):
    B = memory.shape[0]
    decoder_input = np.zeros((B, n_mel_channels*n_frames_per_step), dtype=np.float32)
    return decoder_input


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def parse_decoder_outputs(mel_outputs, gate_outputs, alignments):
    # (T_out, B) -> (B, T_out)
    alignments = np.stack(alignments).transpose((1, 0, 2, 3))
    # (T_out, B) -> (B, T_out)
    # Add a -1 to prevent squeezing the batch dimension in case
    # batch is 1
    gate_outputs = np.stack(gate_outputs).squeeze(-1).transpose((1, 0, 2))
    # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
    mel_outputs = np.stack(mel_outputs).transpose((1, 0, 2, 3))
    # decouple frames per step
    mel_outputs = mel_outputs.reshape(mel_outputs.shape[0], -1, n_mel_channels)
    # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
    mel_outputs = mel_outputs.transpose((0, 2, 1))

    return mel_outputs, gate_outputs, alignments


# only numpy operations
def inference(text, encoder, decoder_iter, postnet):
    sequences, sequence_lengths = clean(text)

    # print("Running Tacotron2 Encoder")
    inputs = {"seq": sequences, "seq_len": sequence_lengths}
    memory, processed_memory, _ = encoder.run(None, inputs)

    # print("Running Tacotron2 Decoder")
    mel_lengths = np.zeros([memory.shape[0]], dtype=np.int32)
    not_finished = np.ones([memory.shape[0]], dtype=np.int32)
    mel_outputs, gate_outputs, alignments = [], [], []
    gate_threshold = 0.5
    max_decoder_steps = 5000
    first_iter = True

    (
        attention_hidden,
        attention_cell,
        decoder_hidden,
        decoder_cell,
        attention_weights,
        attention_weights_cum,
        attention_context,
    ) = initialize_decoder_states(memory)

    decoder_input = get_go_frame(memory)

    while True:
        inputs = {
            "decoder_input": decoder_input,
            "attention_hidden": attention_hidden,
            "attention_cell": attention_cell,
            "decoder_hidden": decoder_hidden,
            "decoder_cell": decoder_cell,
            "attention_weights": attention_weights,
            "attention_weights_cum": attention_weights_cum,
            "attention_context": attention_context,
            "memory": memory,
            "processed_memory": processed_memory,
        }
        (
            mel_output,
            gate_output,
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
        ) = decoder_iter.run(None, inputs)

        if first_iter:
            mel_outputs = [np.expand_dims(mel_output, 2)]
            gate_outputs = [np.expand_dims(gate_output, 2)]
            alignments = [np.expand_dims(attention_weights, 2)]
            first_iter = False
        else:
            mel_outputs += [np.expand_dims(mel_output, 2)]
            gate_outputs += [np.expand_dims(gate_output, 2)]
            alignments += [np.expand_dims(attention_weights, 2)]

        dec = np.less(sigmoid(gate_output), gate_threshold)
        dec = np.squeeze(dec, axis=1)
        not_finished = not_finished * dec
        mel_lengths += not_finished

        if not_finished.sum() == 0:
            # print("Stopping after ", len(mel_outputs), " decoder steps")
            break
        if len(mel_outputs) == max_decoder_steps:
            # print("Warning! Reached max decoder steps")
            break

        decoder_input = mel_output

    mel_outputs, gate_outputs, alignments = parse_decoder_outputs(
        mel_outputs, gate_outputs, alignments
    )

    # print("Running Tacotron2 PostNet")
    inputs = {"mel_spec": mel_outputs}
    mel_outputs_postnet = postnet.run(None, inputs)

    return mel_outputs_postnet

class LunarlistONNX:
    def __init__(self) -> None:
        self.encoder = ort.InferenceSession(hf_hub_download(repo_id="pythainlp/thaitts-onnx",filename="tacotron2encoder-th.onnx"))
        self.decoder = ort.InferenceSession(hf_hub_download(repo_id="pythainlp/thaitts-onnx",filename="tacotron2decoder-th.onnx"))
        self.postnet = ort.InferenceSession(hf_hub_download(repo_id="pythainlp/thaitts-onnx",filename="tacotron2postnet-th.onnx"))
        self.hifi = ort.InferenceSession(hf_hub_download(repo_id="pythainlp/thaitts-onnx",filename="vocoder.onnx"))
    def tts(self, text: str):
        mel = inference(text, self.encoder, self.decoder, self.postnet)
        return self.hifi.run(None, {"spec": mel[0]})
    def __call__(self, text: str,return_type: str = "file", filename: str = None):
        wavs = self.tts(text)
        if return_type == "waveform":
            return wavs[0][0, 0, :]
        import soundfile as sf
        if filename != None:
            sf.write(filename, wavs[0][0, 0, :], 22050)
            return filename
        else:
            with tempfile.NamedTemporaryFile(suffix = ".wav", delete = False) as fp:
                sf.write(fp.name, wavs[0][0, 0, :], 22050)
            return fp.name