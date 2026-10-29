"""Build the fake-mic WAVs for the mobile (headless Chrome) tests.

Chrome's --use-file-for-fake-audio-capture loops the file, so:
  mob_chat_loop.wav  chat question + 3.5 s silence   (repeats = more turns)
  mob_work_once.wav  work request + 150 s silence    (effectively one-shot)
  mob_chat_once.wav  chat question + 60 s silence    (effectively one-shot)
All 16 kHz mono int16 (Phase 1 utterances are 24 kHz Pocket renders).
"""
import numpy as np
import scipy.io.wavfile
from scipy.signal import resample_poly

SR = 16000


def load16k(path):
    sr, data = scipy.io.wavfile.read(path)
    x = data.astype(np.float64) / 32768.0
    if sr != SR:
        x = resample_poly(x, SR, sr)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def build(name, utt_path, tail_silence_s):
    speech = load16k(utt_path)
    out = np.concatenate([speech, np.zeros(int(tail_silence_s * SR), dtype=np.int16)])
    scipy.io.wavfile.write(f"tests/{name}", SR, out)
    print(f"WROTE tests/{name}: speech {len(speech)/SR:.2f}s + {tail_silence_s}s silence")


if __name__ == "__main__":
    build("mob_chat_loop.wav", "tests/utt_chat1.wav", 3.5)
    build("mob_work_once.wav", "tests/utt_work.wav", 150.0)
    build("mob_chat_once.wav", "tests/utt_chat1.wav", 60.0)
    print("WAVS_DONE")
