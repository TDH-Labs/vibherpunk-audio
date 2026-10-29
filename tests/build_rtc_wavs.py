"""48 kHz mono WAVs for the aiortc test peer."""
import numpy as np
import scipy.io.wavfile
from scipy.signal import resample_poly

SR = 48000


def load48(path):
    sr, data = scipy.io.wavfile.read(path)
    x = data.astype(np.float64) / 32768.0
    if sr != SR:
        x = resample_poly(x, SR, sr)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def build(name, utt, tail_s):
    speech = load48(utt)
    out = np.concatenate([speech, np.zeros(int(tail_s * SR), dtype=np.int16)])
    scipy.io.wavfile.write(f"tests/{name}", SR, out)
    print(f"WROTE tests/{name} speech {len(speech)/SR:.2f}s + {tail_s}s")


def build_two_turn():
    a = load48("tests/utt_chat1.wav")
    b = load48("tests/utt_chat2.wav")
    gap = np.zeros(int(8.0 * SR), dtype=np.int16)
    tail = np.zeros(int(6.0 * SR), dtype=np.int16)
    out = np.concatenate([a, gap, b, tail])
    scipy.io.wavfile.write("tests/rtc_chat2turn.wav", SR, out)
    print(f"WROTE tests/rtc_chat2turn.wav seg1 0-{len(a)/SR:.2f}s seg2 {((len(a)+len(gap))/SR):.2f}-{((len(a)+len(gap)+len(b))/SR):.2f}s")


if __name__ == "__main__":
    build_two_turn()
    build("rtc_chat.wav", "tests/utt_chat1.wav", 3.5)
    build("rtc_work.wav", "tests/utt_work.wav", 150.0)
    print("RTC_WAVS_DONE")
