"""48 kHz mono RTC fixtures for the sleep/held mobile tests."""
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


def sil(secs):
    return np.zeros(int(secs * SR), dtype=np.int16)


if __name__ == "__main__":
    sleep = load48("tests/utt_sleep.wav")
    work = load48("tests/utt_work.wav")
    out = np.concatenate([sleep, sil(4.0)])
    scipy.io.wavfile.write("tests/rtc_sleep.wav", SR, out)
    print(f"WROTE tests/rtc_sleep.wav speech {len(sleep)/SR:.2f}s")
    out2 = np.concatenate([work, sil(3.0), sleep, sil(10.0)])
    scipy.io.wavfile.write("tests/rtc_work_sleep.wav", SR, out2)
    print(
        f"WROTE tests/rtc_work_sleep.wav work {len(work)/SR:.2f}s, "
        f"sleep at {(len(work) + 3.0 * SR)/SR:.2f}s"
    )
    print("RTC_SLEEP_WAVS_DONE")
